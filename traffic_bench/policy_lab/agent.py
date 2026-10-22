"""LLM agent loop that writes CaRL policies inside the sandbox.

This process is trusted: it holds the tool implementations and is the only
thing that touches both sides. The model reaches the world only through the
tools below: file access confined to the sandbox, shell commands run in the
sandbox directory, and ``evaluate``, which runs the judge in a separate
process on a snapshot of the sandbox and returns aggregate metrics only.
Model-written code runs on the host, so ``hooks.py`` checks every command and
file before it runs.

The model is served by vLLM (or anything with an OpenAI-compatible
``/v1/chat/completions`` that supports tool calls), e.g.

    vllm serve Qwen/Qwen2.5-Coder-32B-Instruct --enable-auto-tool-choice --tool-call-parser hermes

    python -m traffic_bench.policy_lab.agent \
        --sandbox $LAB/sandbox --run-dir $LAB/runs/r1 \
        --checkpoint checkpoints/carl/nuplan_51479_1B/model_best.pth \
        --dev-manifest data/runs/policy_lab/dev/manifest.jsonl \
        --llm-url http://gpu-node:8000/v1 --model Qwen/Qwen2.5-Coder-32B-Instruct
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from traffic_bench.policy_lab.sandbox_env import sandbox_env
from traffic_bench.policy_lab.hooks import HarnessHooks, public_feedback
from traffic_bench.policy_lab.media import convert_gifs

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_TOOL_OUTPUT = 12_000
MAX_FILE_BYTES = 400_000
SNAPSHOT_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".cache", ".home", ".tmp")

TOOLS = [
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "List a directory of the sandbox.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "relative to the sandbox root", "default": "."}}},
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file of the sandbox, optionally a line range.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer", "default": 1},
            "max_lines": {"type": "integer", "default": 400}},
            "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a text file in the sandbox.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]},
    }},
    {"type": "function", "function": {
        "name": "run",
        "description": "Run a shell command in the sandbox root, in the policy's environment. "
                       "The harness refuses paths outside the sandbox, network, subprocesses and "
                       "dynamic code. Returns exit code and output.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
            "timeout_s": {"type": "integer", "default": 300}},
            "required": ["command"]},
    }},
    {"type": "function", "function": {
        "name": "evaluate",
        "description": "Drive the policy through simulated episodes and return aggregate metrics. "
                       "Limited budget; check the policy with check_policy.py first.",
        "parameters": {"type": "object", "properties": {
            "policy_path": {"type": "string", "description": "e.g. policies/v2.py"}},
            "required": ["policy_path"]},
    }},
    {"type": "function", "function": {
        "name": "submit",
        "description": "Hand in the final policy and stop.",
        "parameters": {"type": "object", "properties": {
            "policy_path": {"type": "string"},
            "summary": {"type": "string", "description": "what the policy does and why"}},
            "required": ["policy_path", "summary"]},
    }},
]

SYSTEM_PROMPT = """You are a research engineer writing a driving policy in a sandboxed \
workspace. Work only through the tools. Read TASK.md first, then the CaRL wrapper \
in carl_sandbox/carl.py. Iterate: write a policy file under policies/, check it \
with `python check_policy.py <file>`, evaluate it, study the metrics, improve. \
Keep each policy version in its own file. When evaluations stop improving or the \
budget runs out, submit your best policy."""


def _clip(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n... [{len(text) - limit} characters cut] ...\n" + text[-half:]


def format_tool_call(name: str, arguments, limit: int = 500) -> str:
    """One console line: tool name plus arguments. Long strings are truncated."""
    if isinstance(arguments, str):
        body = " ".join(arguments.split())
        if len(body) > limit:
            body = body[:limit] + f"... ({len(arguments)} chars)"
        return f"{name} arguments={body!r}"
    if not isinstance(arguments, dict) or not arguments:
        return name
    parts = []
    for key, value in arguments.items():
        if isinstance(value, str):
            text = " ".join(value.split())
            raw_len = len(value)
        else:
            text = json.dumps(value, ensure_ascii=False, default=str)
            raw_len = len(text)
        if len(text) > limit:
            text = text[:limit] + f"... ({raw_len} chars)"
        parts.append(f"{key}={text!r}")
    return name + " " + " ".join(parts)


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str | None, temperature: float, max_tokens: int,
                 thinking: bool = True):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking = thinking

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float | None = None) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if not self.thinking:
            # Qwen3-style templates: skip the <think> block, far fewer output tokens per turn.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for attempt in range(5):
            req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=900) as resp:
                    return json.loads(resp.read())["choices"][0]["message"]
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:2000]
                if 400 <= exc.code < 500 and exc.code != 429:
                    # Same body gets the same answer (e.g. prompt over max_model_len); do not retry.
                    raise RuntimeError(f"LLM request rejected: HTTP {exc.code}: {detail}") from exc
                if attempt == 4:
                    raise RuntimeError(f"LLM request failed: HTTP {exc.code}: {detail}") from exc
                print(f"[agent] LLM request failed (HTTP {exc.code}: {detail[:300]}); retrying", flush=True)
                time.sleep(10 * (attempt + 1))
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == 4:
                    raise
                print(f"[agent] LLM request failed ({exc}); retrying", flush=True)
                time.sleep(10 * (attempt + 1))
        raise RuntimeError("unreachable")


@dataclass
class EvalConfig:
    judge_python: str
    dev_manifest: str
    max_episodes: int
    max_evals: int
    device: str
    scenes_root: str | None = None
    save_gifs: bool = False
    gif_window_m: float = 80.0
    judge_workers: int = 1
    judge_gpus: str | None = None


class SandboxTools:
    def __init__(self, sandbox: Path, run_dir: Path, checkpoint: Path, evalcfg: EvalConfig,
                 hooks: bool = True):
        self.root = Path(sandbox).resolve()
        self.run_dir = Path(run_dir).resolve()
        # Checks on the model's tool calls before they run (hooks.py); the sandbox as it is now is trusted.
        self.hooks = HarnessHooks(self.root, self.run_dir) if hooks else None
        self.checkpoint = Path(checkpoint).resolve()
        self.evalcfg = evalcfg
        self.evals_used = 0
        self.eval_attempts = 0
        self.judge_failures = 0
        self.submitted: dict | None = None
        self.history: list[dict] = []

    # --- path confinement -------------------------------------------------
    def _resolve(self, rel: str, must_exist: bool = True) -> Path:
        if not isinstance(rel, str) or not rel or "\x00" in rel:
            raise ValueError("path must be a non-empty string")
        candidate = (self.root / rel).resolve()  # follows symlinks the model may have created
        if candidate != self.root and self.root not in candidate.parents:
            raise PermissionError(f"{rel!r} is outside the sandbox")
        if must_exist and not candidate.exists():
            raise FileNotFoundError(rel)
        return candidate

    # --- tools ------------------------------------------------------------
    def list_dir(self, path: str = ".") -> str:
        target = self._resolve(path)
        if not target.is_dir():
            raise NotADirectoryError(path)
        lines = []
        for child in sorted(target.iterdir()):
            if child.name == "__pycache__":
                continue
            kind = "/" if child.is_dir() else f"  {child.stat().st_size} B"
            lines.append(f"{child.name}{kind}")
        return "\n".join(lines) or "(empty)"

    def read_file(self, path: str, start_line: int = 1, max_lines: int = 400) -> str:
        target = self._resolve(path)
        if not target.is_file():
            raise IsADirectoryError(path)
        if target.stat().st_size > 5 * MAX_FILE_BYTES:
            raise ValueError("file too large to read")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, int(start_line))
        chunk = lines[start - 1:start - 1 + max(1, int(max_lines))]
        body = "\n".join(f"{start + i:5d}  {line}" for i, line in enumerate(chunk))
        return body + (f"\n[{len(lines)} lines total]" if start - 1 + len(chunk) < len(lines) else "")

    def write_file(self, path: str, content: str) -> str:
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            raise ValueError("file too large")
        target = self._resolve(path, must_exist=False)
        if target.is_symlink() or target.is_dir():
            raise PermissionError(f"{path!r} is not a regular file")
        target.parent.mkdir(parents=True, exist_ok=True)
        # Re-check after mkdir: the parent chain must still resolve inside.
        self._resolve(str(target.parent.relative_to(self.root)) or ".")
        target.write_text(content, encoding="utf-8")
        return f"wrote {path} ({len(content)} chars)"

    def run(self, command: str, timeout_s: int = 300) -> str:
        timeout_s = max(5, min(int(timeout_s), 600))
        # Own process group: a timeout also kills whatever the command started in the background.
        proc = subprocess.Popen(["/bin/sh", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace", env=sandbox_env(self.root, self.checkpoint),
                                cwd=self.root, start_new_session=True)
        try:
            out, _ = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, _ = proc.communicate()
            return _clip(f"[timeout after {timeout_s}s]\n{out}")
        return _clip(f"exit code {proc.returncode}\n{out}")

    def run_judge(self, policy_path: str) -> tuple[Path, dict | None, str | None]:
        """Run the judge on a snapshot of the sandbox, spending one evaluation.

        Returns (eval_dir, feedback, error); feedback is None when the judge
        failed, and then ``error`` is the message to show the model.
        """
        policy_file = self._resolve(policy_path)
        rel = str(policy_file.relative_to(self.root))
        self.evals_used += 1
        self.eval_attempts += 1
        eval_dir = self.run_dir / f"eval_{self.eval_attempts:03d}"
        snapshot = eval_dir / "sandbox"
        shutil.copytree(self.root, snapshot, ignore=SNAPSHOT_IGNORE, symlinks=True)
        feedback_path = eval_dir / "feedback.json"
        cmd = [
            self.evalcfg.judge_python, "-m", "traffic_bench.policy_lab.judge",
            "--sandbox", str(snapshot), "--policy", rel,
            "--checkpoint", str(self.checkpoint),
            "--manifest", self.evalcfg.dev_manifest,
            "--max-episodes", str(self.evalcfg.max_episodes),
            "--device", self.evalcfg.device,
            "--private-dir", str(eval_dir / "judge"),
            "--out", str(feedback_path),
        ]
        if self.evalcfg.judge_workers > 1:
            cmd += ["--workers", str(self.evalcfg.judge_workers)]
        if self.evalcfg.judge_gpus:
            cmd += ["--gpus", self.evalcfg.judge_gpus]
        if self.evalcfg.scenes_root:
            cmd += ["--scenes-root", self.evalcfg.scenes_root]
        if self.evalcfg.save_gifs:
            # Private: next to the snapshot, never inside it, so the model cannot see them.
            cmd += ["--gif-dir", str(eval_dir / "gifs"), "--gif-window", str(self.evalcfg.gif_window_m)]
        with open(eval_dir / "judge.log", "w") as log:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
        if self.evalcfg.save_gifs:
            _, media_errors = convert_gifs(eval_dir / "gifs")
            if media_errors:
                (eval_dir / "media_errors.log").write_text("\n".join(media_errors) + "\n", encoding="utf-8")
                print(f"[policy_lab] {len(media_errors)} GIF(s) could not be converted to MP4; "
                      f"see {eval_dir / 'media_errors.log'}", flush=True)
        if proc.returncode != 0 or not feedback_path.is_file():
            # Judge internals stay private; the agent only learns it failed.
            self.evals_used -= 1
            self.judge_failures += 1
            if self.judge_failures >= 3:
                self.evals_used = self.evalcfg.max_evals
                return eval_dir, None, ("the simulator keeps failing; evaluation is unavailable, "
                                        "submit your best policy")
            return eval_dir, None, ("evaluation failed on the simulator side "
                                    "(not counted against your budget); retry later")
        feedback = json.loads(feedback_path.read_text())
        self.history.append({"eval": self.evals_used, "dir": eval_dir.name, "policy": rel,
                             "overall": feedback.get("overall")})
        return eval_dir, feedback, None

    def evaluate(self, policy_path: str) -> str:
        if self.evals_used >= self.evalcfg.max_evals:
            return "evaluation budget exhausted; submit your best policy"
        rel = str(self._resolve(policy_path).relative_to(self.root))
        try:
            _, feedback, error = self.run_judge(rel)
        except Exception as exc:
            # Do not let simulator paths or tracebacks escape via call()'s generic handler.
            self.run_dir.mkdir(parents=True, exist_ok=True)
            with open(self.run_dir / "evaluation_errors.log", "a", encoding="utf-8") as log:
                log.write(f"{type(exc).__name__}: {exc}\n")
            return "evaluation failed on the simulator side; inspect private logs"
        if feedback is None:
            return error
        left = self.evalcfg.max_evals - self.evals_used
        return _clip(
            f"evaluation #{self.evals_used} of {rel} ({left} left)\n"
            + json.dumps(public_feedback(feedback), indent=1)
        )

    def submit(self, policy_path: str, summary: str) -> str:
        policy_file = self._resolve(policy_path)
        self.submitted = {"policy_path": str(policy_file.relative_to(self.root)), "summary": summary}
        return "submitted"

    def call(self, name: str, args: dict) -> str:
        fn = {"list_dir": self.list_dir, "read_file": self.read_file, "write_file": self.write_file,
              "run": self.run, "evaluate": self.evaluate, "submit": self.submit}.get(name)
        if fn is None:
            return f"unknown tool {name!r}"
        try:
            blocked = self.hooks.before(name, args) if self.hooks else None
            return blocked or fn(**args)
        except TypeError as exc:
            return f"bad arguments for {name}: {exc}"
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"


def strip_reasoning(text: str) -> str:
    """Drop Qwen3-style ``<think>...</think>`` blocks (vLLM without --reasoning-parser)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    if "</think>" in text:  # opening tag eaten by the chat template
        text = text.split("</think>", 1)[1]
    return text.strip()


def _shrink_history(messages: list[dict], budget_chars: int) -> None:
    """Blank out the oldest tool outputs once the transcript gets too long."""
    total = sum(len(json.dumps(m)) for m in messages)
    for m in messages[2:]:
        if total <= budget_chars:
            break
        if m.get("role") == "tool" and not m.get("_shrunk"):
            total -= len(m["content"])
            m["content"] = "[old tool output removed to save context]"
            m["_shrunk"] = True
            total += len(m["content"])


def add_hook_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--no-harness-hooks", action="store_true",
                    help="do not check the model's commands and code before they run (hooks.py); "
                         "model-written code then runs on the host unchecked")


def add_speed_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--save-gifs", action="store_true",
                    help="save one GIF per episode under eval_NNN/gifs (private; slows evaluation)")
    ap.add_argument("--gif-window", type=float, default=80.0, help="GIF view window, metres")
    ap.add_argument("--judge-workers", type=int, default=1,
                    help="parallel simulator processes per evaluation (CPU-bound; about one per free core)")
    ap.add_argument("--judge-gpus", default=None,
                    help="GPU indices for the policy processes during evaluation, e.g. 2 or 1,2")
    ap.add_argument("--no-think", action="store_true",
                    help="disable the model's reasoning block (Qwen3 chat_template_kwargs.enable_thinking=false)")


def hooks_enabled(args: argparse.Namespace) -> bool:
    if args.no_harness_hooks:
        print("[policy_lab] harness hooks OFF: model-written code runs on the host unchecked", flush=True)
    return not args.no_harness_hooks


def run_agent(args: argparse.Namespace) -> dict:
    sandbox = Path(args.sandbox).resolve()
    run_dir = Path(args.run_dir).resolve()
    if run_dir == sandbox or sandbox in run_dir.parents:
        raise SystemExit("--run-dir must be outside the sandbox")
    if not (sandbox / "TASK.md").is_file():
        raise SystemExit(f"{sandbox} is not a policy-lab sandbox (no TASK.md); create it with "
                         f"`python -m traffic_bench.policy_lab.export_sandbox --out {sandbox}`")
    run_dir.mkdir(parents=True, exist_ok=True)
    tools = SandboxTools(sandbox, run_dir, Path(args.checkpoint), EvalConfig(
        judge_python=args.judge_python,
        dev_manifest=args.dev_manifest,
        max_episodes=args.episodes_per_eval,
        max_evals=args.max_evals,
        device=args.device,
        scenes_root=args.scenes_root,
        save_gifs=args.save_gifs,
        gif_window_m=args.gif_window,
        judge_workers=args.judge_workers,
        judge_gpus=args.judge_gpus,
    ), hooks=hooks_enabled(args))
    llm = LLMClient(args.llm_url, args.model, os.environ.get("POLICY_LAB_LLM_API_KEY"),
                    args.temperature, args.max_tokens, thinking=not args.no_think)

    task = (sandbox / "TASK.md").read_text(encoding="utf-8")
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{task}\n\nYou have {args.max_evals} evaluations. Begin."},
    ]
    transcript = open(run_dir / "transcript.jsonl", "a", encoding="utf-8")

    def log(entry: dict) -> None:
        transcript.write(json.dumps(entry, ensure_ascii=False) + "\n")
        transcript.flush()

    idle = 0
    for turn in range(args.max_turns):
        _shrink_history(messages, args.context_chars)
        reply = llm.chat([{k: v for k, v in m.items() if not k.startswith("_")} for m in messages], TOOLS)
        assistant = {"role": "assistant", "content": reply.get("content") or ""}
        calls = reply.get("tool_calls") or []
        if calls:
            assistant["tool_calls"] = calls
        messages.append(assistant)
        log({"turn": turn, "assistant": assistant})
        if not calls:
            idle += 1
            if idle >= 3:
                break
            messages.append({"role": "user", "content": "Continue using the tools. Call submit when done."})
            continue
        idle = 0
        for call in calls:
            name = call["function"]["name"]
            raw_args = call["function"].get("arguments") or "{}"
            try:
                call_args = json.loads(raw_args)
                if not isinstance(call_args, dict):
                    raise ValueError("arguments must be an object")
            except ValueError as exc:
                print(f"[agent] turn {turn}: {format_tool_call(name, raw_args)}", flush=True)
                result = f"could not parse arguments: {exc}"
            else:
                print(f"[agent] turn {turn}: {format_tool_call(name, call_args)}", flush=True)
                result = tools.call(name, call_args)
            messages.append({"role": "tool", "tool_call_id": call.get("id", name), "content": result})
            log({"turn": turn, "tool": name, "result": result[:2000]})
        if tools.submitted:
            break

    transcript.close()
    final = {
        "submitted": tools.submitted,
        "evaluations": tools.history,
        "evals_used": tools.evals_used,
    }
    if tools.submitted:
        final_dir = run_dir / "final"
        if final_dir.exists():
            shutil.rmtree(final_dir)
        shutil.copytree(sandbox, final_dir / "sandbox", ignore=SNAPSHOT_IGNORE, symlinks=True)
    (run_dir / "final.json").write_text(json.dumps(final, indent=2, ensure_ascii=False))
    return final


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sandbox", required=True)
    ap.add_argument("--run-dir", required=True, help="private transcripts and eval results (outside the sandbox)")
    add_hook_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cpu", help="torch device for CaRL in the policy process")
    ap.add_argument("--dev-manifest", required=True, help="manifest the agent is evaluated on")
    ap.add_argument("--scenes-root", default=None)
    ap.add_argument("--episodes-per-eval", type=int, default=60)
    ap.add_argument("--max-evals", type=int, default=20)
    ap.add_argument("--max-turns", type=int, default=200)
    ap.add_argument("--judge-python", default=sys.executable,
                    help="interpreter with the simulator installed")
    ap.add_argument("--llm-url", required=True, help="OpenAI-compatible base URL, e.g. http://host:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--context-chars", type=int, default=80_000,
                    help="approximate transcript budget before old tool outputs are dropped "
                         "(~3 chars per token: 80k fits max_model_len 32768, use ~3x max_model_len)")
    add_speed_args(ap)
    args = ap.parse_args()
    final = run_agent(args)
    print(json.dumps(final, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
