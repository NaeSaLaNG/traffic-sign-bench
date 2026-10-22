"""Gauntlet loop: a builder writes policies, a blind critic judges them against the champion.

Same sandbox, tools and judge as ``agent.py``, but the roles are split:

* the **builder** (fresh conversation every round) writes one new policy file
  and proposes it; it cannot evaluate or submit anything, so it never grades
  its own work;
* the **orchestrator** (this process) checks the file with ``check_policy.py``,
  evaluates it on the dev manifest and keeps the champion, which starts as the
  reference policy (plain CaRL);
* the **critic** (fresh, tool-less call) sees TASK.md plus two policies, A and B
  in random order, each with its source and aggregate metrics. It does not know
  which is the champion. It picks the better one and names the single biggest
  gap; that gap is the brief for the next builder round.

The critic sees exactly what the builder may see (code and per-family rates),
so its gap text cannot leak the private judge output.

    python -m traffic_bench.policy_lab.gauntlet \
        --sandbox $LAB/sandbox --run-dir $LAB/runs/g1 \
        --checkpoint checkpoints/carl/nuplan_51479_1B/model_best.pth \
        --dev-manifest data/runs/policy_lab/dev/manifest.jsonl \
        --llm-url http://gpu-node:8000/v1 --model Qwen3.8-27B-FP8
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from traffic_bench.policy_lab.agent import (
    SNAPSHOT_IGNORE,
    TOOLS,
    EvalConfig,
    LLMClient,
    RepeatGuard,
    SandboxTools,
    _clip,
    _shrink_history,
    assistant_turn,
    compact_call_args,
    format_tool_call,
    add_hook_args,
    hooks_enabled,
    add_speed_args,
    strip_reasoning,
)

MAX_POLICY_CHARS = 20_000

BUILDER_TOOLS = [t for t in TOOLS if t["function"]["name"] in ("list_dir", "read_file", "write_file", "run")] + [
    {"type": "function", "function": {
        "name": "propose",
        "description": "Hand the new policy file to the gauntlet. It is checked with check_policy.py, "
                       "evaluated and judged against the current champion by someone else. "
                       "This ends your round.",
        "parameters": {"type": "object", "properties": {
            "policy_path": {"type": "string", "description": "e.g. policies/r03_stop_plates.py"}},
            "required": ["policy_path"]},
    }},
]

BUILDER_PROMPT = """You are the builder in a gauntlet: you improve a driving policy in a \
sandboxed workspace, working only through the tools. You do not evaluate your own work: \
when you call `propose`, the policy is evaluated in the simulator and a separate, harsh \
critic compares it blind against the current champion. Your job this round is to close \
the gap named in the brief. Make one focused, well-reasoned change on top of the champion \
(or a rewrite if the brief calls for it), save it as a NEW file, test it locally with \
`python check_policy.py <file>` and any small experiments you like, then call `propose`.

Before every tool call, write one or two sentences: what the last result showed and what \
you will do next. Do not re-run a command or rewrite a file without changing the code \
first: the same files give the same output, and the harness refuses exact repeats."""

CRITIC_PROMPT = """You are a harsh, independent reviewer of driving policies. You never see \
how they were written, only the artifacts: the source of two policies, A and B, and the \
aggregate metrics each got on the same set of simulated episodes. The main score is \
clean_success_rate (reached the destination with no collision and no traffic violation); \
policy faults (crashes, timeouts) are failures. Judge which policy is better, considering \
the metrics first and then whether the code is sound and likely to generalise to unseen \
episodes. Then name the ONE biggest meaningful gap that still separates the better policy \
from a careful, law-abiding driver: a concrete change in behaviour or code, tied to the \
sign families and metrics where it shows. Be specific and blunt; no praise.

Answer with a single JSON object and nothing else:
{"winner": "A" | "B" | "tie", "reason": "<2-4 sentences>", "biggest_gap": "<one concrete gap and how to close it>"}"""


@dataclass
class Entry:
    """One evaluated policy: what the builder and the critic are allowed to see about it."""
    path: str
    eval_dir: Path
    feedback: dict

    @property
    def source(self) -> str:
        # Read from the eval snapshot so later edits in the live sandbox do not change it.
        f = self.eval_dir / "sandbox" / self.path
        return f.read_text(encoding="utf-8", errors="replace") if f.is_file() else "(missing)"

    def overall(self, key: str, default: float = 0.0) -> float:
        return float(self.feedback.get("overall", {}).get(key, default))

    def public_metrics(self) -> dict:
        keep = ("by_family", "overall", "policy_fault_examples", "policy_load_error")
        return {k: self.feedback[k] for k in keep if k in self.feedback}


@dataclass
class Gauntlet:
    tools: SandboxTools
    builder_llm: LLMClient
    critic_llm: LLMClient
    task: str
    run_dir: Path
    args: argparse.Namespace
    rng: random.Random
    rounds: list[dict] = field(default_factory=list)
    champion: Entry | None = None
    gap: str | None = None
    stop_reason: str = ""

    # --- logging ----------------------------------------------------------
    def log(self, entry: dict) -> None:
        with open(self.run_dir / "gauntlet.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def write_progress(self) -> None:
        c = self.champion
        lines = ["# Gauntlet progress", ""]
        if c:
            lines += [f"**Champion:** `{c.path}` ({c.eval_dir.name}), "
                      f"clean_success_rate {c.overall('clean_success_rate'):.3f}, "
                      f"policy_fault_rate {c.overall('policy_fault_rate'):.3f}", ""]
        lines += [f"Evaluations used: {self.tools.evals_used} / {self.args.max_evals}", ""]
        if self.stop_reason:
            lines += [f"**Stopped:** {self.stop_reason}", ""]
        if self.gap:
            lines += ["**Current biggest gap:**", "", self.gap, ""]
        lines += ["| round | candidate | clean | Δ vs champion | critic | verdict |",
                  "|---|---|---|---|---|---|"]
        for r in self.rounds:
            clean = "" if r.get("clean") is None else f"{r['clean']:.3f}"
            delta = "" if r.get("delta") is None else f"{r['delta']:+.3f}"
            lines.append(f"| {r['round']} | `{r.get('candidate') or '-'}` | {clean} | {delta} | "
                         f"{r.get('critic_pick') or '-'} | {r['verdict']} |")
        (self.run_dir / "progress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- lead: what to focus on ---------------------------------------------
    def focus_family(self) -> str | None:
        fams = self.champion.feedback.get("by_family", {}) if self.champion else {}
        scored = [((1.0 - float(m.get("clean_success_rate", 0.0))) * int(m.get("episodes", 0)), name)
                  for name, m in fams.items() if m.get("episodes")]
        scored = [s for s in scored if s[0] > 0]
        return max(scored)[1] if scored else None

    def brief(self, rnd: int) -> str:
        c = self.champion
        parts = [
            self.task,
            "",
            f"## Gauntlet round {rnd}",
            f"Current champion: `{c.path}` (read it with read_file; it is the bar you must beat).",
            "Its metrics on the dev episodes:",
            "```json\n" + json.dumps(c.public_metrics(), indent=1) + "\n```",
        ]
        if self.gap:
            parts += ["", "The critic's biggest remaining gap (fix this first):", self.gap]
        focus = self.focus_family()
        if focus:
            parts += ["", f"The sign family with the most lost clean episodes is `{focus}`."]
        if self.rounds:
            parts += ["", "Earlier rounds:"]
            for r in self.rounds[-10:]:
                clean = "n/a" if r.get("clean") is None else f"{r['clean']:.3f}"
                parts.append(f"- round {r['round']}: `{r.get('candidate') or '-'}` clean {clean} -> {r['verdict']}")
        parts += ["", f"Write your policy as a new file `policies/r{rnd:02d}_<short_name>.py` "
                      "(do not overwrite older files), then call `propose`.",
                  f"You have {self.args.builder_turns} tool turns this round; a round that ends without "
                  "`propose` is wasted. Spend at most a third of them reading, then write, check and propose."]
        return "\n".join(parts)

    # --- builder ----------------------------------------------------------
    def build(self, rnd: int) -> tuple[str | None, str]:
        """Run one builder conversation. Returns (policy_path, note)."""
        messages: list[dict] = [
            {"role": "system", "content": BUILDER_PROMPT},
            {"role": "user", "content": self.brief(rnd)},
        ]
        transcript = open(self.run_dir / f"round_{rnd:02d}_builder.jsonl", "a", encoding="utf-8")
        check_failures = 0
        idle = 0
        wrote = False
        last_policy: str | None = None   # newest policies/ file written this round
        guard = RepeatGuard(self.tools.root)
        hot_turns = 0                    # turns left at --loop-temperature after a repeat
        try:
            for turn in range(self.args.builder_turns):
                left = self.args.builder_turns - turn
                if left in (self.args.builder_turns * 2 // 3, self.args.builder_turns // 3, 3) and turn > 0:
                    nudge = f"{left} tool turns left in this round."
                    if not wrote:
                        nudge += f" You have not written a policy yet: write policies/r{rnd:02d}_<name>.py now."
                    messages.append({"role": "user", "content": nudge + " Call `propose` before you run out."})
                _shrink_history(messages, self.args.context_chars)
                try:
                    reply = self.builder_llm.chat([{k: v for k, v in m.items() if not k.startswith("_")}
                                                   for m in messages], BUILDER_TOOLS,
                                                  temperature=self.args.loop_temperature if hot_turns else None)
                except RuntimeError as exc:
                    # A rejected request (e.g. context overflow) ends this round, not the whole gauntlet.
                    print(f"[gauntlet] round {rnd} turn {turn}: {exc}", flush=True)
                    return None, f"builder request rejected: {str(exc)[:300]}"
                hot_turns = max(0, hot_turns - 1)
                assistant, reasoning = assistant_turn(reply)
                calls = assistant.get("tool_calls") or []
                messages.append(assistant)
                transcript.write(json.dumps({"turn": turn, "assistant": assistant,
                                             **({"reasoning": reasoning} if reasoning else {})},
                                            ensure_ascii=False) + "\n")
                if not calls:
                    idle += 1
                    if idle >= 3:
                        return None, "builder stopped using tools"
                    messages.append({"role": "user", "content": "Continue using the tools; call propose when done."})
                    continue
                idle = 0
                proposed = None
                for call in calls:
                    name = call["function"]["name"]
                    raw_args = call["function"].get("arguments") or "{}"
                    try:
                        call_args = json.loads(raw_args)
                        if not isinstance(call_args, dict):
                            raise ValueError("arguments must be an object")
                    except ValueError as exc:
                        print(
                            f"[gauntlet] round {rnd} turn {turn}: {format_tool_call(name, raw_args)}",
                            flush=True,
                        )
                        result = f"could not parse arguments: {exc}"
                    else:
                        print(
                            f"[gauntlet] round {rnd} turn {turn}: {format_tool_call(name, call_args)}",
                            flush=True,
                        )
                        if name == "propose":
                            result, proposed = self._gate(call_args.get("policy_path"))
                            if proposed is None:
                                check_failures += 1
                                if check_failures > self.args.check_retries:
                                    return None, "policy failed check_policy.py too many times"
                        elif name in ("list_dir", "read_file", "write_file", "run"):
                            key, refusal = guard.check(name, call_args)
                            if refusal is None:
                                result = self.tools.call(name, call_args)
                                guard.remember(key, turn, result)
                                if name == "write_file" and result.startswith("wrote"):
                                    wrote = True
                                    path = str(call_args.get("path", ""))
                                    if path.startswith("policies/") and path.endswith(".py"):
                                        last_policy = path
                            else:
                                left = self.args.max_repeats - guard.repeats
                                result = refusal + (f" ({left} more repeats end this round.)" if left > 0 else "")
                                compact_call_args(call, "; a repeat, not run")
                                hot_turns = 3
                                print(f"[gauntlet] round {rnd} turn {turn}: repeat #{guard.repeats} refused",
                                      flush=True)
                        else:
                            result = f"unknown tool {name!r}"
                    messages.append({"role": "tool", "tool_call_id": call.get("id", name), "content": result})
                    transcript.write(json.dumps({"turn": turn, "tool": name, "result": result[:2000]},
                                                ensure_ascii=False) + "\n")
                    if proposed:
                        return proposed, "proposed"
                if guard.repeats >= self.args.max_repeats:
                    return self._end_stuck_round(rnd, last_policy)
            return None, "builder ran out of turns"
        finally:
            transcript.close()

    def _end_stuck_round(self, rnd: int, last_policy: str | None) -> tuple[str | None, str]:
        """The builder keeps repeating itself: propose what it wrote, if anything passes the check."""
        note = f"builder repeated identical calls {self.args.max_repeats} times"
        if last_policy is None:
            return None, note + "; no policy written"
        result, proposed = self._gate(last_policy)
        print(f"[gauntlet] round {rnd}: {note}; auto-proposing {last_policy}: {result[:200]}", flush=True)
        if proposed is None:
            return None, note + f"; {last_policy} failed the check"
        return proposed, note + f"; auto-proposed {last_policy}"

    def _gate(self, policy_path) -> tuple[str, str | None]:
        """check_policy.py, run by the orchestrator, not the builder."""
        try:
            rel = str(self.tools._resolve(policy_path).relative_to(self.tools.root))
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}", None
        if not rel.endswith(".py"):
            return "policy_path must be a .py file", None
        if self.champion and rel == self.champion.path:
            return "that is the champion itself; propose a new file", None
        blocked = self.tools.hooks.before("gate", {"policy_path": rel}) if self.tools.hooks else None
        if blocked:
            return blocked, None
        out = self.tools.run(f"python check_policy.py {_shell_quote(rel)}", timeout_s=600)
        if not out.startswith("exit code 0"):
            return "check_policy.py failed; fix the policy and propose again:\n" + out, None
        return "accepted for evaluation", rel

    # --- critic -----------------------------------------------------------
    def critique(self, rnd: int, candidate: Entry) -> dict:
        """Blind A/B: the critic is not told which policy is the champion."""
        pair = [("champion", self.champion), ("candidate", candidate)]
        self.rng.shuffle(pair)
        labels = {"A": pair[0][0], "B": pair[1][0]}
        body = [self.task, ""]
        for label, (_, entry) in zip("AB", pair):
            body += [f"## Policy {label}",
                     "```python\n" + _clip(entry.source, MAX_POLICY_CHARS) + "\n```",
                     f"Metrics of policy {label}:",
                     "```json\n" + json.dumps(entry.public_metrics(), indent=1) + "\n```", ""]
        body.append("Which policy is better, and what is the biggest remaining gap? Reply with the JSON object only.")
        messages = [{"role": "system", "content": CRITIC_PROMPT}, {"role": "user", "content": "\n".join(body)}]
        verdict = None
        raw = ""
        for _ in range(2):
            reply = self.critic_llm.chat(messages, temperature=self.args.critic_temperature)
            raw = reply.get("content") or ""
            verdict = parse_verdict(raw)
            if verdict:
                break
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": "Reply with exactly one JSON object with keys "
                                                     "winner, reason, biggest_gap."}]
        with open(self.run_dir / f"round_{rnd:02d}_critic.json", "w", encoding="utf-8") as f:
            json.dump({"labels": labels, "prompt": messages[1]["content"], "raw": raw, "verdict": verdict},
                      f, ensure_ascii=False, indent=1)
        verdict = verdict or {"winner": "tie", "reason": "critic reply unparsable", "biggest_gap": ""}
        verdict["pick"] = labels.get(verdict["winner"], "tie")
        return verdict

    # --- decision ---------------------------------------------------------
    def decide(self, candidate: Entry, critic_pick: str) -> tuple[str, float]:
        delta = candidate.overall("clean_success_rate") - self.champion.overall("clean_success_rate")
        faults_worse = candidate.overall("policy_fault_rate") > self.champion.overall("policy_fault_rate")
        if delta > self.args.min_delta and not faults_worse:
            return "win", delta
        if delta < -self.args.min_delta or faults_worse:
            return "loss", delta
        return ("win" if critic_pick == "candidate" else "loss"), delta

    # --- main loop --------------------------------------------------------
    def evaluate(self, rel: str) -> Entry | None:
        eval_dir, feedback, error = self.tools.run_judge(rel)
        if feedback is None:
            print(f"[gauntlet] judge failed for {rel}: {error}", flush=True)
            return None
        return Entry(rel, eval_dir, feedback)

    def budget_left(self) -> int:
        return self.args.max_evals - self.tools.evals_used

    def run(self) -> None:
        while self.champion is None:
            if self.budget_left() <= 0:
                self.stop_reason = "could not evaluate the reference policy"
                return
            self.champion = self.evaluate(self.args.reference)
        self.log({"event": "reference", "policy": self.champion.path, "eval_dir": self.champion.eval_dir.name,
                  "overall": self.champion.feedback.get("overall")})
        self.write_progress()
        since_win = 0
        for rnd in range(1, self.args.max_rounds + 1):
            if self.champion.overall("clean_success_rate") >= self.args.target_clean:
                self.stop_reason = f"champion reached clean_success_rate >= {self.args.target_clean}"
                break
            if self.budget_left() <= 0:
                self.stop_reason = "evaluation budget exhausted"
                break
            if since_win >= self.args.patience:
                self.stop_reason = f"no new champion for {self.args.patience} rounds"
                break
            record: dict = {"round": rnd, "champion": self.champion.path}
            rel, note = self.build(rnd)
            record["candidate"] = rel
            candidate = None
            while rel and candidate is None and self.budget_left() > 0:
                candidate = self.evaluate(rel)
                if self.tools.judge_failures >= 3:
                    break
            if candidate is None:
                record.update(verdict="no candidate", note=note if not rel else "evaluation failed")
                since_win += 1
            else:
                verdict = self.critique(rnd, candidate)
                outcome, delta = self.decide(candidate, verdict["pick"])
                record.update(eval_dir=candidate.eval_dir.name, clean=candidate.overall("clean_success_rate"),
                              delta=round(delta, 3), critic_pick=verdict["pick"], verdict=outcome,
                              reason=verdict.get("reason"), gap=verdict.get("biggest_gap"),
                              overall=candidate.feedback.get("overall"))
                if verdict.get("biggest_gap"):
                    self.gap = verdict["biggest_gap"]
                if outcome == "win":
                    self.champion = candidate
                    since_win = 0
                else:
                    since_win += 1
            self.rounds.append(record)
            self.log(record)
            self.write_progress()
            print(f"[gauntlet] round {rnd}: {record['verdict']} "
                  f"(champion {self.champion.path}, {self.champion.overall('clean_success_rate'):.3f})", flush=True)
        else:
            self.stop_reason = f"reached {self.args.max_rounds} rounds"
        if not self.stop_reason and self.budget_left() <= 0:
            self.stop_reason = "evaluation budget exhausted"


def parse_verdict(text: str) -> dict | None:
    """Last JSON object in the reply with a valid ``winner``."""
    text = strip_reasoning(text)
    text = re.sub(r"```(?:json)?", "", text)
    decoder = json.JSONDecoder()
    found = None
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except ValueError:
            continue
        if isinstance(obj, dict) and str(obj.get("winner", "")).strip().upper() in ("A", "B", "TIE"):
            found = obj
    if found is None:
        return None
    winner = str(found["winner"]).strip().upper()
    return {"winner": "tie" if winner == "TIE" else winner,
            "reason": str(found.get("reason", "")),
            "biggest_gap": str(found.get("biggest_gap", ""))}


def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def run_gauntlet(args: argparse.Namespace) -> dict:
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
        gif_format=args.gif_format,
        judge_workers=args.judge_workers,
        judge_gpus=args.judge_gpus,
    ), hooks=hooks_enabled(args))
    api_key = os.environ.get("POLICY_LAB_LLM_API_KEY")
    builder_llm = LLMClient(args.llm_url, args.model, api_key, args.temperature, args.max_tokens,
                            thinking=not args.no_think)
    critic_llm = LLMClient(args.critic_url or args.llm_url, args.critic_model or args.model, api_key,
                           args.critic_temperature, args.max_tokens)
    g = Gauntlet(
        tools=tools, builder_llm=builder_llm, critic_llm=critic_llm,
        task=(sandbox / "TASK.md").read_text(encoding="utf-8"),
        run_dir=run_dir, args=args, rng=random.Random(args.seed),
    )
    g.run()
    g.write_progress()

    final = {
        "submitted": None,
        "stop_reason": g.stop_reason,
        "rounds": g.rounds,
        "evaluations": tools.history,
        "evals_used": tools.evals_used,
    }
    if g.champion is not None:
        final["submitted"] = {
            "policy_path": g.champion.path,
            "summary": f"gauntlet champion from {g.champion.eval_dir.name}",
            "eval_dir": g.champion.eval_dir.name,
            "overall": g.champion.feedback.get("overall"),
        }
        final_dir = run_dir / "final"
        if final_dir.exists():
            shutil.rmtree(final_dir)
        shutil.copytree(g.champion.eval_dir / "sandbox", final_dir / "sandbox", ignore=SNAPSHOT_IGNORE, symlinks=True)
    (run_dir / "final.json").write_text(json.dumps(final, indent=2, ensure_ascii=False, default=str))
    return final


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sandbox", required=True)
    ap.add_argument("--run-dir", required=True, help="private transcripts and eval results (outside the sandbox)")
    add_hook_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cpu", help="torch device for CaRL in the policy process")
    ap.add_argument("--dev-manifest", required=True, help="manifest the policies are evaluated on")
    ap.add_argument("--scenes-root", default=None)
    ap.add_argument("--episodes-per-eval", type=int, default=60)
    ap.add_argument("--max-evals", type=int, default=20, help="includes the reference evaluation")
    ap.add_argument("--judge-python", default=sys.executable, help="interpreter with the simulator installed")
    ap.add_argument("--reference", default="policies/carl_baseline.py", help="initial champion (the bar)")
    ap.add_argument("--max-rounds", type=int, default=50)
    ap.add_argument("--builder-turns", type=int, default=60, help="tool-call turns per builder round")
    ap.add_argument("--check-retries", type=int, default=3, help="failed check_policy.py proposals per round")
    ap.add_argument("--max-repeats", type=int, default=4,
                    help="refused identical tool calls (same arguments, same files) before the round ends; "
                         "the newest policies/ file written that round is then auto-proposed")
    ap.add_argument("--loop-temperature", type=float, default=0.8,
                    help="builder temperature for the 3 turns after a refused repeat")
    ap.add_argument("--min-delta", type=float, default=0.02,
                    help="clean_success_rate margin that decides without the critic")
    ap.add_argument("--patience", type=int, default=6, help="stop after this many rounds without a new champion")
    ap.add_argument("--target-clean", type=float, default=1.0, help="stop once the champion reaches this")
    ap.add_argument("--seed", type=int, default=0, help="A/B order of the blind comparison")
    ap.add_argument("--llm-url", required=True, help="OpenAI-compatible base URL, e.g. http://host:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--critic-url", default=None, help="defaults to --llm-url")
    ap.add_argument("--critic-model", default=None, help="defaults to --model")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--critic-temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--context-chars", type=int, default=80_000,
                    help="approximate builder transcript budget before old tool outputs are dropped "
                         "(~3 chars per token: 80k fits max_model_len 32768, use ~3x max_model_len)")
    add_speed_args(ap)
    args = ap.parse_args()
    final = run_gauntlet(args)
    print(json.dumps({k: final[k] for k in ("submitted", "stop_reason", "evals_used")}, indent=2,
                     ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
