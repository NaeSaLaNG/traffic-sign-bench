"""One-command cluster entry point for the crosswalk gauntlet.

The ``valid`` label uses the official train scene pool for policy selection;
``test`` uses the official held-out test scene pool if explicitly selected.
"""
from __future__ import annotations

import argparse
import html
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from traffic_bench.policy_lab.media import encoder_available

REPO = Path(__file__).resolve().parents[2]
SPLIT_DIR = {"valid": "train", "test": "test"}


def positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=tuple(SPLIT_DIR), default="valid")
    ap.add_argument("--iterations", type=positive_int, default=10,
                    help="maximum builder rounds; budget also includes one baseline evaluation")
    ap.add_argument("--episodes-per-eval", type=positive_int, default=60)
    ap.add_argument("--work-dir", type=Path, default=REPO.parent / "policy_lab_runs",
                    help="private runs and sandboxes; must be outside the repository")
    ap.add_argument("--run-name", default=None,
                    help="unique run folder name (default: UTC timestamp)")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="existing manifest; default is data/runs/crosswalk/<train|test>/real_manifest.jsonl")
    ap.add_argument("--checkpoint", type=Path,
                    default=REPO / "checkpoints/carl/nuplan_51479_1B/model_best.pth")
    ap.add_argument("--llm-url", default="http://localhost:8000/v1",
                    help="OpenAI-compatible vLLM URL (default: http://localhost:8000/v1)")
    ap.add_argument("--model", default=None,
                    help="model name; auto-detected when the server exposes exactly one model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--judge-workers", type=positive_int, default=1)
    ap.add_argument("--judge-gpus", default=None)
    ap.add_argument("--builder-turns", type=positive_int, default=30)
    ap.add_argument("--context-chars", type=positive_int, default=80_000)
    ap.add_argument("--min-delta", type=float, default=0.04)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--no-gifs", action="store_true", help="disable per-episode GIF and MP4 files")
    ap.add_argument("--dry-run", action="store_true", help="print planned commands without running them")
    return ap.parse_args(argv)


def run_command(command: list[str], *, dry_run: bool) -> None:
    print("+ " + shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO, check=True)


def server_model(url: str, requested: str | None) -> str:
    endpoint = url.rstrip("/") + "/models"
    headers = {}
    if os.environ.get("POLICY_LAB_LLM_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['POLICY_LAB_LLM_API_KEY']}"
    try:
        with urllib.request.urlopen(urllib.request.Request(endpoint, headers=headers), timeout=10) as resp:
            models = json.load(resp).get("data", [])
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        raise SystemExit(f"Qwen API unavailable at {endpoint}: {exc}") from exc
    names = [item["id"] for item in models if isinstance(item, dict) and isinstance(item.get("id"), str)]
    if requested:
        if requested not in names:
            raise SystemExit(f"model {requested!r} not served by {endpoint}; available: {names}")
        return requested
    if len(names) != 1:
        raise SystemExit(f"expected one served model at {endpoint}; available: {names}; pass --model")
    return names[0]


def check_dependencies() -> None:
    needed = ("torch", "numpy", "metadrive", "hydra", "omegaconf", "PIL")
    missing = [module for module in needed if importlib.util.find_spec(module) is None]
    if missing:
        raise SystemExit(f"missing Python modules in {sys.executable}: {', '.join(missing)}")


def check_encoder() -> None:
    if not encoder_available():
        raise SystemExit("MP4 encoding requires ffmpeg, or both opencv-python and Pillow")


def write_gif_index(run_dir: Path) -> Path:
    """Operator-only GIF/MP4 gallery, never copied into the agent sandbox."""
    gifs = sorted(run_dir.glob("eval_*/gifs/*.gif"))
    parts = ["<!doctype html><meta charset='utf-8'><title>Crosswalk videos</title>",
             "<style>body{font:16px sans-serif;margin:2rem;background:#111;color:#eee}"
             "section{margin-bottom:3rem}img,video{width:320px;max-width:100%;margin:.5rem}"
             "a{color:#9cf}figure{display:inline-block;max-width:340px;vertical-align:top}"
             "figcaption{overflow-wrap:anywhere}</style>", "<h1>Crosswalk gauntlet videos</h1>"]
    current = None
    for gif in gifs:
        if gif.parent.parent.name != current:
            if current is not None:
                parts.append("</section>")
            current = gif.parent.parent.name
            parts.append(f"<section><h2>{html.escape(current)}</h2>")
        rel = html.escape(gif.relative_to(run_dir).as_posix(), quote=True)
        name = html.escape(gif.name)
        mp4 = gif.with_suffix(".mp4")
        if mp4.is_file():
            video = html.escape(mp4.relative_to(run_dir).as_posix(), quote=True)
            visual = f'<video controls loop muted playsinline preload="metadata" src="{video}"></video>'
            links = f'<a href="{video}">MP4</a> · <a href="{rel}">GIF</a>'
        else:
            visual = f'<a href="{rel}"><img loading="lazy" src="{rel}" alt="{name}"></a>'
            links = f'<a href="{rel}">GIF</a>'
        parts.append(f'<figure>{visual}<figcaption>{name}<br>{links}</figcaption></figure>')
    if current is not None:
        parts.append("</section>")
    else:
        parts.append("<p>No GIFs were produced. Inspect the judge logs.</p>")
    out = run_dir / "gifs.html"
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    work_dir = args.work_dir.resolve()
    if work_dir == REPO or REPO in work_dir.parents:
        raise SystemExit("--work-dir must be outside the repository")
    name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if name in (".", "..") or Path(name).name != name or not name.strip():
        raise SystemExit("--run-name must be a single non-empty directory name")
    run_dir = work_dir / "runs" / "crosswalk" / args.split / name
    sandbox = work_dir / "sandboxes" / "crosswalk" / args.split / name
    if run_dir.exists() or sandbox.exists():
        raise SystemExit(f"run already exists: {name}; choose another --run-name")

    pool_split = SPLIT_DIR[args.split]
    manifest = (args.manifest or REPO / "data/runs/crosswalk" / pool_split / "real_manifest.jsonl").resolve()
    checkpoint = args.checkpoint.resolve()
    if not args.dry_run and not checkpoint.is_file():
        raise SystemExit(f"CaRL checkpoint missing: {checkpoint}")
    if not args.dry_run and not (REPO / "third_party/CaRL/nuPlan/carl_nuplan").is_dir():
        raise SystemExit("CaRL submodule missing; run: git submodule update --init third_party/CaRL")
    if not args.dry_run:
        check_dependencies()
        if not args.no_gifs:
            check_encoder()

    model = (args.model or "<auto-detected model>") if args.dry_run else server_model(args.llm_url, args.model)
    print(f"crosswalk split={args.split} (scene pool={pool_split}), iterations={args.iterations}, "
          f"model={model}", flush=True)
    if not manifest.is_file():
        if args.manifest is not None:
            raise SystemExit(f"manifest missing: {manifest}")
        run_command([sys.executable, "-m", "traffic_bench.eval", "manifest",
                     "sign=crosswalk", f"paths.split={pool_split}"], dry_run=args.dry_run)
        if not args.dry_run and not manifest.is_file():
            raise SystemExit(f"manifest generation did not produce {manifest}")

    run_command([sys.executable, "-m", "traffic_bench.policy_lab.export_sandbox",
                 "--out", str(sandbox)], dry_run=args.dry_run)
    command = [
        sys.executable, "-m", "traffic_bench.policy_lab.gauntlet",
        "--sandbox", str(sandbox), "--run-dir", str(run_dir),
        "--checkpoint", str(checkpoint), "--device", args.device,
        "--dev-manifest", str(manifest),
        "--episodes-per-eval", str(args.episodes_per_eval),
        "--max-evals", str(args.iterations + 1),
        "--max-rounds", str(args.iterations),
        "--builder-turns", str(args.builder_turns),
        "--context-chars", str(args.context_chars),
        "--min-delta", str(args.min_delta),
        "--judge-workers", str(args.judge_workers),
        "--llm-url", args.llm_url, "--model", model,
    ]
    if args.judge_gpus:
        command += ["--judge-gpus", args.judge_gpus]
    if args.no_think:
        command.append("--no-think")
    if not args.no_gifs:
        command.append("--save-gifs")
    run_command(command, dry_run=args.dry_run)

    if not args.dry_run:
        final = json.loads((run_dir / "final.json").read_text(encoding="utf-8"))
        print(f"Progress: {run_dir / 'progress.md'}")
        print(f"Final policy: {run_dir / 'final/sandbox' / final['submitted']['policy_path']}"
              if final.get("submitted") else "No policy submitted; inspect progress.md")
        if not args.no_gifs:
            print(f"GIF/MP4 gallery: {write_gif_index(run_dir)}")
        print(f"Stop reason: {final['stop_reason']}")
        media_errors = sorted(run_dir.glob("eval_*/media_errors.log"))
        if media_errors:
            raise SystemExit("MP4 conversion failed for some GIFs; inspect "
                             + ", ".join(str(path) for path in media_errors))


if __name__ == "__main__":
    main()
