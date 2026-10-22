"""Evaluate a sandboxed policy and reduce the results to aggregate feedback.

Runs on the trusted side with the full repository. Per-episode results (with
violation timelines) go to ``--private-dir`` and never reach the agent; the
``--out`` JSON holds only per-family rates, which is all the agent sees.

    python -m traffic_bench.policy_lab.judge \
        --sandbox $LAB/sandbox --policy policies/v3.py \
        --checkpoint checkpoints/carl/nuplan_51479_1B/model_best.pth \
        --manifest data/runs/policy_lab/dev/manifest.jsonl --max-episodes 60 \
        --private-dir $LAB/judge/run_007 --out $LAB/judge/run_007/feedback.json

``--workers N`` splits the episodes between N judge processes, each with its
own simulator and policy process (MetaDrive and SUMO run on the CPU, so this
is what speeds an evaluation up); ``--gpus 2`` or ``--gpus 1,2`` pins their
policy processes to those cards. The feedback is the same as with one worker.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf

from traffic_bench.eval.run.episode import run_one_episode
from traffic_bench.eval.run.main import _load_rows, _scenes_root
from traffic_bench.eval.sign_registry import resolve_sign_token
from traffic_bench.policy_lab.remote_policy import PolicyProcess, RemoteCarlPolicy, SandboxLaunch

POLICY_TYPE = "remote_carl"


def sign_family(sign_type) -> str:
    try:
        return resolve_sign_token(str(sign_type)).family
    except Exception:
        return "other"


def sample_rows(rows: list[dict], max_episodes: int | None, seed: int) -> list[dict]:
    """Round-robin over sign codes so every family shows up in small budgets."""
    if not max_episodes or max_episodes >= len(rows):
        return rows
    by_sign: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_sign[str(row.get("_sign_code"))].append(row)
    rng = random.Random(seed)
    for group in by_sign.values():
        rng.shuffle(group)
    picked: list[dict] = []
    keys = sorted(by_sign)
    while len(picked) < max_episodes:
        progressed = False
        for key in keys:
            if by_sign[key] and len(picked) < max_episodes:
                picked.append(by_sign[key].pop())
                progressed = True
        if not progressed:
            break
    return picked


def summarize(episodes: list[dict]) -> dict:
    """Aggregate-only feedback. Policy faults count as failed episodes;
    scenes the simulator itself rejected (bad route) are left out."""
    groups: dict[str, list[dict]] = defaultdict(list)
    invalid_scenes = 0
    for ep in episodes:
        if not ep.get("ok") and not ep.get("policy_fault"):
            invalid_scenes += 1
            continue
        groups[ep["family"]].append(ep)
        groups["ALL"].append(ep)

    def rates(runs: list[dict]) -> dict:
        n = len(runs)
        if n == 0:
            return {"episodes": 0}

        def frac(pred) -> float:
            return round(sum(1 for r in runs if pred(r)) / n, 3)

        def no_violation(r) -> bool:
            # Crosswalk and traffic-light infractions are counted apart from sign_violations.
            return (int(r.get("sign_violations", 0)) == 0
                    and int(r.get("traffic_light_violations", 0)) == 0
                    and int(r.get("crosswalk_violations", 0)) == 0)

        def clean(r) -> bool:
            return r.get("ok") and not r.get("policy_fault") and r.get("success") and no_violation(r)

        completion = [float(r.get("route_completion_pct") or 0.0) for r in runs if r.get("ok")]
        return {
            "episodes": n,
            "clean_success_rate": frac(clean),
            "success_rate": frac(lambda r: r.get("ok") and not r.get("policy_fault") and r.get("success")),
            "no_sign_violation_rate": frac(lambda r: r.get("ok") and not r.get("policy_fault") and no_violation(r)),
            "collision_rate": frac(lambda r: bool(r.get("crashed"))),
            "out_of_road_rate": frac(lambda r: bool(r.get("out_of_road"))),
            "policy_fault_rate": frac(lambda r: bool(r.get("policy_fault"))),
            "mean_route_completion_pct": round(sum(completion) / n, 1) if n else 0.0,
        }

    return {
        "by_family": {k: rates(v) for k, v in sorted(groups.items()) if k != "ALL"},
        "overall": rates(groups["ALL"]),
        "invalid_scenes_skipped": invalid_scenes,
    }


def evaluate(
    launch: SandboxLaunch,
    manifest: str,
    private_dir: Path,
    scenes_root: str | None = None,
    max_episodes: int | None = None,
    max_steps: int = 1500,
    seed: int = 0,
    gif_dir: Path | None = None,
    gif_window_m: float = 80.0,
    shard: int = 0,
    num_shards: int = 1,
) -> dict:
    cfg = OmegaConf.create({"manifest": manifest, "scenes_root": scenes_root})
    rows, manifest_path = _load_rows(cfg)
    rows = sample_rows(rows, max_episodes, seed)[shard::num_shards]
    scenes = _scenes_root(cfg, manifest_path)
    private_dir = Path(private_dir)
    private_dir.mkdir(parents=True, exist_ok=True)
    if gif_dir is not None:
        Path(gif_dir).mkdir(parents=True, exist_ok=True)

    process = PolicyProcess(launch, log_path=private_dir / "policy.log")
    RemoteCarlPolicy.process = process
    episodes: list[dict] = []
    load_error = None
    t0 = time.time()
    try:
        with open(private_dir / "episodes.jsonl", "w", encoding="utf-8") as handle:
            for idx, row in enumerate(rows):
                if not process.alive():
                    try:
                        process.start()
                    except Exception as exc:
                        load_error = f"{type(exc).__name__}: {exc}"
                        break
                RemoteCarlPolicy.episode_info = {"episode_index": idx}
                RemoteCarlPolicy.last_fault = None
                gif_path = None
                if gif_dir is not None:
                    seed_val = int(row.get("seed") or row.get("deterministic_seed") or 0)
                    gif_path = Path(gif_dir) / (f"{row.get('scene_id') or 'scene'}_v{int(row.get('var_idx', 0) or 0)}"
                                                f"_s{seed_val}_{Path(launch.policy_path).stem}.gif")
                try:
                    result = run_one_episode(
                        row=row,
                        policy_type=POLICY_TYPE,
                        models={"policy_cls": RemoteCarlPolicy},
                        scenes_root=scenes,
                        max_steps=max_steps,
                        ego_variant="default",
                        ego_sample_seed_base=42,
                        save_gif=gif_path,
                        gif_window_m=gif_window_m,
                    )
                except Exception as exc:
                    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    if RemoteCarlPolicy.last_fault is None:
                        result["judge_error"] = True
                result["family"] = sign_family(row.get("_sign_code"))
                result["sign_type"] = row.get("_sign_code")
                result["policy_fault"] = RemoteCarlPolicy.last_fault
                if result["policy_fault"]:
                    result["success"] = False
                episodes.append(result)
                handle.write(json.dumps(result, default=str) + "\n")
                handle.flush()
    finally:
        process.stop()
        RemoteCarlPolicy.process = None

    if load_error is not None:
        # Remaining rows were never driven: score them as faults.
        with open(private_dir / "episodes.jsonl", "a", encoding="utf-8") as handle:
            for row in rows[len(episodes):]:
                episodes.append({"ok": False, "policy_fault": "policy failed to start",
                                 "family": sign_family(row.get("_sign_code"))})
                handle.write(json.dumps(episodes[-1]) + "\n")
    return build_feedback(episodes, launch.policy_path, time.time() - t0, load_error)


def build_feedback(episodes: list[dict], policy_path: str, wall_time_s: float, load_error: str | None) -> dict:
    feedback = summarize(episodes)
    feedback["policy"] = policy_path
    feedback["wall_time_s"] = round(wall_time_s, 1)
    if load_error is not None:
        feedback["policy_load_error"] = load_error
    faults = []
    for e in episodes:
        msg = e.get("policy_fault")
        if msg and msg not in faults:
            faults.append(msg)
    if faults:
        # The policy's own errors (timeouts, bad actions, its tracebacks).
        feedback["policy_fault_examples"] = [m[-800:] for m in faults[:3]]
    judge_errors = sum(1 for e in episodes if e.get("judge_error"))
    if judge_errors:
        # Simulator-side exceptions: details stay in episodes.jsonl.
        feedback["judge_errors"] = judge_errors
    return feedback


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sandbox", required=True, type=Path)
    ap.add_argument("--policy", required=True, help="policy file relative to the sandbox")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--step-timeout", type=float, default=5.0)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--scenes-root", default=None)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--private-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="aggregate feedback JSON")
    ap.add_argument("--gif-dir", type=Path, default=None, help="save one GIF per episode here")
    ap.add_argument("--gif-window", type=float, default=80.0, help="GIF view window, metres")
    ap.add_argument("--workers", type=int, default=1, help="parallel judge processes")
    ap.add_argument("--gpus", default=None,
                    help="comma-separated GPU indices for the policy processes, round-robin over workers")
    ap.add_argument("--shard", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.workers > 1:
        feedback = run_parallel(args, argv if argv is not None else sys.argv[1:])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(feedback, indent=2), encoding="utf-8")
        print(json.dumps(feedback["overall"]))
        return
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus.split(",")[0]

    launch = SandboxLaunch(
        sandbox_dir=args.sandbox,
        policy_path=args.policy,
        checkpoint=args.checkpoint,
        device=args.device,
        step_timeout_s=args.step_timeout,
    )
    feedback = evaluate(
        launch,
        manifest=args.manifest,
        private_dir=args.private_dir,
        scenes_root=args.scenes_root,
        max_episodes=args.max_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        gif_dir=args.gif_dir,
        gif_window_m=args.gif_window,
        shard=args.shard,
        num_shards=args.num_shards,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(feedback, indent=2), encoding="utf-8")
    print(json.dumps(feedback["overall"]))


def _drop_options(argv: list[str], names: tuple[str, ...]) -> list[str]:
    out, skip = [], False
    for tok in argv:
        if skip:
            skip = False
            continue
        if tok in names:
            skip = True
            continue
        if any(tok.startswith(n + "=") for n in names):
            continue
        out.append(tok)
    return out


def run_parallel(args: argparse.Namespace, argv: list[str]) -> dict:
    """Run ``--workers`` single-worker judges on interleaved slices of the
    episodes and merge their results into one feedback."""
    t0 = time.time()
    private_dir = Path(args.private_dir)
    private_dir.mkdir(parents=True, exist_ok=True)
    gpus = [g.strip() for g in (args.gpus or "").split(",") if g.strip()]
    base = _drop_options(argv, ("--workers", "--gpus", "--private-dir", "--out", "--shard", "--num-shards"))
    # Each worker gets an even share of the cores instead of every one using all of them.
    threads = str(max(1, (os.cpu_count() or 1) // args.workers))
    procs = []
    for k in range(args.workers):
        shard_dir = private_dir / f"w{k:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", threads)
        env.setdefault("MKL_NUM_THREADS", threads)
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = gpus[k % len(gpus)]
        cmd = [sys.executable, "-m", "traffic_bench.policy_lab.judge", *base,
               "--private-dir", str(shard_dir), "--out", str(shard_dir / "feedback.json"),
               "--shard", str(k), "--num-shards", str(args.workers)]
        log = open(shard_dir / "judge.log", "w")
        procs.append((k, shard_dir, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    print(f"[judge] {args.workers} workers" + (f" on GPU {','.join(gpus)}" if gpus else ""), flush=True)

    episodes: list[dict] = []
    load_error = None
    failed = []
    for k, shard_dir, proc, log in procs:
        proc.wait()
        log.close()
        out = shard_dir / "feedback.json"
        if proc.returncode != 0 or not out.is_file():
            failed.append(f"worker {k} exited with {proc.returncode} (see {shard_dir / 'judge.log'})")
            continue
        load_error = load_error or json.loads(out.read_text()).get("policy_load_error")
        episodes += [json.loads(l) for l in (shard_dir / "episodes.jsonl").read_text().splitlines() if l.strip()]
    if failed:
        raise SystemExit("judge workers failed:\n  " + "\n  ".join(failed))
    # Same private layout as one worker: merged episodes and policy log at the top.
    with open(private_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(e, default=str) + "\n" for e in episodes)
    with open(private_dir / "policy.log", "w", encoding="utf-8") as f:
        for k, shard_dir, _, _ in procs:
            log = shard_dir / "policy.log"
            if log.is_file():
                f.write(f"--- worker {k} ---\n" + log.read_text(errors="replace"))
    return build_feedback(episodes, args.policy, time.time() - t0, load_error)


if __name__ == "__main__":
    main()
