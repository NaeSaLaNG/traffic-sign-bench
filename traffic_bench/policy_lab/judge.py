"""Evaluate a sandboxed policy and reduce the results to aggregate feedback.

Runs on the trusted side with the full repository. Per-episode results (with
violation timelines) go to ``--private-dir`` and never reach the agent; the
``--out`` JSON holds only per-family rates, which is all the agent sees.

    python -m traffic_bench.policy_lab.judge \
        --sandbox /scratch/policy_lab/sandbox --policy policies/v3.py \
        --checkpoint checkpoints/carl/nuplan_51479_1B/model_best.pth \
        --image /scratch/policy_lab/policy.sif \
        --manifest data/runs/policy_lab/dev/manifest.jsonl --max-episodes 60 \
        --private-dir /scratch/policy_lab/judge/run_007 --out /scratch/policy_lab/judge/run_007/feedback.json
"""
from __future__ import annotations

import argparse
import json
import random
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

        def clean(r) -> bool:
            return (r.get("ok") and not r.get("policy_fault") and r.get("success")
                    and int(r.get("sign_violations", 0)) == 0
                    and int(r.get("traffic_light_violations", 0)) == 0
                    and int(r.get("crosswalk_violations", 0)) == 0)

        completion = [float(r.get("route_completion_pct") or 0.0) for r in runs if r.get("ok")]
        return {
            "episodes": n,
            "clean_success_rate": frac(clean),
            "success_rate": frac(lambda r: r.get("ok") and not r.get("policy_fault") and r.get("success")),
            "no_sign_violation_rate": frac(lambda r: r.get("ok") and not r.get("policy_fault")
                                           and int(r.get("sign_violations", 0)) == 0),
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
) -> dict:
    cfg = OmegaConf.create({"manifest": manifest, "scenes_root": scenes_root})
    rows, manifest_path = _load_rows(cfg)
    rows = sample_rows(rows, max_episodes, seed)
    scenes = _scenes_root(cfg, manifest_path)
    private_dir = Path(private_dir)
    private_dir.mkdir(parents=True, exist_ok=True)

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
                try:
                    result = run_one_episode(
                        row=row,
                        policy_type=POLICY_TYPE,
                        models={"policy_cls": RemoteCarlPolicy},
                        scenes_root=scenes,
                        max_steps=max_steps,
                        ego_variant="default",
                        ego_sample_seed_base=42,
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
        for row in rows[len(episodes):]:
            episodes.append({"ok": False, "policy_fault": "policy failed to start",
                             "family": sign_family(row.get("_sign_code"))})
    feedback = summarize(episodes)
    feedback["policy"] = launch.policy_path
    feedback["wall_time_s"] = round(time.time() - t0, 1)
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
    ap.add_argument("--image", type=Path, default=None, help="Apptainer .sif for the policy process")
    ap.add_argument("--launcher", choices=["apptainer", "local"], default="apptainer")
    ap.add_argument("--apptainer-bin", default="apptainer")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--allow-network", action="store_true", help="skip --network none")
    ap.add_argument("--step-timeout", type=float, default=5.0)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--scenes-root", default=None)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--private-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="aggregate feedback JSON")
    args = ap.parse_args(argv)

    launch = SandboxLaunch(
        sandbox_dir=args.sandbox,
        policy_path=args.policy,
        checkpoint=args.checkpoint,
        image=args.image,
        launcher=args.launcher,
        apptainer_bin=args.apptainer_bin,
        gpu=args.gpu,
        isolate_network=not args.allow_network,
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
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(feedback, indent=2), encoding="utf-8")
    print(json.dumps(feedback["overall"]))


if __name__ == "__main__":
    main()
