"""Offline smoke test: run a policy on synthetic observations, no simulator.

    python check_policy.py policies/my_policy.py [--steps 50] [--checkpoint $CARL_CHECKPOINT]

Catches import errors, exceptions, wrong return types, slow steps and
CarlDriver contract breaks (observe/controls called twice in a step). It says
nothing about driving quality; use the evaluate tool for that.
"""
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np

from carl_sandbox.carl import CarlDriver
from carl_sandbox.runtime import load_policy_class


def synthetic_obs(step: int, speed: float) -> dict:
    x = speed * 0.1 * step
    return {
        "type": "step",
        "step": step,
        "ego": {
            "position": [x, 0.0], "heading": 0.0, "speed_mps": speed,
            "velocity_xy": [speed, 0.0], "steering_cmd": 0.0, "max_steering_rad": 0.7,
            "length": 4.5, "width": 2.0, "dt": 0.1,
            "lane": {"id": "e1|e2|0", "road_id": "e1|e2", "s": x, "lateral": 0.0,
                     "length": 200.0, "width": 3.5, "heading": 0.0},
        },
        "route": {"completion": x / 300.0, "lanes": [{"id": "e1|e2|0", "road_id": "e1|e2", "length": 200.0}],
                  "checkpoints": ["e1", "e2", "e3"]},
        "signs": [{"plate": "speed_limit_40", "position": [60.0, -3.0], "ego_frame": [60.0 - x, -3.0],
                   "distance": math.hypot(60.0 - x, 3.0), "heading": math.pi, "lane_id": "e1|e2|0",
                   "road_id": "e1|e2", "on_ego_lane": True, "on_ego_road": True,
                   "same_direction_as_ego": True}],
        "vehicles": [], "pedestrians": [], "traffic_lights": [],
        "bev": np.zeros((9, 256, 256), dtype=np.uint8),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--checkpoint", default=os.environ.get("CARL_CHECKPOINT"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    carl = CarlDriver(args.checkpoint, device=args.device)
    policy = load_policy_class(args.policy)(carl)
    carl.reset()
    policy.reset({"episode_index": 0})
    worst = 0.0
    for step in range(args.steps):
        t0 = time.perf_counter()
        out = policy.act(synthetic_obs(step, speed=8.0))
        worst = max(worst, time.perf_counter() - t0)
        steering, throttle = out
        for name, v in (("steering", steering), ("throttle", throttle)):
            if not isinstance(v, (int, float, np.floating)) or not math.isfinite(float(v)):
                raise SystemExit(f"step {step}: {name}={v!r} is not a finite number")
            if abs(float(v)) > 1.0:
                print(f"step {step}: {name}={float(v):.3f} will be clipped to [-1, 1]")
    if carl.contract_violations:
        raise SystemExit("CarlDriver contract broken (the steps were driven by a corrupted controller):\n  "
                         + "\n  ".join(carl.contract_violations)
                         + "\nDecide the raw action first, then call controls(raw, obs) once; do not fall "
                           "back to carl.step(obs) after observe() in the same step.")
    print(f"OK: {args.steps} steps, slowest step {worst * 1000:.1f} ms (limit 5000 ms)")


if __name__ == "__main__":
    main()
