"""Policy server: loads ``Policy`` from a file and answers the simulator.

    python -m carl_sandbox.runtime --socket /ipc/policy.sock \
        --policy policies/v1.py --checkpoint /ckpt/carl.pth

The policy module must define ``class Policy`` with
``__init__(self, carl: CarlDriver)``, ``reset(self, info: dict)`` and
``act(self, obs: dict) -> (steering, throttle)``; see TASK.md.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import socket
import sys
import traceback
from pathlib import Path

import numpy as np

from carl_sandbox.carl import CarlDriver
from carl_sandbox.protocol import recv_frame, send_frame


def load_policy_class(path: str):
    file = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("submitted_policy", file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["submitted_policy"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "Policy"):
        raise AttributeError(f"{path} defines no Policy class")
    return module.Policy


def decode_obs(header: dict, payload: bytes) -> dict:
    obs = dict(header)
    shape = tuple(int(x) for x in obs.pop("bev_shape"))
    obs.pop("bev_dtype", None)
    obs.pop("payload_bytes", None)
    obs["bev"] = np.frombuffer(payload, dtype=np.uint8).reshape(shape)
    return obs


def serve(sock_path: str, policy_path: str, checkpoint: str, device: str) -> None:
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_path)
    server.listen(1)
    conn, _ = server.accept()
    policy = carl = None
    while True:
        try:
            header, payload = recv_frame(conn)
        except ConnectionError:
            return
        kind = header.get("type")
        try:
            if kind == "hello":
                carl = CarlDriver(checkpoint, device=device)
                policy = load_policy_class(policy_path)(carl)
                send_frame(conn, {"type": "ready"})
            elif kind == "reset":
                carl.reset()
                policy.reset(dict(header.get("info") or {}))
                send_frame(conn, {"type": "ok"})
            elif kind == "step":
                steering, throttle = policy.act(decode_obs(header, payload))
                send_frame(conn, {"type": "action", "steering": float(steering), "throttle": float(throttle)})
            elif kind == "close":
                return
            else:
                send_frame(conn, {"type": "error", "message": f"unknown message {kind!r}"})
        except Exception:
            msg = traceback.format_exc()
            print(msg, file=sys.stderr, flush=True)
            send_frame(conn, {"type": "error", "message": msg[-4000:]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    serve(args.socket, args.policy, args.checkpoint, args.device)


if __name__ == "__main__":
    main()
