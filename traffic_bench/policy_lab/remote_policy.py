"""Bridge from the MetaDrive episode loop to a policy running in the sandbox.

``PolicyProcess`` starts ``carl_sandbox.runtime`` inside an Apptainer container
that sees only the exported sandbox, the CaRL checkpoint and a socket directory.
``RemoteCarlPolicy`` is the MetaDrive policy class the episode runner
instantiates; each step it sends an observation (``observation.py``) and gets
back two floats. A timeout, a malformed reply or a dead process is a policy
fault: the ego brakes for the rest of the episode and the judge scores the
episode as failed.
"""
from __future__ import annotations

import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from metadrive.policy.base_policy import BasePolicy

from traffic_bench.policy_lab.container import CKPT_MOUNT, ContainerSpec, apptainer_exec, local_env
from traffic_bench.policy_lab.observation import ObservationBuilder
from traffic_bench.policy_lab.sandbox_template.carl_sandbox.protocol import recv_frame, send_frame

IPC_MOUNT = "/ipc"
SOCKET_NAME = "policy.sock"
BRAKE_ACTION = [0.0, -1.0]


@dataclass
class SandboxLaunch:
    sandbox_dir: Path
    policy_path: str                      # relative to sandbox_dir, e.g. "policies/v1.py"
    checkpoint: Path
    image: Path | None = None             # Apptainer .sif; required unless launcher == "local"
    launcher: str = "apptainer"           # "apptainer" | "local" (debug only: no isolation)
    apptainer_bin: str = "apptainer"
    gpu: bool = False
    isolate_network: bool = True
    device: str = "cpu"
    ready_timeout_s: float = 300.0
    step_timeout_s: float = 5.0
    extra_apptainer_args: list[str] = field(default_factory=list)

    def container(self) -> ContainerSpec:
        if self.image is None:
            raise ValueError("an Apptainer image is required for launcher=apptainer")
        return ContainerSpec(
            image=self.image,
            checkpoint=self.checkpoint,
            apptainer_bin=self.apptainer_bin,
            gpu=self.gpu,
            isolate_network=self.isolate_network,
            extra_args=list(self.extra_apptainer_args),
        )


class PolicyProcess:
    """One sandboxed policy server; restartable between episodes."""

    def __init__(self, launch: SandboxLaunch, log_path: Path):
        self.launch = launch
        self.log_path = Path(log_path)
        self._proc: subprocess.Popen | None = None
        self._sock: socket.socket | None = None
        self._ipc_dir: Path | None = None
        self.broken = False

    def start(self) -> None:
        self.stop()
        self.broken = False
        self._ipc_dir = Path(tempfile.mkdtemp(prefix="policy_lab_ipc_"))
        os.chmod(self._ipc_dir, 0o777)
        if self.launch.launcher == "local":
            sock_arg = str(self._ipc_dir / SOCKET_NAME)
            argv = [sys.executable, "-m", "carl_sandbox.runtime",
                    "--socket", sock_arg,
                    "--policy", self.launch.policy_path,
                    "--checkpoint", str(Path(self.launch.checkpoint).resolve()),
                    "--device", self.launch.device]
            env = local_env(self.launch.sandbox_dir, self.launch.checkpoint)
            cwd = str(Path(self.launch.sandbox_dir).resolve())
            cmd = argv
        else:
            argv = ["python", "-m", "carl_sandbox.runtime",
                    "--socket", f"{IPC_MOUNT}/{SOCKET_NAME}",
                    "--policy", self.launch.policy_path,
                    "--checkpoint", CKPT_MOUNT,
                    "--device", self.launch.device]
            cmd = apptainer_exec(self.launch.container(), self.launch.sandbox_dir, argv,
                                 binds=((self._ipc_dir, IPC_MOUNT),))
            env, cwd = None, None
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "ab")
        self._proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=cwd)
        log.close()

        sock_path = self._ipc_dir / SOCKET_NAME
        deadline = time.monotonic() + self.launch.ready_timeout_s
        while not sock_path.exists():
            if self._proc.poll() is not None:
                raise RuntimeError(f"policy process exited with code {self._proc.returncode} before listening")
            if time.monotonic() > deadline:
                raise TimeoutError("policy process did not open its socket in time")
            time.sleep(0.2)
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(max(1.0, deadline - time.monotonic()))
        self._sock.connect(str(sock_path))
        send_frame(self._sock, {"type": "hello"})
        reply, _ = recv_frame(self._sock, max_payload=0)
        if reply.get("type") != "ready":
            raise RuntimeError(f"policy failed to load: {str(reply.get('message', reply))[:4000]}")

    def _request(self, header: dict, payload: bytes = b"", timeout: float | None = None) -> dict:
        if self._sock is None or self.broken:
            raise RuntimeError("policy process is not running")
        try:
            self._sock.settimeout(timeout or self.launch.step_timeout_s)
            send_frame(self._sock, header, payload)
            reply, _ = recv_frame(self._sock, max_payload=0)
        except Exception:
            # A late reply would desync the stream: never reuse this socket.
            self.broken = True
            raise
        if reply.get("type") == "error":
            raise RuntimeError(f"policy error: {str(reply.get('message'))[:4000]}")
        return reply

    def reset(self, info: dict) -> None:
        self._request({"type": "reset", "info": info}, timeout=max(30.0, self.launch.step_timeout_s))

    def step(self, header: dict, payload: bytes) -> list[float]:
        reply = self._request(header, payload)
        if reply.get("type") != "action":
            raise ValueError(f"expected an action, got {reply.get('type')!r}")
        out = []
        for key in ("steering", "throttle"):
            val = reply.get(key)
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{key} must be a finite number, got {val!r}")
            out.append(float(min(1.0, max(-1.0, val))))
        return out

    def alive(self) -> bool:
        return (self._proc is not None and self._proc.poll() is None
                and self._sock is not None and not self.broken)

    def stop(self) -> None:
        if self._sock is not None:
            try:
                self._sock.settimeout(2.0)
                send_frame(self._sock, {"type": "close"})
            except Exception:
                pass
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait()
            self._proc = None
        if self._ipc_dir is not None:
            shutil.rmtree(self._ipc_dir, ignore_errors=True)
            self._ipc_dir = None


class RemoteCarlPolicy(BasePolicy):
    """MetaDrive policy whose decisions come from the sandboxed process."""

    process: PolicyProcess | None = None
    episode_info: dict = {}
    # Set when the current episode hit a policy fault; read by the judge.
    last_fault: str | None = None

    def __init__(self, control_object, random_seed=None, config=None):
        BasePolicy.__init__(self, control_object, random_seed, config)
        cls = type(self)
        cls.last_fault = None
        self._builder: ObservationBuilder | None = None
        self._step = 0
        self._fault: str | None = None
        try:
            if cls.process is None:
                raise RuntimeError("RemoteCarlPolicy.process is not set")
            cls.process.reset(dict(cls.episode_info))
        except Exception as exc:
            self._set_fault(f"reset: {exc}")

    def _set_fault(self, message: str) -> None:
        self._fault = message
        type(self).last_fault = message
        print(f"[RemoteCarlPolicy] policy fault at step {self._step}: {message}", flush=True)

    def act(self, agent_id=None):
        action = BRAKE_ACTION
        if self._fault is None:
            try:
                if self._builder is None:
                    self._builder = ObservationBuilder(self.engine)
                header, payload = self._builder.build(self.control_object, self._step)
                action = type(self).process.step(header, payload)
            except Exception as exc:
                self._set_fault(f"step: {type(exc).__name__}: {exc}")
                action = BRAKE_ACTION
        self._step += 1
        self.action_info["action"] = list(action)
        return list(action)
