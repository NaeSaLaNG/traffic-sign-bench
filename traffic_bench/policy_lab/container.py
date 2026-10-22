"""Apptainer command lines for the sandbox. Standard library only, so the
agent orchestrator can use it without the simulator installed."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

SANDBOX_MOUNT = "/work"
CKPT_MOUNT = "/ckpt/carl.pth"


@dataclass
class ContainerSpec:
    image: Path | None                    # required unless launcher == "local"
    checkpoint: Path
    apptainer_bin: str = "apptainer"
    gpu: bool = False
    isolate_network: bool = True          # --net --network none (may need unprivileged netns support)
    launcher: str = "apptainer"           # "apptainer" | "local" (debug only: no isolation)
    extra_args: list[str] = field(default_factory=list)


def local_env(sandbox_dir: Path, checkpoint: Path) -> dict[str, str]:
    """Environment for launcher=local: the sandbox imports as it does in the
    container and ``python`` is the orchestrator's interpreter. No isolation:
    the process sees the host filesystem and network."""
    root = Path(sandbox_dir).resolve()
    env = {
        "PATH": os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "/usr/bin:/bin")]),
        "PYTHONPATH": str(root),
        "CARL_NUPLAN_PATH": str(root / "third_party" / "CaRL" / "nuPlan"),
        "CARL_CHECKPOINT": str(Path(checkpoint).resolve()),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for key in ("HOME", "TMPDIR", "CUDA_VISIBLE_DEVICES"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def apptainer_exec(
    spec: ContainerSpec,
    sandbox_dir: Path,
    argv: list[str],
    *,
    sandbox_writable: bool = False,
    binds: tuple[tuple[Path, str], ...] = (),
) -> list[str]:
    """Only the sandbox, the checkpoint and ``binds`` are visible inside:
    no home, no host environment, no working directory, no network."""
    mode = "rw" if sandbox_writable else "ro"
    cmd = [
        spec.apptainer_bin, "exec",
        "--containall", "--cleanenv", "--no-home", "--writable-tmpfs",
        "--bind", f"{Path(sandbox_dir).resolve()}:{SANDBOX_MOUNT}:{mode}",
        "--bind", f"{Path(spec.checkpoint).resolve()}:{CKPT_MOUNT}:ro",
    ]
    for host, target in binds:
        cmd += ["--bind", f"{Path(host).resolve()}:{target}"]
    cmd += [
        "--pwd", SANDBOX_MOUNT,
        "--env", f"PYTHONPATH={SANDBOX_MOUNT}",
        "--env", f"CARL_NUPLAN_PATH={SANDBOX_MOUNT}/third_party/CaRL/nuPlan",
        "--env", "PYTHONDONTWRITEBYTECODE=1",
    ]
    if spec.isolate_network:
        cmd += ["--net", "--network", "none"]
    if spec.gpu:
        cmd.append("--nv")
    cmd += list(spec.extra_args)
    cmd.append(str(spec.image))
    return cmd + list(argv)
