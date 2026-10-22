"""Environment for the processes that run model-written code: the policy
server, ``check_policy.py`` and the agent's ``run`` tool. They run on the host
with the sandbox as working directory; ``hooks.py`` checks what the model
asks them to do. Standard library only, so the agent orchestrator can use it
without the simulator installed."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def sandbox_env(sandbox_dir: Path, checkpoint: Path) -> dict[str, str]:
    """The sandbox is the import root and ``python`` is the orchestrator's
    interpreter; nothing else of the orchestrator's environment (API keys,
    tokens) is passed on."""
    root = Path(sandbox_dir).resolve()
    home = root / ".home"
    temp = root / ".tmp"
    home.mkdir(exist_ok=True)
    temp.mkdir(exist_ok=True)
    env = {
        "PATH": os.pathsep.join([str(Path(sys.executable).parent), "/usr/bin", "/bin"]),
        "PYTHONPATH": str(root),
        "CARL_NUPLAN_PATH": str(root / "third_party" / "CaRL" / "nuPlan"),
        "CARL_CHECKPOINT": str(Path(checkpoint).resolve()),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": str(home),
        "TMPDIR": str(temp),
        "XDG_CACHE_HOME": str(home / ".cache"),
    }
    for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env
