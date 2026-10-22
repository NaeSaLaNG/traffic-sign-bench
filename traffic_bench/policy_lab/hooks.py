"""Harness hooks: checks the agent harness runs on the model's tool calls
before anything executes. Model-written code runs on the host as the
operator's user, so these checks are what keeps it inside the sandbox.

* ``check_command`` (the ``run`` tool): no absolute paths outside the
  sandbox, no ``..``, ``~`` or ``$HOME``; no network, package or VCS tools;
  no process control, background jobs, environment assignments or privilege
  changes; no environment dumps or encoded payloads. Inline Python
  (``python -c``, heredocs) gets the Python rules.
* ``check_python`` (``write_file`` of a .py file, and every model-changed
  file before ``run``, ``evaluate``, ``submit`` and the gauntlet's gate): an
  AST scan for network, subprocess, native-code and dynamic-import modules,
  ``os.system``-style calls, ``eval``/``exec``, introspection escapes
  (``__globals__``, ``__subclasses__``...) and string literals naming host
  paths. Shell scripts get the command rules. ``sitecustomize.py``, ``.pth``
  files and symlinks pointing out of the sandbox are refused outright.

Files the sandbox had when the harness started (the exported template, e.g.
``carl_sandbox/runtime.py`` with its socket) are trusted; only files the model
added or changed are scanned. A flagged file found in the sandbox (written by
a shell command) is moved to ``<run-dir>/quarantine/``, or restored if it was
a template file, so it never runs; the call that found it is refused.

These are guardrails against mistakes and casual escapes, not a security
boundary: code written to hide its behaviour can get past a static scan.
Standard library only.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import shlex
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

# Absolute paths a command or policy may legitimately name besides the sandbox.
ALLOWED_ABSOLUTE = ("/dev/null", "/dev/stdout", "/dev/stderr", "/bin/sh", "/bin/bash", "/usr/bin/env")

BANNED_COMMANDS = {
    "network": {"curl", "wget", "nc", "ncat", "netcat", "socat", "ssh", "scp", "sftp", "rsync", "ftp",
                "telnet", "ping", "dig", "nslookup", "host"},
    "package or VCS tool": {"pip", "pip3", "conda", "mamba", "micromamba", "apt", "apt-get", "yum", "dnf",
                            "npm", "git", "huggingface-cli", "uv"},
    "process control": {"kill", "pkill", "killall", "nohup", "setsid", "disown", "screen", "tmux",
                        "crontab", "at", "nice", "renice", "ps", "top", "pgrep"},
    "privilege change": {"sudo", "su", "doas", "chown", "chgrp", "setfacl", "mount", "umount"},
    "environment dump": {"env", "printenv", "export", "declare", "set", "id", "whoami", "hostname"},
    "encoded payload": {"base64", "base32", "xxd", "uudecode", "openssl", "eval", "exec"},
    "container or scheduler": {"apptainer", "singularity", "docker", "podman", "srun", "sbatch", "kubectl"},
}
_COMMAND_KIND = {cmd: kind for kind, cmds in BANNED_COMMANDS.items() for cmd in cmds}

BANNED_MODULES = {
    "network": {"socket", "ssl", "urllib", "http", "requests", "httpx", "aiohttp", "ftplib", "smtplib",
                "telnetlib", "paramiko", "websocket", "websockets", "asyncio", "socketserver", "xmlrpc"},
    "subprocess": {"subprocess", "pty", "multiprocessing", "concurrent", "signal"},
    "native code": {"ctypes", "cffi", "mmap"},
    "dynamic import": {"importlib", "builtins", "runpy", "pkgutil", "zipimport", "code", "codeop"},
    "host access": {"pwd", "grp", "resource", "webbrowser", "getpass", "shutil", "tempfile"},
}
_MODULE_KIND = {mod: kind for kind, mods in BANNED_MODULES.items() for mod in mods}

BANNED_NAMES = {
    "eval", "exec", "compile", "__import__", "breakpoint", "globals", "__builtins__", "__loader__",
    "os.system", "os.popen", "os.fork", "os.forkpty", "os.kill", "os.killpg", "os.setsid", "os.putenv",
    "os.chdir", "os.chroot", "os.symlink", "os.link", "os.chmod", "os.chown", "os.environ", "os.getenv",
    "os.path.expanduser", "os.path.expandvars", "sys.modules", "sys.path", "sys.addaudithook",
    "sys.settrace", "sys.setprofile", "sys._getframe", "torch.hub",
}
_BANNED_PREFIXES = ("os.exec", "os.spawn", "os.posix_spawn", "torch.hub.")
BANNED_DUNDERS = {"__globals__", "__builtins__", "__subclasses__", "__code__", "__closure__", "__bases__",
                  "__base__", "__mro__", "__getattribute__", "__reduce__", "__reduce_ex__", "__loader__",
                  "__spec__", "__import__", "__self__", "__func__", "__traceback__", "__frame__"}

# Python in a command line that did not parse as a whole program.
_PY_TOKENS = re.compile(
    r"\b(?:import|from)\s+(?:" + "|".join(sorted(_MODULE_KIND)) + r")\b"
    r"|\bos\.(?:system|popen|fork|kill|exec\w*|spawn\w*)\b|\b__import__\b|\beval\s*\(|\bexec\s*\("
)
_ABS_PATH = re.compile(r"(?<![\w.\-/~$])(/[^\s'\"`;|&<>(){}]+)")
_ROOT_ARG = re.compile(r"\b(?:cd|ls|find|du|tree|cat|cp|mv|rm|ln|tar|grep|head|tail|less|stat)\s+(?:-\S+\s+)*/(?:\s|$)")
_PARENT_DIR = re.compile(r"(?:^|[\s/'\"=:])\.\.(?:/|[\s'\"]|$)")
_HOME = re.compile(r"(?<![\w-])~(?=/|\s|$)|\$\{?(?:HOME|USER)\b")
_BACKGROUND = re.compile(r"(?<![&>|])&(?![&>])")
_HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1")
_PY_DASH_C = re.compile(r"\bpython[\d.]*\s+(?:-[A-Za-z]+\s+)*-c\s+('[^']*'|\"(?:[^\"\\]|\\.)*\")")


@dataclass
class Verdict:
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.reasons

    def add(self, reason: str) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)

    def extend(self, other: "Verdict") -> None:
        for r in other.reasons:
            self.add(r)


def _path_allowed(path: str, root: Path) -> bool:
    if any(path == a or path.startswith(a + "/") for a in ALLOWED_ABSOLUTE):
        return True
    return path == str(root) or path.startswith(f"{root}/")


def _check_snippet(code: str, root: Path, label: str) -> Verdict:
    """Inline Python if it parses, else the command rules (a heredoc can hold a shell script)."""
    try:
        ast.parse(code)
    except SyntaxError:
        v = check_command(code, root)
        for m in _PY_TOKENS.finditer(code):
            v.add(f"{label}: Python {m.group(0).strip('( ')}")
        return v
    return check_python(code, root, label)


def _split_inline(command: str, root: Path) -> tuple[str, Verdict]:
    """Take heredoc bodies and ``python -c`` code out of the command and check them as code."""
    v = Verdict()
    lines, shell, i = command.split("\n"), [], 0
    while i < len(lines):
        shell.append(lines[i])
        delims = [m.group(2) for m in _HEREDOC.finditer(lines[i])]
        i += 1
        for delim in delims:
            body = []
            while i < len(lines) and lines[i].strip() != delim:
                body.append(lines[i])
                i += 1
            i += 1                                       # the delimiter line
            v.extend(_check_snippet("\n".join(body), root, "heredoc"))
    shell_text = "\n".join(shell)

    def take(m: re.Match) -> str:
        try:
            code = shlex.split(m.group(1))[0]
        except ValueError:
            code = m.group(1)[1:-1]
        v.extend(_check_snippet(code, root, "python -c"))
        return m.group(0)[:m.start(1) - m.start(0)] + "CODE"
    return _PY_DASH_C.sub(take, shell_text), v


def check_command(command: str, root: Path) -> Verdict:
    """Rules for a shell command from the ``run`` tool (or a shell script)."""
    shell, v = _split_inline(command, root)
    for path in _ABS_PATH.findall(shell):
        if not _path_allowed(path.rstrip(",.:"), root):
            v.add(f"absolute path outside the sandbox: {path}")
    if _ROOT_ARG.search(shell):
        v.add("the filesystem root '/'")
    if _PARENT_DIR.search(shell):
        v.add("'..' path component (stay inside the sandbox)")
    if _HOME.search(shell):
        v.add("home directory or user reference (~, $HOME, $USER)")
    if _BACKGROUND.search(shell):
        v.add("background job '&' (commands must finish within the timeout)")
    try:
        lexer = shlex.shlex(shell, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        tokens = shell.split()
    at_start = True
    for tok in tokens:
        if set(tok) <= set(";&|()\n"):
            at_start = True
            continue
        if not at_start:
            continue
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tok, flags=re.DOTALL):
            v.add(f"environment assignment {tok.split('=', 1)[0]}= (the harness sets the environment)")
            continue                                     # still at the command word
        name = tok.rsplit("/", 1)[-1]
        if name in _COMMAND_KIND:
            v.add(f"{_COMMAND_KIND[name]} command: {name}")
        elif name in (".", "source"):
            v.add("sourcing a script")
        at_start = False
    return v


def _dotted(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _banned(full: str) -> bool:
    return full in BANNED_NAMES or full.startswith(_BANNED_PREFIXES)


def check_python(source: str, root: Path, filename: str = "<file>") -> Verdict:
    """AST rules for Python the model wrote."""
    v = Verdict()
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return v                        # cannot run either; check_policy.py reports it
    modules: dict[str, str] = {}        # local name -> what it was imported as
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                modules[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for a in node.names:
                modules[a.asname or a.name] = f"{node.module}.{a.name}"
    for node in ast.walk(tree):
        where = f"{filename}:{getattr(node, 'lineno', '?')}"
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                if top in _MODULE_KIND:
                    v.add(f"{where}: import {a.name} ({_MODULE_KIND[top]})")
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if node.level == 0 and top in _MODULE_KIND:
                v.add(f"{where}: from {node.module} import ... ({_MODULE_KIND[top]})")
            for a in node.names:
                if node.level == 0 and _banned(f"{node.module}.{a.name}"):
                    v.add(f"{where}: from {node.module} import {a.name}")
        elif isinstance(node, (ast.Attribute, ast.Name)):
            if isinstance(node, ast.Attribute) and node.attr in BANNED_DUNDERS:
                v.add(f"{where}: introspection attribute {node.attr}")
                continue
            name = _dotted(node)
            if name is None:
                continue
            head, _, rest = name.partition(".")
            full = modules.get(head, head) + (f".{rest}" if rest else "")
            if _banned(full):
                v.add(f"{where}: {full}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("getattr", "setattr", "delattr") and len(node.args) >= 2:
            target, attr = node.args[0], node.args[1]
            if isinstance(target, ast.Name) and target.id in modules and not (
                    isinstance(attr, ast.Constant) and isinstance(attr.value, str)
                    and not _banned(f"{modules[target.id]}.{attr.value}")):
                v.add(f"{where}: {node.func.id} on module {modules[target.id]} with a computed or banned name")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value
            if s.startswith("/") and len(s) > 1 and not any(c.isspace() for c in s) and not _path_allowed(s, root):
                v.add(f"{where}: host path literal {s[:80]!r}")
            elif s.startswith(("~/", "../")) or "/../" in s:
                v.add(f"{where}: path leaving the sandbox {s[:80]!r}")
    return v


_SCANNED_SUFFIXES = (".py", ".sh", ".bash", ".pth")


class HarnessHooks:
    """Per-run hook state: the trusted baseline, a scan cache, quarantine and the block log."""

    def __init__(self, root: Path, run_dir: Path):
        self.root = Path(root).resolve()
        self.log_path = Path(run_dir) / "hooks.jsonl"
        self.quarantine = Path(run_dir) / "quarantine"
        self.blocks = 0
        self._digests: dict[str, tuple[int, int, str]] = {}   # rel -> (size, mtime_ns, sha256)
        self._scans: dict[tuple[str, str], list[str]] = {}
        self.baseline = {rel: digest for rel, digest in self._files()}
        # Original bytes of trusted code files, to undo a change that fails the checks.
        self._originals = {rel: (self.root / rel).read_bytes() for rel in self.baseline
                           if rel.endswith(_SCANNED_SUFFIXES)}

    def _files(self):
        """(rel, sha256) of every regular file; hashes only what changed since the last call."""
        for p in sorted(self.root.rglob("*")):
            if "__pycache__" in p.parts or p.is_symlink() or not p.is_file():
                continue
            rel, st = str(p.relative_to(self.root)), p.stat()
            cached = self._digests.get(rel)
            if cached is None or cached[:2] != (st.st_size, st.st_mtime_ns):
                cached = (st.st_size, st.st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
                self._digests[rel] = cached
            yield rel, cached[2]

    def _scan_file(self, rel: str, digest: str) -> list[str]:
        if (rel, digest) not in self._scans:
            path = self.root / rel
            if path.name in ("sitecustomize.py", "usercustomize.py") or path.suffix == ".pth":
                reasons = [f"{rel}: startup hook files are not allowed"]
            else:
                text = path.read_bytes().decode("utf-8", errors="replace")
                shebang = text.split("\n", 1)[0] if text.startswith("#!") else ""
                if path.suffix == ".py" or "python" in shebang:
                    reasons = check_python(text, self.root, rel).reasons
                elif path.suffix in (".sh", ".bash") or shebang:
                    reasons = [f"{rel}: {r}" for r in check_command(text, self.root).reasons]
                else:
                    reasons = []
            self._scans[(rel, digest)] = reasons
        return self._scans[(rel, digest)]

    def _remove(self, rel: str) -> str:
        """Take a flagged file out of the sandbox; returns what was done."""
        path = self.root / rel
        if rel in self._originals and not path.is_symlink():
            path.write_bytes(self._originals[rel])
            return "restored the original"
        dest = self.quarantine / time.strftime("%Y%m%d-%H%M%S") / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(dest))
        return "removed from the sandbox"

    def scan_sandbox(self) -> Verdict:
        """Every file the model added or changed, plus symlinks leaving the sandbox.
        Whatever fails is taken out of the sandbox before the verdict returns."""
        v = Verdict()
        flagged: dict[str, list[str]] = {}
        for p in self.root.rglob("*"):
            if p.is_symlink():
                target = p.resolve()
                if target != self.root and self.root not in target.parents:
                    rel = str(p.relative_to(self.root))
                    flagged[rel] = [f"{rel}: symlink pointing outside the sandbox"]
        for rel, digest in self._files():
            if self.baseline.get(rel) != digest:
                reasons = self._scan_file(rel, digest)
                if reasons:
                    flagged[rel] = reasons
        for rel, reasons in flagged.items():
            done = self._remove(rel)
            self._digests.pop(rel, None)
            for r in reasons:
                v.add(f"{r} ({done})")
        return v

    def _block(self, tool: str, detail: str, v: Verdict) -> str:
        self.blocks += 1
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"t": time.time(), "tool": tool, "detail": detail[:2000],
                                    "reasons": v.reasons}) + "\n")
        print(f"[hooks] blocked {tool}: {'; '.join(v.reasons)[:500]}", flush=True)
        return ("blocked by the harness, nothing was executed:\n- " + "\n- ".join(v.reasons[:20])
                + "\nWork inside the sandbox with relative paths, without network, subprocesses, "
                  "process control or dynamic code; policies and helpers must be plain Python. "
                  "Fix or delete the files listed above.")

    def before(self, tool: str, args: dict) -> str | None:
        """None if the call may run, else the message to give the model instead."""
        if tool == "write_file":
            path, content = str(args.get("path", "")), args.get("content")
            name = Path(path).name
            if name in ("sitecustomize.py", "usercustomize.py") or name.endswith(".pth"):
                return self._block(tool, path, Verdict([f"{path}: startup hook files are not allowed"]))
            if not isinstance(content, str):
                return None
            shebang = content.split("\n", 1)[0] if content.startswith("#!") else ""
            if path.endswith(".py") or "python" in shebang:
                v = check_python(content, self.root, path)
            elif path.endswith((".sh", ".bash")) or shebang:
                v = check_command(content, self.root)
            else:
                return None
            return None if v.ok else self._block(tool, path, v)
        if tool == "run":
            command = str(args.get("command", ""))
            v = check_command(command, self.root)
            v.extend(self.scan_sandbox())
            return None if v.ok else self._block(tool, command, v)
        if tool in ("evaluate", "submit", "gate"):
            v = self.scan_sandbox()
            return None if v.ok else self._block(tool, str(args.get("policy_path", "")), v)
        return None
