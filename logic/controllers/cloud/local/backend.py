"""Local subprocess backend.

Each attempt runs in ``<workdir>/attempt-<n>/`` through :mod:`._wrap`, which
records the exit code in ``exit_code`` when the command ends. The handle is that
directory, so any process can poll or cancel an attempt, and a scheduler restart
does not lose running tasks. Starts are held back while available memory is
below the task's request, like the paper rerun drivers' memory gate.

Attributes:
    LocalProcessBackend: :class:`~logic.controllers.cloud.interfaces.ComputeBackend` on local subprocesses.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional, Union

from logic.controllers.cloud.interfaces import (
    AttemptState,
    BackendCapabilities,
    BackendStatus,
    CapacityError,
    ComputeBackend,
    LaunchSpec,
    SecretStore,
)
from logic.controllers.cloud.model import ComputeKind, Provider

_GIB = 1 << 30
# The directory holding the ``logic`` package: the wrapper is started as ``-m logic...``.
_PACKAGE_ROOT = Path(__file__).resolve().parents[4]


def mem_available() -> int:
    """``MemAvailable`` in bytes, or 0 where ``/proc/meminfo`` is missing."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


class LocalProcessBackend(ComputeBackend):
    """Runs attempts as local subprocesses.

    Args:
        cwd: Working directory of every command (the repository root, where ``main.py`` is).
        secrets: Store that resolves ``LaunchSpec.secret_env`` at start, if any.
        memory_gate: Refuse starts while ``MemAvailable`` is below the task's ``memory_gb``.
    """

    provider = Provider.LOCAL
    kind = ComputeKind.PROCESS

    def __init__(self, cwd: Union[str, Path], secrets: Optional[SecretStore] = None, memory_gate: bool = True) -> None:
        self.cwd = Path(cwd)
        self.secrets = secrets
        self.memory_gate = memory_gate

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(provider=Provider.LOCAL, kinds=[ComputeKind.PROCESS], max_wall_seconds=None)

    def submit(self, launch: LaunchSpec) -> str:
        if not launch.workdir:
            raise ValueError("the local backend needs LaunchSpec.workdir")
        available = mem_available()
        if self.memory_gate and available and available < launch.resources.memory_gb * _GIB:
            raise CapacityError(f"{available / _GIB:.1f} GiB available, task asks for {launch.resources.memory_gb} GiB")
        for previous in Path(launch.workdir).glob("attempt-*"):
            pid = _pid(previous)
            if pid is not None and not (previous / "exit_code").exists() and _alive(pid):
                # A stopped attempt still shutting down would share the task's outputs.
                raise CapacityError(f"{previous.name} of this task is still running")
        attempt = Path(launch.workdir) / f"attempt-{launch.task.attempt}"
        attempt.mkdir(parents=True, exist_ok=True)
        for stale in ("exit_code", "cancelled"):
            (attempt / stale).unlink(missing_ok=True)
        env = dict(os.environ)
        env.update(launch.env)
        # The task's cwd need not contain this checkout, so make ``-m logic...`` importable from anywhere.
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(_PACKAGE_ROOT), env.get("PYTHONPATH", "")) if p)
        for var, secret_name in launch.secret_env.items():
            value = self.secrets.get(secret_name) if self.secrets is not None else None
            if value is None:
                raise ValueError(f"secret {secret_name!r} for {var} is not available")
            env[var] = value
        with open(attempt / "log.txt", "ab") as log:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "logic.controllers.cloud.local._wrap",
                    str(attempt / "exit_code"),
                    "--",
                    *launch.argv,
                ],
                cwd=self.cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        (attempt / "pid").write_text(str(proc.pid))
        (attempt / "launch.json").write_text(json.dumps({"argv": launch.argv, "task_id": launch.task.task_id}))
        return str(attempt)

    def poll(self, handle: str) -> BackendStatus:
        attempt = Path(handle)
        exit_file = attempt / "exit_code"
        if exit_file.exists():
            code = int(exit_file.read_text().strip() or "1")
            if (attempt / "cancelled").exists():
                return BackendStatus(AttemptState.FAILED, exit_code=code, message="cancelled")
            if code == 0:
                return BackendStatus(AttemptState.SUCCEEDED, exit_code=0)
            return BackendStatus(AttemptState.FAILED, exit_code=code, message=f"exit code {code}")
        pid = _pid(attempt)
        if pid is None:
            return BackendStatus(AttemptState.UNKNOWN, message="no such attempt")
        if _alive(pid):
            return BackendStatus(AttemptState.RUNNING)
        # Killed before the wrapper could record an exit code (SIGKILL, OOM, reboot).
        return BackendStatus(AttemptState.FAILED, message="process ended without an exit code")

    def cancel(self, handle: str) -> None:
        attempt = Path(handle)
        pid = _pid(attempt)
        if pid is None or (attempt / "exit_code").exists():
            return
        (attempt / "cancelled").touch()
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGTERM)

    def last_activity(self, handle: str) -> Optional[float]:
        """Unix time the attempt last wrote to its log (the idle limit's heartbeat)."""
        try:
            return (Path(handle) / "log.txt").stat().st_mtime
        except OSError:
            return None

    def logs(self, handle: str, tail: Optional[int] = None) -> str:
        log = Path(handle) / "log.txt"
        if not log.exists():
            return ""
        text = log.read_text(errors="replace")
        if tail is None:
            return text
        return "\n".join(text.splitlines()[-tail:])


def _pid(attempt: Path) -> Optional[int]:
    try:
        return int((attempt / "pid").read_text())
    except (OSError, ValueError):
        return None


def _alive(pid: int) -> bool:
    try:
        finished, _ = os.waitpid(pid, os.WNOHANG)
        if finished == pid:
            return False
    except ChildProcessError:
        pass  # not our child: fall through to a signal-0 probe
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
