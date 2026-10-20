"""One background scheduler per local state root.

A worker holds an exclusive ``flock`` on ``<root>/worker.lock`` while it runs
:meth:`~logic.controllers.cloud.service.CloudService.run`, and exits when every
job has finished. ``cloud submit`` starts one with :func:`spawn_worker` when none
holds the lock, so a job submitted from the CLI or the Studio always gets
scheduled without a terminal kept open.

Attributes:
    worker_running: Whether a worker holds the root's lock.
    spawn_worker: Start a detached worker unless one is running.
    serve: The worker loop.
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:
    from logic.controllers.cloud.service import CloudService

LOCK_NAME = "worker.lock"
LOG_NAME = "worker.log"
_FINISHED = ("succeeded", "failed", "cancelled")


@contextmanager
def worker_lock(root: Path) -> Iterator[bool]:
    """Hold the root's worker lock; yields False (without waiting) if another process holds it."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / LOCK_NAME, "a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()))
            handle.flush()
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def worker_running(root: Path) -> bool:
    """Whether a worker currently holds ``<root>/worker.lock``."""
    with worker_lock(root) as mine:
        return not mine


def spawn_worker(
    root: Path,
    project_root: Path,
    python: str = sys.executable,
    poll_seconds: float = 5.0,
    profile: Optional[str] = None,
) -> Optional[int]:
    """Start a detached ``cloud worker`` for ``root`` unless one is running; returns its pid or None.

    The worker imports this package (it runs from this checkout) and starts tasks in ``project_root``;
    with ``profile`` it schedules that cloud profile's jobs.
    """
    root = Path(root).resolve()
    if worker_running(root):
        return None
    package_root = Path(__file__).resolve().parents[3]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(package_root), env.get("PYTHONPATH", "")) if p)
    with open(root / LOG_NAME, "ab") as log:
        proc = subprocess.Popen(
            [
                python,
                "-m",
                "logic.controllers.cloud",
                "--root",
                str(root),
                "--project-root",
                str(Path(project_root).resolve()),
                *(["--profile", profile] if profile else []),
                "worker",
                "--poll",
                str(poll_seconds),
            ],
            cwd=package_root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return proc.pid


def _unfinished(service: "CloudService") -> bool:
    return any(service.status(job.job_id)["state"] not in _FINISHED for job in service.cp.list_jobs(limit=1000))


def serve(service: "CloudService", root: Path, poll_seconds: float = 5.0) -> str:
    """Schedule every job of ``root`` until all have finished; returns why the worker stopped.

    After releasing the lock the worker looks once more for unfinished jobs: a
    submit that found the lock held while this worker was finishing did not start
    a new worker, so its job is picked up here.
    """
    while True:
        with worker_lock(root) as mine:
            if not mine:
                return "another worker is running"
            service.run(None, poll_seconds=poll_seconds)
        if not _unfinished(service):
            return "queue drained"
