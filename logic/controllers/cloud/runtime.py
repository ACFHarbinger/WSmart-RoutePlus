"""Task runtime: runs one task attempt on whatever machine a backend started.

Entry point of the release image: ``python -m logic.controllers.cloud.runtime``.
It reads its task from the environment:

- ``WSR_CLOUD_CONTROL_PLANE`` and ``WSR_CLOUD_STORE``: URIs (see :mod:`.profiles`);
- ``WSR_CLOUD_JOB_ID``, ``WSR_CLOUD_TASK_ID``: the task;
- ``WSR_CLOUD_WORKDIR``: scratch directory (default: a temporary directory);
- ``WSR_CLOUD_PROJECT_ROOT``: where ``main.py`` is (default: this checkout);
- ``WSR_CLOUD_SYNC_SECONDS``: seconds between output uploads (default 30);
- ``WSR_CLOUD_SAMPLE_SECONDS``: seconds between resource samples and progress reads (default 5);
- ``WSR_GUROBI_LICENSE``: the license text, injected by the backend from the secret store;
- ``WSR_SECRET_REFS``: secret references to resolve with the machine's identity (:mod:`.auth.secret_refs`),
  for services that cannot inject secrets natively. ``WSR_GUROBI_LICENSE`` may be one of them.

Then it:

1. resolves ``WSR_SECRET_REFS`` (exit 4 if one cannot be resolved) and fetches the staged inputs (:mod:`.store.staging`);
2. writes the Gurobi license file (mode 0600) when the job has a license mode,
   and fails before running when a cell needs Gurobi and no license was injected;
3. restores the task's outputs mirror from the object store, so a new attempt on
   another machine resumes from the last finished day;
4. runs ``main.py test_sim`` with ``sim.resume=true sim.checkpoint_days=1``, uploads
   changed output files every ``WSR_CLOUD_SYNC_SECONDS`` and records a heartbeat
   whenever outputs change;
5. streams resource, progress and day records to the control plane (:mod:`.telemetry`)
   and keeps the latest day in the task's progress;
6. kills the run after ``Resources.idle_seconds`` without new output;
7. on SIGTERM (spot preemption, cancellation) stops the simulator, uploads once
   more and exits 143; on failure records the log tail as a ``log_tail`` event;
   on success records the results index and exits 0.

The scheduler owns the task's state; the runtime only writes heartbeats,
progress and results.

Attributes:
    OutputMirror: Incremental upload and restore of a task's output directory.
    run_task: Run one attempt; returns the exit code.
    EXIT_NO_LICENSE, EXIT_NO_SECRET, EXIT_IDLE, EXIT_TERMINATED: Runtime exit codes.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from logic.controllers.cloud.auth.secret_refs import parse_refs, redact, resolve_refs
from logic.controllers.cloud.interfaces import ControlPlane, ObjectStore
from logic.controllers.cloud.model import CellSpec, Event, GurobiMode, JobSpec, ResultFile
from logic.controllers.cloud.store.staging import fetch_inputs
from logic.controllers.cloud.store.storage import sha256_file
from logic.controllers.cloud.telemetry import TaskTelemetry, days_from_argv

EXIT_NO_LICENSE = 3
EXIT_NO_SECRET = 4
EXIT_IDLE = 124
EXIT_TERMINATED = 143
PACKAGE_ROOT = Path(__file__).resolve().parents[3]


class OutputMirror:
    """Mirror of a local output directory under an object-store prefix.

    Args:
        store: Object store.
        prefix: Key prefix shared by every attempt of the task.
        root: Local output directory.
    """

    def __init__(self, store: ObjectStore, prefix: str, root: Path) -> None:
        self.store = store
        self.prefix = prefix.rstrip("/")
        self.root = Path(root)
        self._seen: Dict[str, Tuple[int, int]] = {}
        self.files: Dict[str, ResultFile] = {}

    def restore(self) -> int:
        """Download the mirror into the output directory; returns the number of files."""
        keys = self.store.list(self.prefix + "/")
        for key in keys:
            relpath = key[len(self.prefix) + 1 :]
            target = self.root / relpath
            self.store.get_file(key, target)
            stat = target.stat()
            self._seen[relpath] = (stat.st_size, stat.st_mtime_ns)
        return len(keys)

    def sync(self) -> int:
        """Upload files that are new or changed since the last sync; returns how many."""
        uploaded = 0
        if not self.root.is_dir():
            return 0
        for path in sorted(p for p in self.root.rglob("*") if p.is_file()):
            relpath = path.relative_to(self.root).as_posix()
            try:
                stat = path.stat()
            except FileNotFoundError:  # removed while scanning (a rotated checkpoint)
                continue
            signature = (stat.st_size, stat.st_mtime_ns)
            if self._seen.get(relpath) == signature:
                continue
            try:
                record = self.store.put_file(path, f"{self.prefix}/{relpath}")
            except FileNotFoundError:
                # Renamed or removed during the upload (a checkpoint's ``.tmp`` replaced by the final
                # file): skip it; the next sync uploads what took its place.
                continue
            self.files[relpath] = ResultFile(relpath=relpath, key=record.key, sha256=record.sha256, size=record.size)
            self._seen[relpath] = signature
            uploaded += 1
        return uploaded

    def index(self) -> List[ResultFile]:
        """Every file currently in the mirror, with the hash of its latest upload."""
        records = dict(self.files)
        for key in self.store.list(self.prefix + "/"):
            relpath = key[len(self.prefix) + 1 :]
            if relpath not in records and (self.root / relpath).is_file():
                local = self.root / relpath
                records[relpath] = ResultFile(relpath, key, sha256_file(local), local.stat().st_size)
        return [records[relpath] for relpath in sorted(records)]


def entry_script(project_root: Path) -> str:
    """The simulator's entry script: ``main.py`` in a checkout, ``__main__.py`` in a release tree."""
    return "main.py" if (Path(project_root) / "main.py").is_file() else "__main__.py"


def _tee(stream: Any, path: Path, echo: Any) -> threading.Thread:
    """Copy the simulator's output to a log file and to the runtime's stdout (the cloud log)."""

    def pump() -> None:
        with open(path, "ab") as log:
            for line in iter(stream.readline, b""):
                log.write(line)
                log.flush()
                if echo is not None:
                    echo.write(line)
                    echo.flush()

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    return thread


def _tail(path: Path, lines: int = 20) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def _write_license(workdir: Path, text: str) -> Path:
    path = workdir / "gurobi.lic"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(text if text.endswith("\n") else text + "\n")
    return path


def _simulator_env(
    spec: JobSpec, cell: CellSpec, workdir: Path, policy_dir: Optional[Path], license_text: Optional[str]
) -> Optional[Dict[str, str]]:
    """The simulator's environment, or None when a cell needs Gurobi and no license was injected."""
    env = dict(os.environ)
    env.pop("WSR_GUROBI_LICENSE", None)  # the simulator gets a license file, never the secret itself
    env.update(cell.env)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[var] = "1"
    if policy_dir is not None:
        env["WSR_POLICY_CONFIG_DIR"] = str(policy_dir)
    if GurobiMode(spec.gurobi) != GurobiMode.NONE:
        if license_text:
            env["GRB_LICENSE_FILE"] = str(_write_license(workdir, license_text))
        elif cell.needs_gurobi:
            return None
    return env


def _supervise(
    proc: subprocess.Popen,
    mirror: OutputMirror,
    heartbeat: Callable[[float, int], object],
    idle_seconds: Optional[float],
    sync_seconds: float,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    on_tick: Optional[Callable[[float], None]] = None,
) -> Tuple[Optional[int], bool]:
    """Wait for the simulator while mirroring its outputs (and calling ``on_tick`` every loop).

    Returns:
        ``(exit code, terminated)``; the code is None when the idle limit stopped the run.
    """
    terminated = {"flag": False}

    def on_term(signum: int, _frame: object) -> None:
        terminated["flag"] = True
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)

    previous = signal.signal(signal.SIGTERM, on_term)
    last_change = last_sync = clock()
    try:
        while proc.poll() is None:
            sleep(min(1.0, sync_seconds))
            now = clock()
            if on_tick is not None:
                on_tick(now)
            if now - last_sync >= sync_seconds:
                last_sync = now
                if mirror.sync():
                    last_change = now
                    heartbeat(now, len(mirror.files))
            if idle_seconds is not None and now - last_change > idle_seconds:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                return None, False
        return proc.returncode, terminated["flag"]
    finally:
        signal.signal(signal.SIGTERM, previous)
        mirror.sync()


def run_task(
    cp: ControlPlane,
    store: ObjectStore,
    job_id: str,
    task_id: str,
    workdir: Path,
    project_root: Path = PACKAGE_ROOT,
    python: str = sys.executable,
    sync_seconds: float = 30.0,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    license_text: Optional[str] = None,
    sample_seconds: float = 5.0,
    secret_refs: Optional[Dict[str, str]] = None,
    resolvers: Optional[Dict[str, Any]] = None,
) -> int:
    """Run one attempt of a task; returns the process exit code (see the module docstring)."""
    job = cp.get_job(job_id)
    task = cp.get_task(task_id)
    spec, cell = job.spec, task.cell
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    def note(message: str, **fields: Any) -> None:
        cp.update_task(task_id, message=message, **fields)

    try:
        secrets = resolve_refs(secret_refs or {}, resolvers)
    except LookupError as exc:
        note(str(exc))
        return EXIT_NO_SECRET
    license_text = secrets.pop("WSR_GUROBI_LICENSE", None) or license_text

    policy_dir = None
    if spec.inputs_manifest:
        policy_dir = fetch_inputs(store, spec.inputs_manifest, cell.cell_id, project_root, workdir)
    env = _simulator_env(spec, cell, workdir, policy_dir, license_text)
    if env is not None:
        env.pop("WSR_SECRET_REFS", None)
        env.update(secrets)
    if env is None:
        note("no Gurobi license was injected (WSR_GUROBI_LICENSE)")
        return EXIT_NO_LICENSE

    outputs = workdir / "outputs"
    mirror = OutputMirror(store, f"work/{job_id}/{task_id}/outputs", outputs)
    restored = mirror.restore()
    note(f"attempt {task.attempt}: restored {restored} output file(s)", heartbeat=clock())

    argv = [
        python,
        entry_script(project_root),
        *cell.argv,
        "sim.cpu_cores=1",
        "sim.resume=true",
        "sim.checkpoint_days=1",
        f"sim.output_dir={outputs}",
    ]
    log_path = workdir / f"simulator-attempt-{task.attempt}.log"
    proc = subprocess.Popen(
        argv, cwd=project_root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True
    )
    pump = _tee(proc.stdout, log_path, getattr(sys.stdout, "buffer", None))
    telemetry = TaskTelemetry(
        cp, task_id, outputs, workdir, days=days_from_argv(cell.argv), sample_seconds=sample_seconds, clock=clock
    )
    telemetry.attach(proc.pid)
    code, terminated = _supervise(
        proc,
        mirror,
        lambda now, files: cp.update_task(
            task_id, heartbeat=now, progress={"files": files, **telemetry.progress.latest}
        ),
        spec.resources.idle_seconds,
        sync_seconds,
        clock,
        sleep,
        on_tick=telemetry.tick,
    )
    pump.join(timeout=10)
    telemetry.close()
    if telemetry.errors:
        cp.append_event(
            Event(job_id=job_id, kind="telemetry_error", task_id=task_id, data={"errors": telemetry.errors}, at=clock())
        )
    if log_path.is_file():
        store.put_file(log_path, f"work/{job_id}/{task_id}/logs/attempt-{task.attempt}.log")
    if code is None:
        note(f"no new output for {spec.resources.idle_seconds:.0f}s; stopped")
        return EXIT_IDLE
    if terminated:
        note("terminated (SIGTERM); outputs uploaded for the next attempt")
        return EXIT_TERMINATED
    if code != 0:
        tail = redact(_tail(log_path), [license_text or "", *secrets.values()])
        cp.append_event(
            Event(job_id=job_id, kind="log_tail", task_id=task_id, data={"code": code, "tail": tail}, at=clock())
        )
        note(f"simulator exited with code {code}:\n{tail}")
        return code
    cp.record_results(task_id, mirror.index())
    cp.update_task(
        task_id, heartbeat=clock(), progress={"files": len(mirror.files), "done": True, **telemetry.progress.latest}
    )
    return 0


def main() -> int:
    """Run the task named by the environment (see the module docstring)."""
    from logic.controllers.cloud.profiles import control_plane_from_uri, store_from_uri

    cp = control_plane_from_uri(os.environ["WSR_CLOUD_CONTROL_PLANE"])
    store = store_from_uri(os.environ["WSR_CLOUD_STORE"])
    workdir = os.environ.get("WSR_CLOUD_WORKDIR") or tempfile.mkdtemp(prefix="wsr-task-")
    return run_task(
        cp,
        store,
        job_id=os.environ["WSR_CLOUD_JOB_ID"],
        task_id=os.environ["WSR_CLOUD_TASK_ID"],
        workdir=Path(workdir),
        project_root=Path(os.environ.get("WSR_CLOUD_PROJECT_ROOT") or PACKAGE_ROOT),
        sync_seconds=float(os.environ.get("WSR_CLOUD_SYNC_SECONDS", "30")),
        sample_seconds=float(os.environ.get("WSR_CLOUD_SAMPLE_SECONDS", "5")),
        license_text=os.environ.get("WSR_GUROBI_LICENSE"),
        secret_refs=parse_refs(os.environ.get("WSR_SECRET_REFS")),
    )


if __name__ == "__main__":
    sys.exit(main())
