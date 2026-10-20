"""Submit, dispatch and manage cloud jobs.

:class:`CloudService` ties a control plane, an object store and the compute
backends together. :meth:`CloudService.tick` is one scheduling pass: it returns
expired leases to the queue, honours cancellations, polls running attempts
(applying the retry policy and the wall-time and idle limits) and starts queued
tasks up to each job's ``max_parallel``. The full scheduler (cross-provider
placement, auto-scaling, warm pools) is CR-09; this pass places every task on
its job's provider and compute kind.

Every attempt runs ``main.py`` with ``sim.resume=true sim.checkpoint_days=1``
and a per-task ``sim.output_dir``, so a retried attempt resumes from the last
finished day.

Attributes:
    CloudService: The job manager.
"""

from __future__ import annotations

import os
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from logic.controllers.cloud.interfaces import (
    AttemptState,
    CapacityError,
    ComputeBackend,
    ControlPlane,
    LaunchSpec,
    ObjectStore,
)
from logic.controllers.cloud.model import (
    ACTIVE,
    CellSpec,
    ComputeKind,
    Event,
    GurobiMode,
    JobRecord,
    JobSpec,
    JobState,
    Provider,
    TaskRecord,
    TaskState,
    job_state,
    new_id,
)
from logic.controllers.cloud.runtime import entry_script
from logic.controllers.cloud.store.staging import stage_job_inputs
from logic.controllers.cloud.store.storage import download_results, upload_tree

BackendKey = Tuple[Provider, ComputeKind]
_LEASE_SECONDS = 120.0


@dataclass
class RemoteConfig:
    """How remote (container) tasks reach the control plane and the object store.

    Attributes:
        control_plane_uri: Control-plane URI the task runtime opens (see :mod:`.profiles`).
        store_uri: Object-store URI the task runtime opens.
        project_root: Local repository whose ``JobSpec.inputs`` are staged at submit.
        python: Interpreter inside the image.
    """

    control_plane_uri: str
    store_uri: str
    project_root: Path
    python: str = "python"


def extract_policy_from_cell(cell: CellSpec) -> Optional[str]:
    """Return a single explicitly requested policy; paths and multiple policies are ambiguous."""
    argv = cell.argv or []
    for index, arg in enumerate(argv):
        if arg in ("--policies", "-p"):
            values = []
            for value in argv[index + 1 :]:
                if value.startswith("-") or "=" in value:
                    break
                values.append(value)
            return values[0] if len(values) == 1 else None
        if arg.startswith(("sim.policies=", "policies=", "--policies=")):
            values = [value.strip(" '\"[]") for value in arg.split("=", 1)[1].split(",")]
            return values[0] if len(values) == 1 and values[0] else None
    return None


class CloudService:
    """The job manager.

    Args:
        control_plane: Queue and state store.
        store: Object store for results.
        backends: Backends by ``(provider, compute kind)``.
        workroot: Local directory for attempt state and local task outputs.
        python: Interpreter used for ``main.py`` on the local backend.
        clock: Time source (tests pass a fake clock).
        remote: Settings for providers other than ``local``; required to submit to them.
    """

    def __init__(
        self,
        control_plane: ControlPlane,
        store: ObjectStore,
        backends: Dict[BackendKey, ComputeBackend],
        workroot: Path,
        python: str = sys.executable,
        clock: Callable[[], float] = time.time,
        remote: Optional[RemoteConfig] = None,
    ) -> None:
        self.cp = control_plane
        self.remote = remote
        self.store = store
        self.backends = backends
        self.workroot = Path(workroot)
        self.python = python
        self.clock = clock
        self.owner = f"{socket.gethostname()}-{os.getpid()}"

    # Submission -----------------------------------------------------------

    def submit(self, spec: JobSpec) -> JobRecord:
        """Validate and enqueue a job; returns its record."""
        spec.validate()
        self._backend(spec)  # fail now, not at the first start, if no backend serves the job
        job_id = new_id("job")
        if Provider(spec.provider) != Provider.LOCAL:
            if self.remote is None:
                raise ValueError(f"submitting to {Provider(spec.provider).value} needs a RemoteConfig")
            if not spec.image:
                raise ValueError("a remote job needs an image (the release image built by CR-03)")
            spec.inputs_manifest = stage_job_inputs(self.store, spec, self.remote.project_root, job_id)
        job = JobRecord(job_id=job_id, spec=spec, created_at=self.clock())
        tasks = [TaskRecord(task_id=new_id("task"), job_id=job.job_id, cell=cell) for cell in spec.cells]
        self.cp.create_job(job, tasks)
        self._event(job.job_id, "submitted", data={"name": spec.name, "tasks": len(tasks)})
        return job

    def _backend(self, spec: JobSpec) -> ComputeBackend:
        key = (Provider(spec.provider), ComputeKind(spec.compute))
        if key not in self.backends:
            available = ", ".join(f"{p.value}/{k.value}" for p, k in self.backends)
            raise ValueError(f"no backend for {key[0].value}/{key[1].value} (available: {available or 'none'})")
        return self.backends[key]

    # Scheduling -----------------------------------------------------------

    def tick(self) -> int:
        """One scheduling pass over every unfinished job; returns the number of state changes."""
        changes = len(self.cp.expire_leases(self.clock()))
        for job in self.cp.list_jobs(limit=1000):
            tasks = self.cp.list_tasks(job.job_id)
            if job_state((t.state for t in tasks), job.cancel_requested) in (
                JobState.SUCCEEDED,
                JobState.FAILED,
                JobState.CANCELLED,
            ) and not any(t.state in ACTIVE for t in tasks):
                continue
            changes += self._tick_job(job, tasks)
        return changes

    def _tick_job(self, job: JobRecord, tasks: List[TaskRecord]) -> int:
        backend = self._backend(job.spec)
        changes = 0
        if job.cancel_requested:
            for task in tasks:
                if task.state == TaskState.QUEUED:
                    self._set(task, TaskState.CANCELLED, finished_at=self.clock(), message="cancelled")
                    changes += 1
                elif task.state in ACTIVE and task.backend_handle:
                    backend.cancel(task.backend_handle)
        for task in tasks:
            if task.state in (TaskState.STARTING, TaskState.RUNNING) and task.backend_handle:
                changes += self._poll(job, task, backend)
        if job.cancel_requested:
            return changes
        running = sum(1 for t in self.cp.list_tasks(job.job_id) if t.state in ACTIVE)
        while running < job.spec.max_parallel:
            leased = self.cp.lease_task(self.owner, _LEASE_SECONDS, job_id=job.job_id)
            if leased is None:
                break
            if not self._start(job, leased, backend):
                break
            running += 1
            changes += 1
        return changes

    def _start(self, job: JobRecord, task: TaskRecord, backend: ComputeBackend) -> bool:
        try:
            handle = backend.submit(self._launch(job, task))
        except CapacityError as exc:
            self._set(task, TaskState.QUEUED, message=f"waiting for capacity: {exc}", lease_owner=None)
            return False
        except Exception as exc:  # noqa: BLE001 - a broken launch fails the task, not the scheduler
            self._set(task, TaskState.FAILED, finished_at=self.clock(), message=f"launch failed: {exc}")
            return True
        self._set(task, TaskState.STARTING, backend_handle=handle, started_at=self.clock(), heartbeat=self.clock())
        return True

    def task_dir(self, task: TaskRecord) -> Path:
        """Local directory of a task's attempts and outputs."""
        return self.workroot / task.job_id / task.task_id

    def _launch(self, job: JobRecord, task: TaskRecord) -> LaunchSpec:
        if Provider(job.spec.provider) != Provider.LOCAL:
            return self._remote_launch(job, task)
        spec = job.spec
        workdir = self.task_dir(task)
        outputs = workdir / "outputs"
        argv = [
            self.python,
            entry_script(getattr(self._backend(spec), "cwd", Path("."))),
            *task.cell.argv,
            "sim.cpu_cores=1",
            "sim.resume=true",
            "sim.checkpoint_days=1",
            f"sim.output_dir={outputs}",
        ]
        env = {var: "1" for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
        env.update(task.cell.env)
        env["WSR_CLOUD_JOB_ID"], env["WSR_CLOUD_TASK_ID"] = job.job_id, task.task_id
        env["WSR_CLOUD_ATTEMPT"] = str(task.attempt)
        if task.cell.policy_dir:
            env["WSR_POLICY_CONFIG_DIR"] = task.cell.policy_dir
        secret_env = {}
        if GurobiMode(spec.gurobi) != GurobiMode.NONE and spec.gurobi_secret:
            # The task runtime (CR-02) turns this into a license file before the run.
            secret_env["WSR_GUROBI_LICENSE"] = spec.gurobi_secret
        return LaunchSpec(
            task=task,
            resources=spec.resources,
            argv=argv,
            env=env,
            secret_env=secret_env,
            workdir=str(workdir),
            image=spec.image,
        )

    def _remote_launch(self, job: JobRecord, task: TaskRecord) -> LaunchSpec:
        """A container attempt: the image runs the task runtime, which reads the rest from the control plane."""
        assert self.remote is not None
        env = {
            "WSR_CLOUD_CONTROL_PLANE": self.remote.control_plane_uri,
            "WSR_CLOUD_STORE": self.remote.store_uri,
            "WSR_CLOUD_JOB_ID": job.job_id,
            "WSR_CLOUD_TASK_ID": task.task_id,
            "WSR_CLOUD_ATTEMPT": str(task.attempt),
        }
        secret_env = {}
        if GurobiMode(job.spec.gurobi) != GurobiMode.NONE and job.spec.gurobi_secret:
            secret_env["WSR_GUROBI_LICENSE"] = job.spec.gurobi_secret
        return LaunchSpec(
            task=task,
            resources=job.spec.resources,
            argv=[self.remote.python, "-m", "logic.controllers.cloud.runtime"],
            env=env,
            secret_env=secret_env,
            image=job.spec.image,
        )

    def _poll(self, job: JobRecord, task: TaskRecord, backend: ComputeBackend) -> int:
        handle = task.backend_handle
        assert handle is not None
        status = backend.poll(handle)
        now = self.clock()
        limits = job.spec.resources
        if status.state in (AttemptState.PENDING, AttemptState.RUNNING, AttemptState.UNKNOWN):
            activity = _last_activity(backend, handle)
            if activity is not None and (task.heartbeat is None or activity > task.heartbeat):
                task = self.cp.update_task(task.task_id, heartbeat=activity)
            changed = 0
            if status.state == AttemptState.RUNNING and task.state == TaskState.STARTING:
                task = self._set(task, TaskState.RUNNING)
                changed = 1
            reason = None
            if (
                limits.max_wall_seconds is not None
                and task.started_at
                and now - task.started_at > limits.max_wall_seconds
            ):
                reason = f"wall-time limit of {limits.max_wall_seconds:.0f}s reached"
            elif limits.idle_seconds is not None and task.heartbeat and now - task.heartbeat > limits.idle_seconds:
                reason = f"no progress for {limits.idle_seconds:.0f}s"
            if reason is None:
                return changed
            backend.cancel(handle)
            return self._fail(job, self._ensure_running(task), reason)
        task = self._ensure_running(task)
        if status.state == AttemptState.SUCCEEDED:
            files = self._collect(job, task)
            self.cp.record_results(task.task_id, files)
            self._set(task, TaskState.SUCCEEDED, finished_at=now, message=None)
            self._event(job.job_id, "result", task.task_id, {"files": len(files)})
            return 1
        if status.state == AttemptState.PREEMPTED:
            task = self._set(task, TaskState.PREEMPTED, message=status.message or "preempted")
            return self._retry_or_fail(job, task, status.message or "preempted")
        if job.cancel_requested:
            self._set(task, TaskState.CANCELLED, finished_at=now, message="cancelled")
            return 1
        return self._fail(job, task, status.message or "failed")

    def _ensure_running(self, task: TaskRecord) -> TaskRecord:
        """Attempts that finished before a poll saw them run still pass through ``running``."""
        return self._set(task, TaskState.RUNNING) if task.state == TaskState.STARTING else task

    def _fail(self, job: JobRecord, task: TaskRecord, reason: str) -> int:
        task = self._set(task, TaskState.FAILED, finished_at=self.clock(), message=reason)
        return self._retry_or_fail(job, task, reason)

    def _retry_or_fail(self, job: JobRecord, task: TaskRecord, reason: str) -> int:
        if task.attempt > job.spec.resources.max_retries:
            if task.state == TaskState.PREEMPTED:
                self._set(task, TaskState.FAILED, finished_at=self.clock(), message=reason)
            return 1
        self._requeue(task, reason)
        return 1

    def _requeue(self, task: TaskRecord, reason: str) -> TaskRecord:
        task = self._set(task, TaskState.RETRYING, message=reason)
        task = self._set(
            task,
            TaskState.QUEUED,
            attempt=task.attempt + 1,
            backend_handle=None,
            finished_at=None,
            lease_owner=None,
        )
        self._event(task.job_id, "retry", task.task_id, {"attempt": task.attempt, "reason": reason})
        return task

    def _collect(self, job: JobRecord, task: TaskRecord):
        """Upload a local task's outputs. Container tasks upload their own (CR-02) and are left as they are."""
        if Provider(job.spec.provider) != Provider.LOCAL:
            return self.cp.list_results(task.task_id)
        outputs = self.task_dir(task) / "outputs"
        if not outputs.is_dir():
            return []
        return upload_tree(self.store, outputs, f"results/{job.job_id}/{task.task_id}")

    def _set(self, task: TaskRecord, state: TaskState, **fields: Any) -> TaskRecord:
        updated = self.cp.update_task(task.task_id, state=state, expected=task.state, **fields)
        self._event(task.job_id, "state", task.task_id, {"from": task.state.value, "to": TaskState(state).value})
        return updated

    def _event(self, job_id: str, kind: str, task_id: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
        self.cp.append_event(Event(job_id=job_id, kind=kind, task_id=task_id, data=data or {}, at=self.clock()))

    def run(
        self,
        job_id: Optional[str] = None,
        poll_seconds: float = 5.0,
        timeout: Optional[float] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> Optional[JobState]:
        """Tick until ``job_id`` (or every job) finishes; returns the job's final state."""
        deadline = None if timeout is None else self.clock() + timeout
        while True:
            self.tick()
            if job_id is not None:
                state = self.status(job_id)["state"]
                if state in (JobState.SUCCEEDED.value, JobState.FAILED.value, JobState.CANCELLED.value):
                    return JobState(state)
            elif all(
                self.status(job.job_id)["state"] in ("succeeded", "failed", "cancelled")
                for job in self.cp.list_jobs(limit=1000)
            ):
                return None
            if deadline is not None and self.clock() > deadline:
                raise TimeoutError(f"job {job_id or '(all)'} still running after {timeout}s")
            sleep(poll_seconds)

    # Management -----------------------------------------------------------

    def _get_price_per_hour(self, spec: JobSpec, spot: bool) -> Optional[float]:
        key = (Provider(spec.provider), ComputeKind(spec.compute))
        backend = self.backends.get(key)
        if backend is not None:
            try:
                p = backend.price_per_hour(spec.resources.machine_type, spec.resources.region, spot)
                if p is not None:
                    return p
            except Exception:
                pass
        prov = Provider(spec.provider).value
        try:
            import importlib

            mod = importlib.import_module(f"logic.controllers.cloud.providers.{prov}.pricing")
            fn = getattr(mod, "price_per_hour", None)
            if callable(fn):
                return fn(spec.resources.machine_type, spec.resources.region, spot)
        except Exception:
            pass
        return None

    def _task_attempt_intervals(
        self,
        task: TaskRecord,
        events: Sequence[Event],
    ) -> List[Tuple[float, float]]:
        """Scheduler-observed attempt intervals, including the current open attempt."""
        intervals: List[Tuple[float, float]] = []
        attempt_start: Optional[float] = None
        for event in events:
            if event.task_id != task.task_id or event.kind != "state":
                continue
            to_state = event.data.get("to")
            if to_state in ("starting", "running") and attempt_start is None:
                attempt_start = event.at
            elif to_state in ("succeeded", "failed", "cancelled", "preempted") and attempt_start is not None:
                if event.at >= attempt_start:
                    intervals.append((attempt_start, event.at))
                attempt_start = None
        if attempt_start is not None and task.state in ACTIVE:
            intervals.append((attempt_start, max(attempt_start, self.clock())))
        if not intervals and task.started_at is not None and task.attempt == 1:
            end = task.finished_at if task.finished_at is not None else (self.clock() if task.state in ACTIVE else None)
            if end is not None and end >= task.started_at:
                intervals.append((task.started_at, end))
        return intervals

    def _task_duration_and_cost(
        self,
        task: TaskRecord,
        events: Sequence[Event],
        price_per_hour: Optional[float],
        provider: Provider,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Return scheduler elapsed seconds and a prorated price-hint estimate.

        These are not provider allocation records or invoiced charges. Billing
        minimums and rounding vary by product and cannot be inferred from provider.
        """
        intervals = self._task_attempt_intervals(task, events)
        # A retry without complete attempt history must not appear as a full total.
        if not intervals or len(intervals) < task.attempt:
            return None, None
        duration = sum(end - start for start, end in intervals)
        cost = None if price_per_hour is None else round(duration / 3600.0 * price_per_hour, 6)
        return duration, cost

    def status(self, job_id: str) -> Dict[str, Any]:
        """A job's state and per-task summary."""
        job = self.cp.get_job(job_id)
        tasks = self.cp.list_tasks(job_id)
        events_by_task: Dict[str, List[Event]] = {}
        for event in self.cp.list_events(job_id):
            if event.task_id is not None:
                events_by_task.setdefault(event.task_id, []).append(event)
        counts: Dict[str, int] = {}
        for task in tasks:
            counts[task.state.value] = counts.get(task.state.value, 0) + 1

        provider = Provider(job.spec.provider)
        compute = ComputeKind(job.spec.compute)
        spot = bool(job.spec.resources.spot)
        machine_type = job.spec.resources.machine_type
        region = job.spec.resources.region
        price_per_hour = self._get_price_per_hour(job.spec, spot=spot)
        on_demand_price_per_hour = self._get_price_per_hour(job.spec, spot=False)

        task_summaries = []
        for t in tasks:
            events = events_by_task.get(t.task_id, [])
            dur, cost = self._task_duration_and_cost(t, events, price_per_hour, provider)
            intervals = self._task_attempt_intervals(t, events)
            task_summaries.append(
                {
                    "task_id": t.task_id,
                    "cell_id": t.cell.cell_id,
                    "state": t.state.value,
                    "attempt": t.attempt,
                    "message": t.message,
                    "started_at": intervals[0][0] if intervals else None,
                    "finished_at": intervals[-1][1] if intervals else None,
                    "duration": dur,
                    "expected_wall_seconds": t.cell.expected_wall_seconds,
                    "policy": extract_policy_from_cell(t.cell),
                    "cost": cost,
                }
            )

        return {
            "job_id": job_id,
            "name": job.spec.name,
            "state": job_state((t.state for t in tasks), job.cancel_requested).value,
            "counts": counts,
            "provider": provider.value,
            "compute": compute.value,
            "spot": spot,
            "machine_type": machine_type,
            "region": region,
            "price_per_hour": price_per_hour,
            "on_demand_price_per_hour": on_demand_price_per_hour,
            "cost_source": "scheduler_elapsed_price_hint",
            "tasks": task_summaries,
        }

    def cancel(self, job_id: str) -> None:
        """Cancel a job: queued tasks never start, running attempts are stopped at the next tick."""
        self.cp.request_cancel(job_id)
        self._event(job_id, "cancel_requested")

    def retry(self, job_id: str) -> int:
        """Requeue a job's failed and cancelled tasks (they resume from their checkpoints)."""
        job = self.cp.get_job(job_id)
        if job.cancel_requested:
            raise ValueError("the job is cancelled; clone it instead")
        retried = 0
        for task in self.cp.list_tasks(job_id):
            if task.state in (TaskState.FAILED, TaskState.CANCELLED):
                self._requeue(task, "retry requested")
                retried += 1
        return retried

    def logs(self, task_id: str, tail: Optional[int] = None) -> str:
        """The current attempt's log."""
        task = self.cp.get_task(task_id)
        if not task.backend_handle:
            return ""
        return self._backend(self.cp.get_job(task.job_id).spec).logs(task.backend_handle, tail)

    def download(self, job_id: str, dest: Path) -> List[Path]:
        """Download every succeeded task's results under ``dest`` (the ``outputs/`` layout)."""
        written: List[Path] = []
        for task in self.cp.list_tasks(job_id):
            if task.state == TaskState.SUCCEEDED:
                written += download_results(self.store, self.cp.list_results(task.task_id), dest)
        return written


def _last_activity(backend: ComputeBackend, handle: str) -> Optional[float]:
    activity = getattr(backend, "last_activity", None)
    return activity(handle) if callable(activity) else None


__all__ = ["CloudService"]
