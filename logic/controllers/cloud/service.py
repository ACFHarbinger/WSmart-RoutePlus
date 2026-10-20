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
from typing import Any, Callable, Dict, List, Optional, Tuple

from logic.controllers.cloud.cost.budget import BudgetPolicy, CostMixin
from logic.controllers.cloud.interfaces import (
    AttemptState,
    ComputeBackend,
    ControlPlane,
    ObjectStore,
)
from logic.controllers.cloud.model import (
    ACTIVE,
    ComputeKind,
    Event,
    JobRecord,
    JobSpec,
    JobState,
    Provider,
    TaskRecord,
    TaskState,
    job_state,
    new_id,
)
from logic.controllers.cloud.provisioning_lease import renew_provisioning_lease
from logic.controllers.cloud.scheduler.mixin import SchedulerMixin
from logic.controllers.cloud.scheduler.placement import TargetHealth, count_by_target, targets_of
from logic.controllers.cloud.scheduler.pool import PoolMixin
from logic.controllers.cloud.service_launch import ServiceLaunchMixin
from logic.controllers.cloud.service_status import ServiceStatusMixin
from logic.controllers.cloud.service_status import extract_policy_from_cell as extract_policy_from_cell  # noqa: F401
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


class CloudService(ServiceStatusMixin, ServiceLaunchMixin, CostMixin, SchedulerMixin, PoolMixin):
    """The job manager.

    Args:
        control_plane: Queue and state store.
        store: Object store for results.
        backends: Backends by ``(provider, compute kind)``.
        workroot: Local directory for attempt state and local task outputs.
        python: Interpreter used for ``main.py`` on the local backend.
        clock: Time source (tests pass a fake clock).
        remote: Settings for providers other than ``local``; required to submit to them.
        predictor_factory: Builds the CR-11 runtime predictor on first use (submit or
            :meth:`predict`); None disables prediction.
        prices: CR-10 price table (:class:`.cost.PriceTable`); None uses the backends' and
            providers' static prices.
        budget: CR-10 monthly cap and warning fractions; None means no monthly cap.
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
        predictor_factory: Optional[Callable[[], Any]] = None,
        prices: Optional[Any] = None,
        budget: Optional[BudgetPolicy] = None,
        pool_mode: bool = False,
        pool_slots: int = 1,
        pool: Optional[Any] = None,
    ) -> None:
        self.cp = control_plane
        self.health = TargetHealth()  # CR-09 per-target circuit breaker (in memory: resets with the process)
        self.pool_mode, self.pool_slots = pool_mode, max(1, int(pool_slots))
        self.pool = pool  # CR-09 PoolPolicy: the coordinator keeps this many workers alive
        self._pool_workers: Dict[str, float] = {}
        self.prices = prices
        self.budget = budget
        self._cost_written: Dict[str, Any] = {}
        self.predictor_factory = predictor_factory
        self._predictor: Optional[Any] = None
        self.remote = remote
        self.store = store
        self.backends = backends
        self.workroot = Path(workroot)
        self.python = python
        self.clock = clock
        self.owner = f"{socket.gethostname()}-{os.getpid()}-{new_id('scheduler')}"
        scope = self.remote.control_plane_uri if self.remote else str(self.workroot.resolve())
        for backend in self.backends.values():
            configure = getattr(backend, "configure_recovery", None)
            if callable(configure):
                configure(self.cp, scope)

    # Submission -----------------------------------------------------------

    def submit(self, spec: JobSpec) -> JobRecord:
        """Validate and enqueue a job; returns its record."""
        spec.validate()
        if spec.placement:  # fail now, not at the first start, if no backend serves the job
            if not any((Provider(t.provider), ComputeKind(t.compute)) in self.backends for t in spec.placement):
                raise ValueError("no backend serves any of the job's placement targets")
        else:
            self._backend(spec)
        job_id = new_id("job")
        remote_targets = [Provider(t.provider) for t in targets_of(spec) if Provider(t.provider) != Provider.LOCAL]
        if remote_targets:
            if self.remote is None:
                raise ValueError(f"submitting to {remote_targets[0].value} needs a RemoteConfig")
            if not spec.image:
                raise ValueError("a remote job needs an image (the release image built by CR-03)")
            spec.inputs_manifest = stage_job_inputs(self.store, spec, self.remote.project_root, job_id)
        predictions = self.predict(spec)
        estimate = self.estimate(spec, predictions) if predictions is not None else None
        budget_warnings = self._check_budget(spec, estimate)  # raises BudgetExceeded before anything is stored
        for cell, prediction in zip(spec.cells, predictions or [], strict=False):
            if cell.expected_wall_seconds is None and prediction.p50 is not None:
                cell.expected_wall_seconds = prediction.p50  # what the scheduler and the Analysis page read
        job = JobRecord(job_id=job_id, spec=spec, created_at=self.clock())
        tasks = [TaskRecord(task_id=new_id("task"), job_id=job.job_id, cell=cell) for cell in spec.cells]
        self.cp.create_job(job, tasks)
        self._event(job.job_id, "submitted", data={"name": spec.name, "tasks": len(tasks)})
        if predictions:
            from logic.controllers.cloud.prediction import record_predictions

            record_predictions(self.cp, job.job_id, predictions, self.clock())
        if estimate is not None:
            self._event(job.job_id, "cost_estimate", data=estimate.to_dict())
        for warning in budget_warnings:
            self._event(job.job_id, "budget_warning", data=warning)
        return job

    def predict(self, spec: JobSpec) -> Optional[List[Any]]:
        """CR-11 predictions for ``spec``'s cells on its machine type, or None without a predictor."""
        if self.predictor_factory is None:
            return None
        if self._predictor is None:
            self._predictor = self.predictor_factory()
        return self._predictor.predict(spec.cells, spec.resources.machine_type)

    def _backend(self, spec: JobSpec) -> ComputeBackend:
        key = (Provider(spec.provider), ComputeKind(spec.compute))
        if key not in self.backends:
            available = ", ".join(f"{p.value}/{k.value}" for p, k in self.backends)
            raise ValueError(f"no backend for {key[0].value}/{key[1].value} (available: {available or 'none'})")
        return self.backends[key]

    # Scheduling -----------------------------------------------------------

    def tick(self) -> int:
        """One scheduling pass over every unfinished job; returns the number of state changes."""
        # Expiration atomically fences crashed placing owners before reconciliation.
        changes = len(self.cp.expire_leases(self.clock()))
        self._month_memo: Optional[Dict[str, float]] = {}  # CR-10 monthly spend, once per pass (not jobs x jobs)
        for backend in self.backends.values():
            reconcile = getattr(backend, "reconcile_provisions", None)
            if callable(reconcile):
                reconcile()
            drain = getattr(backend, "drain_orphans", None)
            if callable(drain):
                drain()
        jobs = self._by_priority(self.cp.list_jobs(limit=1000))  # CR-09: higher priority first
        tasks_by_job = {job.job_id: self.cp.list_tasks(job.job_id) for job in jobs}
        every_task = [task for tasks in tasks_by_job.values() for task in tasks]
        running = count_by_target(every_task, ACTIVE)  # quota use per target, across jobs
        starting = count_by_target(every_task, (TaskState.PLACING, TaskState.STARTING))
        for job in jobs:
            tasks = tasks_by_job[job.job_id]
            if job_state((t.state for t in tasks), job.cancel_requested) in (
                JobState.SUCCEEDED,
                JobState.FAILED,
                JobState.CANCELLED,
            ) and not any(t.state in ACTIVE for t in tasks):
                continue
            if bool(job.spec.pool) != self.pool_mode:
                # Pool jobs run on pool workers; the coordinator only requeues their dead workers' tasks.
                changes += self._requeue_stale_pool_tasks(job, tasks) if job.spec.pool else 0
                continue
            changes += self._tick_job(job, tasks, running, starting)
        changes += self._scale_pool(jobs, tasks_by_job)  # CR-09 warm pool (coordinator only)
        self._month_memo = None  # outside a pass the monthly spend is computed fresh
        return changes

    def _tick_job(
        self,
        job: JobRecord,
        tasks: List[TaskRecord],
        running: Optional[Dict[str, int]] = None,
        starting: Optional[Dict[str, int]] = None,
    ) -> int:
        running = count_by_target(tasks, ACTIVE) if running is None else running
        starting = count_by_target(tasks, (TaskState.PLACING, TaskState.STARTING)) if starting is None else starting
        changes = 0 if self.pool_mode else self._accrue(job, tasks)  # CR-10 (the coordinator accrues)
        job = self.cp.get_job(job.job_id)
        mine = [t for t in tasks if not self.pool_mode or t.lease_owner == self.owner]  # a pool worker's own tasks
        if self.pool_mode:
            self._pool_heartbeat(mine)
        if job.cancel_requested:
            for task in tasks:
                if task.state == TaskState.QUEUED:
                    self._set(task, TaskState.CANCELLED, finished_at=self.clock(), message="cancelled")
                    changes += 1
            for task in mine:
                if task.state in ACTIVE and task.backend_handle:
                    self._task_backend(job, task).cancel(task.backend_handle)
        for task in mine:
            if task.state in (TaskState.STARTING, TaskState.RUNNING) and task.backend_handle:
                changes += self._poll(job, task, self._task_backend(job, task))
        if job.cancel_requested:
            return changes
        current = self.cp.list_tasks(job.job_id)
        active = sum(1 for t in current if t.state in ACTIVE)
        limit = self._parallelism(job, current, running)  # CR-09 auto-scaling cap
        if self.pool_mode:
            limit = min(
                limit, self.pool_slots - sum(1 for t in current if t.state in ACTIVE and t.lease_owner == self.owner)
            )
        while active < limit:
            leased = self.cp.lease_task(self.owner, _LEASE_SECONDS, job_id=job.job_id)
            if leased is None:
                break
            started = self._place_and_start(job, leased, running, starting)
            if not started:
                break
            active += 1
            changes += 1
        return changes

    def _submit_attempt(self, backend: ComputeBackend, view: JobRecord, task: TaskRecord) -> str:
        """Launch one attempt of ``task`` on ``backend`` with ``view``'s (target-specific) spec."""
        if self.pool_mode:
            # A pool worker runs the task runtime (staging, license, output mirror) as a local process.
            launch = self._remote_launch(view, task) if self.remote is not None else self._launch(view, task)
            launch.workdir = str(self.task_dir(task))
            return backend.submit(launch)
        if callable(getattr(backend, "configure_recovery", None)):
            with renew_provisioning_lease(self.cp, task, _LEASE_SECONDS):
                return backend.submit(self._launch(view, task))
        return backend.submit(self._launch(view, task))

    def _owns_placing(self, task: TaskRecord) -> bool:
        current = self.cp.get_task(task.task_id)
        return (
            current.state == TaskState.PLACING
            and current.attempt == task.attempt
            and current.lease_owner == task.lease_owner
        )

    def task_dir(self, task: TaskRecord) -> Path:
        """Local directory of a task's attempts and outputs."""
        return self.workroot / task.job_id / task.task_id

    def _poll(self, job: JobRecord, task: TaskRecord, backend: ComputeBackend) -> int:
        handle = task.backend_handle
        assert handle is not None
        status = backend.poll(handle)
        now = self.clock()
        limits = job.spec.resources
        if status.state == AttemptState.UNKNOWN and job.cancel_requested:
            # A backend that stops attempts by deleting them (Batch jobs, VMs) cannot answer for one
            # cancelled before a restart: the cancel was asked for, so the attempt is cancelled.
            self._set(task, TaskState.CANCELLED, finished_at=now, message="cancelled (attempt no longer visible)")
            return 1
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
        self._record_outcome(job, task, status.state.value)  # CR-09 target health (succeeded/preempted/failed)
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
        if Provider(self._task_spec(job, task).provider) != Provider.LOCAL:
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

    # Management (status, metrics and price hints are in ServiceStatusMixin) ----------

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
        return self._task_backend(self.cp.get_job(task.job_id), task).logs(task.backend_handle, tail)

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
