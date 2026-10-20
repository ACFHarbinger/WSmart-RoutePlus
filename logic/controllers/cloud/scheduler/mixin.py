"""Scheduling decisions of :class:`.service.CloudService` (CR-09).

- **Priorities:** jobs are served in order of ``JobSpec.priority`` (higher first), then age,
  so a high-priority job takes parallelism and quota before a low-priority one.
- **Placement:** :meth:`SchedulerMixin._place_and_start` ranks the job's targets
  (:mod:`.placement`) and tries them in order; capacity errors and launch failures are
  recorded in :class:`TargetHealth` and spill the task to the next target. The chosen
  target's key is stored on the task (``TaskRecord.placement``), and polling, cancelling
  and pricing use it.
- **Auto-scaling:** a job's running tasks grow with its queue up to the smallest of
  ``max_parallel``, its targets' free quota and, with a budget, the tasks its remaining
  budget can pay for at their estimated cost (CR-10). The scheduler never stops a running
  task to shrink; it starts fewer.

Attributes:
    SchedulerMixin: The service methods.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from logic.controllers.cloud.interfaces import CapacityError, ComputeBackend
from logic.controllers.cloud.model import (
    ACTIVE,
    ComputeKind,
    JobRecord,
    JobSpec,
    PlacementTarget,
    Provider,
    TaskRecord,
    TaskState,
)
from logic.controllers.cloud.scheduler.placement import TargetHealth, rank_targets, spec_for, targets_of

# A pool worker runs every task it leases as a local process on its own machine.
POOL_TARGET = PlacementTarget(Provider.LOCAL, ComputeKind.PROCESS)


class SchedulerMixin:
    """Placement, priorities and auto-scaling; needs the service's ``cp``, ``backends``, ``clock``,
    ``_launch``, ``_set``, ``_event``, ``_owns_placing`` and ``_get_price_per_hour``."""

    health: TargetHealth

    # Targets --------------------------------------------------------------

    def _task_target(self, job: JobRecord, task: TaskRecord) -> PlacementTarget:
        if getattr(self, "pool_mode", False):
            return POOL_TARGET
        targets = targets_of(job.spec)
        return next((t for t in targets if t.key() == task.placement), targets[0])

    def _task_spec(self, job: JobRecord, task: TaskRecord) -> JobSpec:
        """The spec of the target ``task``'s current attempt runs on (a pool job keeps its own)."""
        if getattr(self, "pool_mode", False) or not job.spec.placement:
            return job.spec
        return spec_for(job.spec, self._task_target(job, task))

    def _task_backend(self, job: JobRecord, task: TaskRecord) -> ComputeBackend:
        target = self._task_target(job, task)
        return self.backends[(Provider(target.provider), ComputeKind(target.compute))]

    @staticmethod
    def _by_priority(jobs: Sequence[JobRecord]) -> List[JobRecord]:
        return sorted(jobs, key=lambda j: (-int(j.spec.priority or 0), j.created_at or 0.0))

    # Auto-scaling ---------------------------------------------------------

    def _parallelism(self, job: JobRecord, tasks: Sequence[TaskRecord], running: Dict[str, int]) -> int:
        """How many of the job's tasks may run now (see the module docstring)."""
        cap = job.spec.max_parallel
        quotas = [t.max_tasks for t in targets_of(job.spec)]
        if all(q is not None for q in quotas):
            mine = {t.key(): 0 for t in targets_of(job.spec)}
            for task in tasks:
                if task.state in ACTIVE and task.placement in mine:
                    mine[task.placement] += 1
            free = sum(max(0, q - running.get(t.key(), 0)) for t, q in zip(targets_of(job.spec), quotas))
            cap = min(cap, sum(mine.values()) + free)
        budget = job.spec.resources.budget_usd
        if budget is not None:
            per_task = self._estimated_task_cost(job)
            if per_task:
                active = sum(1 for t in tasks if t.state in ACTIVE)
                headroom = budget - self.job_accrued(job, tasks)
                affordable = max(0, math.floor(headroom / per_task))
                cap = min(cap, active + affordable)
                queued = any(t.state == TaskState.QUEUED for t in tasks)
                if affordable == 0 and active == 0 and queued and not job.cancel_requested:
                    # Nothing runs and the rest of the budget cannot pay for one more task: stop, do not wait.
                    data = {
                        "scope": "job",
                        "reason": "exhausted",
                        "headroom_usd": round(headroom, 6),
                        "task_estimate_usd": per_task,
                    }
                    self._event(job.job_id, "budget_stop", data=data)
                    self.cancel(job.job_id)
        return cap

    def _estimated_task_cost(self, job: JobRecord) -> Optional[float]:
        estimates = [e.data for e in self.cp.list_events(job.job_id) if e.kind == "cost_estimate"]
        if not estimates or not estimates[-1].get("cells"):
            return None
        mids = [c["mid"] for c in estimates[-1]["cells"] if c.get("mid")]
        return sum(mids) / len(mids) if mids else None

    # Placement ------------------------------------------------------------

    def _expected_seconds(self, job: JobRecord, task: TaskRecord) -> Optional[float]:
        return task.cell.expected_wall_seconds

    def _place_and_start(
        self, job: JobRecord, task: TaskRecord, running: Dict[str, int], starting: Dict[str, int]
    ) -> Optional[bool]:
        """Start ``task`` on the best target that accepts it.

        Returns True when it started (or failed for good), False when no target had capacity
        (the task is queued again), None when the lease was lost.
        """
        now = self.clock()
        served = set(self.backends)
        multi = bool(job.spec.placement) and not getattr(self, "pool_mode", False)
        ranked = (
            [POOL_TARGET]
            if getattr(self, "pool_mode", False)
            else rank_targets(
                job.spec,
                served,
                running,
                starting,
                self.health if multi else TargetHealth(),
                now,
                self._price_for,
                self._expected_seconds(job, task),
            )
        )
        reasons: List[str] = []
        last_error: Optional[Exception] = None
        launch_failed = 0
        for target in ranked:
            key = target.key()
            if multi:
                task = self.cp.update_task(task.task_id, placement=key)
            backend = self.backends[(Provider(target.provider), ComputeKind(target.compute))]
            spec = job.spec if not multi else spec_for(job.spec, target)
            view = JobRecord(job_id=job.job_id, spec=spec, created_at=job.created_at)
            try:
                handle = self._submit_attempt(backend, view, task)
            except CapacityError as exc:
                self.health.record(key, "capacity", now)
                reasons.append(f"{key}: no capacity ({exc})")
                last_error = exc
                continue
            except Exception as exc:  # noqa: BLE001 - a broken target spills over; the scheduler keeps going
                self.health.record(key, "launch_failed", now)
                reasons.append(f"{key}: launch failed ({exc})")
                last_error, launch_failed = exc, launch_failed + 1
                continue
            if not self._owns_placing(task):
                return None
            self.health.record(key, "launched", now)
            self._set(task, TaskState.STARTING, backend_handle=handle, started_at=now, heartbeat=now)
            confirm = getattr(backend, "confirm_provision", None)
            if callable(confirm):
                confirm(handle)
            running[key] = running.get(key, 0) + 1
            starting[key] = starting.get(key, 0) + 1
            if len(ranked) > 1 or reasons:
                self._event(job.job_id, "placement", task.task_id, {"target": key, "skipped": reasons})
            return True
        if not self._owns_placing(task):
            return None
        if launch_failed and launch_failed == len(reasons) == len(targets_of(job.spec)):
            # Every target refused the launch itself (not capacity): the task fails, as with one backend.
            detail = str(last_error) if not multi else "; ".join(reasons)
            self._set(task, TaskState.FAILED, finished_at=now, message=f"launch failed: {detail}")
            return True
        if not multi and last_error is not None:
            message = f"waiting for capacity: {last_error}"
        else:
            message = "waiting for capacity: " + (
                "; ".join(reasons) if reasons else "every target is full or cooling down"
            )
        self._set(task, TaskState.QUEUED, message=message, lease_owner=None)
        return False

    def _price_for(self, spec: JobSpec) -> Optional[float]:
        if Provider(spec.provider) == Provider.LOCAL:
            return 0.0
        return self._get_price_per_hour(spec, spot=bool(spec.resources.spot))

    def _record_outcome(self, job: JobRecord, task: TaskRecord, outcome: str) -> None:
        self.health.record(self._task_target(job, task).key(), outcome, self.clock())

    def _submit_attempt(self, backend: ComputeBackend, view: JobRecord, task: TaskRecord) -> str:
        raise NotImplementedError  # provided by CloudService (it holds the provisioning-lease logic)
