"""Warm worker pool (CR-09).

A job with ``JobSpec.pool`` is not launched task by task. Instead long-lived **pool
workers** lease its tasks from the control plane and run them on their own machine:

- a worker is a :class:`.service.CloudService` with ``pool_mode=True`` and a local process
  backend (``cloud worker --pool --slots N --idle S``). It ticks pool jobs only, polls only
  the tasks it leased, refreshes their heartbeats every tick, runs up to ``N`` at once and
  exits after ``S`` seconds with nothing to do (scale to zero);
- the coordinator (``pool_mode=False``) never starts pool tasks. With a :class:`PoolPolicy`
  it keeps enough workers alive: the pool tasks that are queued or running are packed onto
  worker machines (:func:`.packing.pack`, CR-11 memory predictions), and the deficit,
  capped by ``max_workers``, is launched on the policy's target. Workers are never killed;
  they stop themselves when idle;
- a pool task whose heartbeat is older than ``heartbeat_timeout`` belongs to a lost worker:
  the coordinator fails the attempt and the retry policy requeues it (with resume).

Attributes:
    PoolPolicy: The pool's machine, size limits and timeouts.
    PoolMixin: Coordinator scaling, worker heartbeats and stale-task recovery.
    run_pool_worker: A pool worker's loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from logic.controllers.cloud.interfaces import AttemptState, ComputeBackend, ControlPlane, LaunchSpec
from logic.controllers.cloud.model import (
    ACTIVE,
    CellSpec,
    ComputeKind,
    JobRecord,
    PlacementTarget,
    Provider,
    Resources,
    TaskRecord,
    TaskState,
)
from logic.controllers.cloud.scheduler.packing import pack, slots_per_machine

FINISHED_ATTEMPT = (AttemptState.SUCCEEDED, AttemptState.FAILED, AttemptState.PREEMPTED)


@dataclass
class PoolPolicy:
    """The warm pool.

    Attributes:
        target: Where workers run (a VM or batch target).
        cpus: vCPUs of one worker machine.
        memory_gb: Memory of one worker machine.
        max_workers: Most workers alive at once.
        min_workers: Workers kept even when idle (warm capacity).
        idle_seconds: A worker exits after this long with nothing to run.
        heartbeat_timeout: A pool task silent this long is requeued (its worker is lost).
        image: Image workers run (the release image).
    """

    target: PlacementTarget
    cpus: float = 4.0
    memory_gb: float = 16.0
    max_workers: int = 4
    min_workers: int = 0
    idle_seconds: float = 300.0
    heartbeat_timeout: float = 600.0
    image: Optional[str] = None
    extra_env: Dict[str, str] = field(default_factory=dict)


class PoolMixin:
    """Pool coordination for :class:`.service.CloudService`."""

    # Annotations only: CloudService.__init__ and its other mixins own these at runtime.
    pool: Optional[PoolPolicy] = None
    cp: ControlPlane
    clock: Callable[[], float]
    owner: str
    pool_mode: bool
    backends: Dict[Tuple[Provider, ComputeKind], ComputeBackend]
    _pool_workers: Dict[str, float]
    _event: Callable[..., None]
    _fail: Callable[..., int]
    _ensure_running: Callable[[TaskRecord], TaskRecord]

    def _pool_heartbeat(self, tasks: Sequence[TaskRecord]) -> None:
        """A pool worker vouches for its running tasks every tick."""
        now = self.clock()
        for task in tasks:
            if task.state in ACTIVE and task.lease_owner == self.owner:
                self.cp.update_task(task.task_id, heartbeat=now)

    def _requeue_stale_pool_tasks(self, job: JobRecord, tasks: Sequence[TaskRecord]) -> int:
        """Fail (and so retry) pool attempts whose worker stopped reporting."""
        timeout = self.pool.heartbeat_timeout if self.pool is not None else 600.0
        now, changes = self.clock(), 0
        for task in tasks:
            if (
                task.state in (TaskState.STARTING, TaskState.RUNNING)
                and task.heartbeat
                and now - task.heartbeat > timeout
            ):
                changes += self._fail(
                    job, self._ensure_running(task), f"pool worker lost (no heartbeat for {timeout:.0f}s)"
                )
        return changes

    def _pool_demand(self, jobs: Sequence[JobRecord], tasks_by_job: Dict[str, List[TaskRecord]]) -> tuple:
        """``(machines needed, slots per machine)`` for the queued and running pool tasks."""
        assert self.pool is not None
        items, slots = [], []
        for job in jobs:
            if not job.spec.pool or job.cancel_requested:
                continue
            memory = self._pool_memory(job)
            for task in tasks_by_job[job.job_id]:
                if task.state == TaskState.QUEUED or task.state in ACTIVE:
                    need = memory.get(task.cell.cell_id, job.spec.resources.memory_gb)
                    items.append((task.task_id, float(job.spec.resources.cpus), float(need)))
                    slots.append(slots_per_machine(self.pool.cpus, self.pool.memory_gb, job.spec.resources.cpus, need))
        return len(pack(items, self.pool.cpus, self.pool.memory_gb)), (min(slots) if slots else 1)

    def _pool_memory(self, job: JobRecord) -> Dict[str, float]:
        """CR-11 predicted memory per cell, from the job's ``prediction`` event."""
        events = [e for e in self.cp.list_events(job.job_id) if e.kind == "prediction"]
        cells = events[-1].data.get("cells", {}) if events else {}
        return {cell_id: p["memory_gb"] for cell_id, p in cells.items() if p.get("memory_gb")}

    def _scale_pool(self, jobs: Sequence[JobRecord], tasks_by_job: Dict[str, List[TaskRecord]]) -> int:
        """Launch workers up to the packed demand; returns how many were launched."""
        if self.pool is None or self.pool_mode:
            return 0
        workers = self._pool_workers  # handle -> launched at
        backend = self.backends.get((Provider(self.pool.target.provider), ComputeKind(self.pool.target.compute)))
        if backend is None:
            return 0
        for handle in list(workers):
            if backend.poll(handle).state in FINISHED_ATTEMPT:
                workers.pop(handle)
        needed, slots = self._pool_demand(jobs, tasks_by_job)
        desired = min(self.pool.max_workers, max(self.pool.min_workers, needed))
        launched = 0
        while len(workers) < desired:
            handle = backend.submit(self._worker_launch(slots, len(workers)))
            workers[handle] = self.clock()
            launched += 1
        if launched:
            self._event("pool", "pool_scale", data={"workers": len(workers), "needed": needed, "slots": slots})
        return launched

    def _worker_launch(self, slots: int, index: int) -> LaunchSpec:
        assert self.pool is not None
        target = self.pool.target
        remote = getattr(self, "remote", None)
        env = dict(self.pool.extra_env)
        if remote is not None:
            env.update(WSR_CLOUD_CONTROL_PLANE=remote.control_plane_uri, WSR_CLOUD_STORE=remote.store_uri)
        argv = [
            "python",
            "-m",
            "logic.controllers.cloud",
            "worker",
            "--pool",
            "--slots",
            str(slots),
            "--idle",
            str(self.pool.idle_seconds),
        ]
        worker = TaskRecord(
            task_id=f"pool-worker-{int(self.clock())}-{index}",
            job_id="pool",
            cell=CellSpec(cell_id="pool-worker", argv=["test_sim"]),
        )
        resources = Resources(
            cpus=self.pool.cpus,
            memory_gb=self.pool.memory_gb,
            spot=target.spot,
            machine_type=target.machine_type,
            region=target.region,
        )
        return LaunchSpec(task=worker, resources=resources, argv=argv, env=env, image=self.pool.image)


def run_pool_worker(
    service: Any,
    idle_seconds: float,
    poll_seconds: float = 5.0,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    max_ticks: Optional[int] = None,
) -> str:
    """Tick ``service`` (``pool_mode=True``) until it has had nothing to do for ``idle_seconds``."""
    idle_since: Optional[float] = None
    ticks = 0
    while max_ticks is None or ticks < max_ticks:
        service.tick()
        ticks += 1
        busy = False
        for job in service.cp.list_jobs(limit=1000):
            if not job.spec.pool:
                continue
            for task in service.cp.list_tasks(job.job_id):
                if task.state == TaskState.QUEUED or (task.state in ACTIVE and task.lease_owner == service.owner):
                    busy = True
                    break
            if busy:
                break
        now = clock()
        if busy:
            idle_since = None
        elif idle_since is None:
            idle_since = now
        elif now - idle_since >= idle_seconds:
            return f"idle for {idle_seconds:.0f}s"
        sleep(poll_seconds)
    return "max ticks"
