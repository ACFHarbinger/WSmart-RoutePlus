"""Scheduler (CR-09): placement across providers, priorities, auto-scaling, packing and the warm pool.

Attributes:
    TargetHealth: Per-target circuit breaker and preemption rates.
    rank_targets: A job's eligible placement targets, best first.
    targets_of: A job's placement targets.
    spec_for: A job spec as one target sees it.
    pack: First-fit decreasing packing of tasks onto machines.
    slots_per_machine: Tasks of one size that fit on a machine.
    PoolPolicy: The warm worker pool.
    run_pool_worker: A pool worker's loop.
    SchedulerMixin, PoolMixin: The service methods.
"""

from logic.controllers.cloud.scheduler.mixin import POOL_TARGET, SchedulerMixin
from logic.controllers.cloud.scheduler.packing import pack, slots_per_machine
from logic.controllers.cloud.scheduler.placement import (
    TargetHealth,
    count_by_target,
    rank_targets,
    spec_for,
    targets_of,
)
from logic.controllers.cloud.scheduler.pool import PoolMixin, PoolPolicy, run_pool_worker

__all__ = [
    "POOL_TARGET",
    "PoolMixin",
    "PoolPolicy",
    "SchedulerMixin",
    "TargetHealth",
    "count_by_target",
    "pack",
    "rank_targets",
    "run_pool_worker",
    "slots_per_machine",
    "spec_for",
    "targets_of",
]
