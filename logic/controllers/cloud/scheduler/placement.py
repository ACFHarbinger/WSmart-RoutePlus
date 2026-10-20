"""Cross-provider placement: target health and ranking (CR-09).

A job's tasks may run on any of its :class:`PlacementTarget` s (``JobSpec.placement``; a
job without one has a single target, its own provider and compute kind). Each start ranks
the eligible targets and tries them in order, so a target that is full or failing spills
the task over to the next one in the same scheduling pass.

**Eligible:** a backend serves the target's provider and compute kind, the target is
under its quota (``max_tasks`` running across all jobs), and its circuit is closed.

**Health (circuit breaker):** ``failure_threshold`` consecutive capacity errors or launch
failures open a target's circuit for ``cooldown_seconds``; a successful launch closes it.
Preemptions on spot targets are tracked over the last ``window`` attempt outcomes.

**Rank:** expected cost of the task there (CR-11 p50 x CR-10 hourly price, inflated by
``1 + 2 x preemption rate`` on spot), then tasks already starting there (queue depth), then
the order the job lists its targets. A target with no known price ranks after priced ones.

Attributes:
    TargetHealth: Circuit breaker and preemption rates per target.
    targets_of: A job's targets.
    spec_for: The job spec as seen by one target (provider, compute, machine type, region, spot).
    rank_targets: Eligible targets, best first.
"""

from __future__ import annotations

import collections
import math
from dataclasses import replace
from typing import Callable, Deque, Dict, Iterable, List, Optional, Set, Tuple

from logic.controllers.cloud.model import ComputeKind, JobSpec, PlacementTarget, Provider

LAUNCH_FAILURES = ("capacity", "launch_failed")


class TargetHealth:
    """Per-target outcomes: ``capacity``, ``launch_failed``, ``launched``, ``succeeded``, ``failed``, ``preempted``.

    Args:
        failure_threshold: Consecutive capacity or launch failures that open the circuit.
        cooldown_seconds: How long an open circuit keeps the target out of placement.
        window: Attempt outcomes kept for the preemption rate.
    """

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 300.0, window: int = 20) -> None:
        self.failure_threshold, self.cooldown_seconds = failure_threshold, cooldown_seconds
        self.consecutive: Dict[str, int] = collections.defaultdict(int)
        self.opened_at: Dict[str, float] = {}
        self.outcomes: Dict[str, Deque[str]] = collections.defaultdict(lambda: collections.deque(maxlen=window))

    def record(self, key: str, outcome: str, at: float) -> None:
        if outcome in LAUNCH_FAILURES:
            self.consecutive[key] += 1
            if self.consecutive[key] >= self.failure_threshold:
                self.opened_at[key] = at
        elif outcome == "launched":
            self.consecutive[key] = 0
            self.opened_at.pop(key, None)
        else:
            self.outcomes[key].append(outcome)

    def available(self, key: str, now: float) -> bool:
        opened = self.opened_at.get(key)
        if opened is None:
            return True
        if now - opened >= self.cooldown_seconds:  # half-open: one more try; a failure re-opens it
            self.opened_at.pop(key, None)
            self.consecutive[key] = self.failure_threshold - 1
            return True
        return False

    def preemption_rate(self, key: str) -> float:
        outcomes = self.outcomes.get(key)
        return sum(o == "preempted" for o in outcomes) / len(outcomes) if outcomes else 0.0


def targets_of(spec: JobSpec) -> List[PlacementTarget]:
    """``spec.placement``, or the job's own provider and compute kind as its one target."""
    if spec.placement:
        return list(spec.placement)
    r = spec.resources
    return [PlacementTarget(Provider(spec.provider), ComputeKind(spec.compute), r.region, r.machine_type, bool(r.spot))]


def spec_for(spec: JobSpec, target: PlacementTarget) -> JobSpec:
    """The spec a backend sees when ``target`` runs the task."""
    resources = replace(
        spec.resources,
        machine_type=target.machine_type or spec.resources.machine_type,
        region=target.region or spec.resources.region,
        spot=bool(target.spot),
    )
    return replace(spec, provider=Provider(target.provider), compute=ComputeKind(target.compute), resources=resources)


def rank_targets(
    spec: JobSpec,
    served: Set[Tuple[Provider, ComputeKind]],
    running: Dict[str, int],
    starting: Dict[str, int],
    health: TargetHealth,
    now: float,
    price: Callable[[JobSpec], Optional[float]],
    expected_seconds: Optional[float],
) -> List[PlacementTarget]:
    """The job's eligible targets, best first (see the module docstring)."""
    ranked = []
    for order, target in enumerate(targets_of(spec)):
        key = target.key()
        if (Provider(target.provider), ComputeKind(target.compute)) not in served:
            continue
        if target.max_tasks is not None and running.get(key, 0) >= target.max_tasks:
            continue
        if not health.available(key, now):
            continue
        hourly = price(spec_for(spec, target))
        if hourly is None:
            cost = math.inf
        else:
            cost = (expected_seconds or 3600.0) / 3600.0 * hourly
            if target.spot:
                cost *= 1.0 + 2.0 * health.preemption_rate(key)
        ranked.append(((cost, starting.get(key, 0), order), target))
    return [target for _, target in sorted(ranked, key=lambda item: item[0])]


def count_by_target(tasks: Iterable, states: Iterable) -> Dict[str, int]:
    """Tasks per placement key among ``states``."""
    wanted = set(states)
    counts: Dict[str, int] = collections.Counter()
    for task in tasks:
        if task.state in wanted and task.placement:
            counts[task.placement] += 1
    return dict(counts)
