"""Status, price hints and telemetry views of :class:`.service.CloudService` (moved from ``service.py``).

Attributes:
    ServiceStatusMixin: ``status``, ``metrics`` and the price and duration helpers they use.
    extract_policy_from_cell: The single routing algorithm a cell asks for, if any.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from logic.controllers.cloud.model import (
    ACTIVE,
    CellSpec,
    ComputeKind,
    Event,
    JobSpec,
    Provider,
    TaskRecord,
    job_state,
)


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


class ServiceStatusMixin:
    """Read-only job views of :class:`.service.CloudService`; needs ``cp``, ``backends`` and ``clock``."""

    def _get_price_per_hour(self, spec: JobSpec, spot: bool) -> Optional[float]:
        prices = getattr(self, "prices", None)  # CR-10 price table: overrides first, then static tables
        if prices is not None:
            price = prices.price_per_hour(
                Provider(spec.provider).value, spec.resources.machine_type, spec.resources.region, spot
            )
            if price is not None:
                return price
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
        if not intervals:
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
            resource = self._latest_resource(t.task_id)
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
                    "cpu_percent": resource.get("cpu_percent"),
                    "memory_rss_mb": resource.get("memory_rss_mb"),
                    "progress": dict(t.progress),
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

    def _latest_resource(self, task_id: str) -> Dict[str, Any]:
        """The task's most recent resource record (CR-12), or an empty dict."""
        try:
            records = self.cp.get_metrics(task_id, limit=20, reverse=True)
            for record in records:
                if record.get("kind") == "resource":
                    return record
        except TypeError:
            pass  # Older control planes do not support bounded reads.
        # Progress/day/cost events can outnumber resource samples by any ratio.
        # Keep the bounded fast path, but never mistake a missing page for no sample.
        for record in reversed(self.cp.get_metrics(task_id)):
            if record.get("kind") == "resource":
                return record
        return {}

    def metrics(self, task_id: str, since: Optional[float] = None) -> Dict[str, Any]:
        """A task's telemetry: resource samples, peaks, latest progress and per-day KPIs (CR-12)."""
        from logic.controllers.cloud.telemetry import task_metrics

        return task_metrics(self.cp.get_metrics(task_id, since=since), task_id)
