"""Budget caps, warnings and cost accrual of :class:`.service.CloudService` (CR-10).

**Caps.** A job's cap is ``Resources.budget_usd``; the monthly cap and the warning
fractions are a :class:`BudgetPolicy`. A month is the UTC calendar month a job was
created in, and its spend is the accrued cost of that month's jobs.

**At submit** the job's p50 estimate (:mod:`.estimate`) must fit under its own cap and
under what is left of the monthly cap, or the submit raises :class:`BudgetExceeded`; a p90
over a cap only warns (``budget_warning`` with ``"estimate": true``). What is left of the
month counts every unfinished job of the month at the larger of its p50 estimate and its
accrued cost (:meth:`CostMixin.month_committed`), so queued jobs cannot overcommit it.

**While running** each scheduling pass prices every task's attempts (scheduler-observed
time x hourly price, the same estimate the status view shows) and:

- writes ``kind: "cost"`` records (``accrued_usd``, ``rate_usd_h``, ``elapsed_s``) to the
  task's metric stream, at most every ``ACCRUAL_SECONDS`` and once more when it finishes;
- emits one ``budget_warning`` event per scope and fraction crossed;
- cancels the job when its accrued cost reaches its cap, and every running job of the
  month when the month's reaches the monthly cap (``budget_stop`` events).

Accrued cost is an estimate from list prices, not an invoice; :mod:`.billing` imports
the providers' billing exports when they are configured.

Attributes:
    BudgetPolicy: Monthly cap and warning fractions.
    BudgetExceeded: Raised by submit.
    CostMixin: The service methods.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from logic.controllers.cloud.cost.estimate import CostEstimate, estimate_job
from logic.controllers.cloud.model import ACTIVE, JobRecord, JobSpec, Provider, TaskRecord, TaskState

ACCRUAL_SECONDS = 60.0


@dataclass
class BudgetPolicy:
    """Account-wide limits.

    Attributes:
        monthly_usd: Cap on a calendar month's accrued cost (None: no cap).
        warn_fractions: Fractions of a cap at which a warning event is emitted.
    """

    monthly_usd: Optional[float] = None
    warn_fractions: Tuple[float, ...] = (0.5, 0.8, 0.95)


class BudgetExceeded(ValueError):
    """A job's estimate does not fit under its cap or the month's remaining budget."""


def _month(at: Optional[float]) -> str:
    return _dt.datetime.fromtimestamp(at or 0.0, tz=_dt.timezone.utc).strftime("%Y-%m")


class CostMixin:
    """Estimates, caps and accrual; needs ``cp``, ``clock``, ``predict``, ``cancel``, ``_event``,
    ``_get_price_per_hour``, ``_task_duration_and_cost`` and an optional ``budget``."""

    budget: Optional[BudgetPolicy] = None
    _cost_written: Dict[str, Any]

    # Estimates ------------------------------------------------------------

    def estimate(self, spec: JobSpec, predictions: Optional[Sequence[Any]] = None) -> Optional[CostEstimate]:
        """The pre-launch estimate of ``spec`` (None without a predictor)."""
        predictions = self.predict(spec) if predictions is None else predictions
        if predictions is None:
            return None
        spot = bool(spec.resources.spot)
        # A local process costs nothing to run; a remote price may be unknown (None).
        price = 0.0 if Provider(spec.provider) == Provider.LOCAL else self._get_price_per_hour(spec, spot=spot)
        return estimate_job(predictions, price, spot, spec.resources.machine_type)

    def _check_budget(self, spec: JobSpec, estimate: Optional[CostEstimate]) -> List[Dict[str, Any]]:
        """Raise :class:`BudgetExceeded` if the p50 estimate breaks a cap; returns p90 warnings."""
        policy = self.budget
        if estimate is None or estimate.mid is None:
            if spec.resources.budget_usd is not None:
                raise BudgetExceeded("the job has a budget cap but no cost estimate (no runtime prediction or price)")
            if (
                policy is not None
                and policy.monthly_usd is not None
                and Provider(spec.provider) != Provider.LOCAL
                and (policy.monthly_usd - self.month_committed(self.clock())) <= 0
            ):
                remaining = policy.monthly_usd - self.month_committed(self.clock())
                raise BudgetExceeded(f"monthly budget exceeded ({remaining:.2f} USD left)")
            return []
        warnings = []
        caps = [("job", spec.resources.budget_usd)]
        if policy is not None and policy.monthly_usd is not None:
            caps.append(("month", policy.monthly_usd - self.month_committed(self.clock())))
        for scope, cap in caps:
            if cap is None:
                continue
            if Provider(spec.provider) != Provider.LOCAL and estimate.mid > cap:
                raise BudgetExceeded(
                    f"estimated {estimate.mid:.2f} USD exceeds the {scope} budget ({cap:.2f} USD left)"
                )
            if Provider(spec.provider) != Provider.LOCAL and estimate.high is not None and estimate.high > cap:
                warnings.append({"scope": scope, "estimate": True, "high_usd": estimate.high, "cap_usd": cap})
        return warnings

    # Accrual --------------------------------------------------------------

    def _task_costs(
        self, job: JobRecord, tasks: Sequence[TaskRecord]
    ) -> Dict[str, Tuple[Optional[float], Optional[float], float]]:
        """``task_id -> (elapsed seconds, accrued USD, USD/h)`` from the scheduler's attempt history."""
        task_spec = getattr(self, "_task_spec", None)  # CR-09: a placed attempt is priced at its target
        events: Dict[str, List[Any]] = {}
        for event in self.cp.list_events(job.job_id):
            if event.task_id is not None:
                events.setdefault(event.task_id, []).append(event)
        out = {}
        for task in tasks:
            spec = task_spec(job, task) if task_spec is not None else job.spec
            price = self._get_price_per_hour(spec, spot=bool(spec.resources.spot))
            elapsed, cost = self._task_duration_and_cost(
                task, events.get(task.task_id, []), price, Provider(spec.provider)
            )
            out[task.task_id] = (elapsed, cost, price)
        return out

    def job_accrued(self, job: JobRecord, tasks: Optional[Sequence[TaskRecord]] = None) -> float:
        tasks = self.cp.list_tasks(job.job_id) if tasks is None else tasks
        return round(sum(c or 0.0 for _, c, _ in self._task_costs(job, tasks).values()), 6)

    def month_accrued(self, now: float) -> float:
        """Accrued cost of every job created in ``now``'s UTC month (once per scheduling pass)."""
        month = _month(now)
        memo = getattr(self, "_month_memo", None)
        if memo is not None and month in memo:
            return memo[month]
        jobs = [j for j in self.cp.list_jobs(limit=1000) if _month(j.created_at) == month]
        total = round(sum(self.job_accrued(j) for j in jobs), 6)
        if memo is not None:
            memo[month] = total
        return total

    def month_committed(self, now: float) -> float:
        """The month's accrued cost, with unfinished jobs at max(p50 estimate, accrued)."""
        month = _month(now)
        memo = getattr(self, "_month_memo", None)
        memo_key = f"committed_{month}"
        if memo is not None and memo_key in memo:
            return memo[memo_key]
        total = 0.0
        for job in self.cp.list_jobs(limit=1000):
            if _month(job.created_at) != month:
                continue
            tasks = self.cp.list_tasks(job.job_id)
            accrued = self.job_accrued(job, tasks)
            if any(t.state not in (TaskState.SUCCEEDED, TaskState.FAILED, TaskState.CANCELLED) for t in tasks):
                estimates = [e.data.get("mid") for e in self.cp.list_events(job.job_id) if e.kind == "cost_estimate"]
                accrued = max(accrued, estimates[-1] or 0.0) if estimates else accrued
            total += accrued
        res = round(total, 6)
        if memo is not None:
            memo[memo_key] = res
        return res

    def _accrue(self, job: JobRecord, tasks: Sequence[TaskRecord]) -> int:
        """Write cost records, emit warnings and stop the job at its caps; returns state changes (0)."""
        now = self.clock()
        written = self._cost_written
        costs = self._task_costs(job, tasks)
        for task in tasks:
            elapsed, accrued, rate = costs[task.task_id]
            if accrued is None:
                continue
            last = written.get(task.task_id)
            final = task.state not in ACTIVE
            if (final and last != "final") or (not final and (last is None or now - last >= ACCRUAL_SECONDS)):
                record = {
                    "at": now,
                    "kind": "cost",
                    "accrued_usd": accrued,
                    "rate_usd_h": rate,
                    "elapsed_s": round(elapsed, 1),
                }
                self.cp.put_metrics(task.task_id, [record])
                written[task.task_id] = "final" if final else now
        total = round(sum(c or 0.0 for _, c, _ in costs.values()), 6)
        self._enforce("job", job, total, job.spec.resources.budget_usd)
        policy = self.budget
        if policy is not None and policy.monthly_usd is not None and any(t.state in ACTIVE for t in tasks):
            spent = self.month_accrued(now)
            if spent >= policy.monthly_usd:
                for other in self.cp.list_jobs(limit=1000):
                    if _month(other.created_at) == _month(now) and any(
                        t.state in ACTIVE for t in self.cp.list_tasks(other.job_id)
                    ):
                        self._enforce("month", other, spent, policy.monthly_usd)
            else:
                self._enforce("month", job, spent, policy.monthly_usd)
        return 0

    def _enforce(self, scope: str, job: JobRecord, accrued: float, cap: Optional[float]) -> None:
        if cap is None or cap <= 0:
            return
        seen = {
            (e.data.get("scope"), e.data.get("fraction"))
            for e in self.cp.list_events(job.job_id)
            if e.kind == "budget_warning"
        }
        for fraction in (self.budget or BudgetPolicy()).warn_fractions:
            if accrued >= fraction * cap and (scope, fraction) not in seen:
                self._event(
                    job.job_id,
                    "budget_warning",
                    data={"scope": scope, "fraction": fraction, "accrued_usd": accrued, "cap_usd": cap},
                )
        if accrued >= cap and not self.cp.get_job(job.job_id).cancel_requested:
            self._event(job.job_id, "budget_stop", data={"scope": scope, "accrued_usd": accrued, "cap_usd": cap})
            self.cancel(job.job_id)
