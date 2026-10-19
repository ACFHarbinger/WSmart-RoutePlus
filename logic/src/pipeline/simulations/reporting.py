"""What the simulator records about a day and a cell.

Solver status: each routing algorithm publishes its per-day solver status and
optimality certificate, and the day logger reads them. The values live in
:class:`contextvars.ContextVar` objects so a worker process keeps its own status
and a later day cannot inherit the previous day's code.

Sample aggregation, shared by the finishing state and the test orchestrator:
failed samples keep their partial metrics and stay out of the mean. An
all-failed cell has no numeric estimate.
"""

import math
from contextvars import ContextVar
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from logic.src.pipeline.learning.envs.common.metrics import (
    DEFAULT_METRICS,
    METRICS,
    MetricSummary,
    SampleResult,
    summarize_samples,
)

# Gurobi model status codes (gurobipy.GRB). Names are stored so the logger
# does not need a solver license to decode a recorded day.
GUROBI_STATUS_NAMES = {
    1: "LOADED",
    2: "OPTIMAL",
    3: "INFEASIBLE",
    4: "INF_OR_UNBD",
    5: "UNBOUNDED",
    6: "CUTOFF",
    7: "ITERATION_LIMIT",
    8: "NODE_LIMIT",
    9: "TIME_LIMIT",
    10: "SOLUTION_LIMIT",
    11: "INTERRUPTED",
    12: "NUMERIC",
    13: "SUBOPTIMAL",
    15: "USER_OBJ_LIMIT",
    17: "MEM_LIMIT",
}

# OR-Tools MPSolver status codes (pywraplp.Solver).
ORTOOLS_STATUS_NAMES = {
    0: "OPTIMAL",
    1: "FEASIBLE",
    2: "INFEASIBLE",
    3: "UNBOUNDED",
    4: "ABNORMAL",
    6: "NOT_SOLVED",
}

_solver_status: ContextVar[Optional[str]] = ContextVar("wsmart_solver_status", default=None)


def reset_solver_status() -> None:
    """Clear the status before a day so a previous day cannot leak forward."""
    _solver_status.set(None)
    reset_solver_certificate()


def note_solver_status(status: Optional[str], append: bool = False) -> None:
    """Record the status string for the day that is being solved.

    Args:
        status: Backend status, or None to clear it.
        append: Keep a preceding solve status when recording a retry.
    """
    previous = _solver_status.get()
    if append and previous and status is not None:
        _solver_status.set(f"{previous} -> {status}")
    else:
        _solver_status.set(None if status is None else str(status))


def current_solver_status() -> Optional[str]:
    """Return the status noted for the current context, if any."""
    return _solver_status.get()


def format_backend_status(backend: str, code: object) -> str:
    """Turn a backend status code into a stable ``backend:NAME`` string.

    Args:
        backend: ``gurobi``, ``ortools``, or ``pyomo``.
        code: Numeric status, or a Pyomo termination condition.

    Returns:
        A short status token written into the daily log.
    """
    if backend == "gurobi":
        try:
            number = int(code)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return f"gurobi:{code}"
        return f"gurobi:{GUROBI_STATUS_NAMES.get(number, str(number))}"
    if backend == "ortools":
        try:
            number = int(code)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return f"ortools:{code}"
        return f"ortools:{ORTOOLS_STATUS_NAMES.get(number, str(number))}"
    name = getattr(code, "name", None)
    if isinstance(name, str) and name:
        return f"{backend}:{name}"
    return f"{backend}:{code}"


_solver_certificate: ContextVar[Optional[Dict[str, Any]]] = ContextVar("wsmart_solver_certificate", default=None)
_solver_instance: ContextVar[Optional[str]] = ContextVar("wsmart_solver_instance", default=None)


def reset_solver_certificate(instance: Optional[str] = None) -> None:
    """Clear proof metadata before each independent construction."""
    _solver_certificate.set(None)
    _solver_instance.set(instance)


def note_root_bound(bound: float) -> None:
    """Record a root CG upper bound only after certified pricing exhaustion."""
    if not math.isfinite(bound):
        return
    old = _solver_certificate.get()
    best = min(bound, old["bound"]) if old else bound
    _solver_certificate.set({"bound": best, "source": "root_cg_exhausted", "instance": _solver_instance.get()})


def note_solver_incumbent(incumbent: float) -> None:
    """Pair a returned constructor incumbent with the recorded root proof."""
    proof = _solver_certificate.get()
    if proof is not None and math.isfinite(incumbent) and proof["bound"] >= incumbent - 1e-6:
        _solver_certificate.set(dict(proof, incumbent=float(incumbent)))
    else:
        _solver_certificate.set(None)


def current_solver_certificate() -> Optional[Dict[str, Any]]:
    """Return this construction's proof; incomplete pricing produces no certificate."""
    proof = _solver_certificate.get()
    return dict(proof) if proof is not None and "incumbent" in proof else None


def coerce_metrics(payload: Any) -> Dict[str, float]:
    """Read canonical KPI numbers from a dict or a legacy positional vector."""
    from logic.src.constants import SIM_METRICS

    if isinstance(payload, Mapping):
        out: Dict[str, float] = {}
        for key, value in payload.items():
            if key in METRICS and isinstance(value, (int, float)) and not isinstance(value, bool):
                out[str(key)] = float(value)
        return out
    if isinstance(payload, (list, tuple)):
        return {
            key: float(value)
            for key, value in zip(SIM_METRICS, payload, strict=False)
            if key in METRICS and isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    raise TypeError(f"Cannot read metrics from {type(payload).__name__}")


def sample_from_payload(payload: Any, sample_id: str) -> SampleResult:
    """Build one sample record. A mapping may flag itself ``status: failed``."""
    status = "completed"
    failed_day: Optional[int] = None
    exception_type: Optional[str] = None
    if isinstance(payload, Mapping):
        if payload.get("status") == "failed":
            status = "failed"
        if payload.get("failed_day") is not None:
            failed_day = int(payload["failed_day"])
        if payload.get("exception_type"):
            exception_type = str(payload["exception_type"])
        if payload.get("sample_id") is not None:
            sample_id = str(payload["sample_id"])
    return SampleResult(
        sample_id=sample_id,
        metrics=coerce_metrics(payload),
        status=status,
        failed_day=failed_day,
        exception_type=exception_type,
    )


def summarize_payloads(
    payloads: Iterable[Any],
    names: Sequence[str] = DEFAULT_METRICS,
) -> MetricSummary:
    """Equal-weight summary. Failed payloads count in ``n_failed`` only."""
    samples: List[SampleResult] = []
    for index, payload in enumerate(payloads):
        samples.append(sample_from_payload(payload, str(index)))
    return summarize_samples(samples, names)


def summary_pair(
    payloads: Iterable[Any],
    names: Sequence[str] = DEFAULT_METRICS,
) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[float]], MetricSummary]:
    """Mean and sample std dicts plus the full summary (includes n_failed)."""
    summary = summarize_payloads(payloads, names)
    return dict(summary.mean), dict(summary.std), summary
