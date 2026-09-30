"""Per-day solver status published by a constructor and read by the day logger.

The value lives in a :class:`contextvars.ContextVar` so a worker process keeps
its own status and a later day cannot inherit the previous day's code.
"""

from contextvars import ContextVar
from typing import Optional

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
