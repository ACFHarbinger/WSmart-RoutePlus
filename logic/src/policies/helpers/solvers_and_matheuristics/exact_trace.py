"""Opt-in execution-trace notes for exact routing solvers.

Every helper returns before building arguments when no trace is bound, so an
untraced solve does not allocate trace records and does not change the search.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from logic.src.tracking.execution_trace import solution_record


def bind_engine_trace(engine: Any, problem: Any) -> None:
    """Copy an opt-in trace onto an engine. Sets nothing when the problem is untraced."""
    trace = getattr(problem, "trace", None)
    parent = getattr(problem, "trace_parent_id", None)
    if trace is None:
        extra = getattr(problem, "extra", None)
        if isinstance(extra, dict):
            trace = extra.get("trace")
            if parent is None:
                parent = extra.get("trace_parent_id")
    if trace is None:
        return
    engine.trace = trace
    engine.trace_parent_id = parent


def begin_solver(problem: Any, name: str) -> Tuple[Any, Optional[str]]:
    """Open a solver stage on ``problem.trace``. ``(None, None)`` when untraced."""
    trace = getattr(problem, "trace", None)
    if trace is None:
        return None, None
    parent = getattr(problem, "trace_parent_id", None)
    if parent is None:
        parent = trace.events[0]["id"] if trace.events else trace.add(name="instance", kind="input")
    stage = trace.add(name=name, kind="stage", parent_id=parent)
    return trace, stage


def attach_master(master: Any, trace: Any, stage_id: Optional[str]) -> None:
    """Bind a trace onto a master. No-op, and sets nothing, when ``trace`` is None."""
    if trace is None:
        return
    master._exec_trace = trace
    master._exec_trace_stage = stage_id
    master._exec_trace_parent = stage_id
    master._exec_trace_node = stage_id


def detach_master(master: Any) -> None:
    """Drop a trace binding. No-op when the master was never traced."""
    if getattr(master, "_exec_trace", None) is None:
        return
    master._exec_trace = None
    master._exec_trace_parent = None
    master._exec_trace_node = None
    master._exec_trace_stage = None


def _parent(owner: Any) -> Optional[str]:
    parent = getattr(owner, "_exec_trace_parent", None)
    if parent is None:
        parent = getattr(owner, "trace_parent_id", None)
    return parent


def _cut_token(cut_id: Any) -> str:
    if isinstance(cut_id, (set, frozenset)):
        return "S:" + ",".join(str(node) for node in sorted(cut_id))
    if isinstance(cut_id, (list, tuple)) and cut_id and all(isinstance(node, int) for node in cut_id):
        if len(cut_id) == 2:
            return f"e:{cut_id[0]}-{cut_id[1]}"
        return "S:" + ",".join(str(node) for node in cut_id)
    return str(cut_id)


def emit_cut(owner: Any, cut_type: str, cut_id: Any, accepted: bool) -> None:
    """Record a cut that was added or rejected. No-op without a bound trace."""
    trace = getattr(owner, "_exec_trace", None)
    if trace is None:
        trace = getattr(owner, "trace", None)
    if trace is None:
        return
    node_id = getattr(owner, "_exec_trace_node", None)
    trace.add(
        name=cut_type,
        kind="operator",
        parent_id=_parent(owner),
        changed_solution=False,
        arguments={
            "cut_id": _cut_token(cut_id),
            "node_id": node_id,
            "decision": "added" if accepted else "rejected",
        },
    )


def emit_cut_on(
    trace: Any,
    parent_id: Optional[str],
    cut_type: str,
    cut_id: Any,
    accepted: bool,
    node_id: Any = None,
) -> None:
    """Record a cut against an explicit trace. No-op when ``trace`` is None."""
    if trace is None:
        return
    trace.add(
        name=cut_type,
        kind="operator",
        parent_id=parent_id,
        changed_solution=False,
        arguments={
            "cut_id": _cut_token(cut_id),
            "node_id": node_id if node_id is not None else parent_id,
            "decision": "added" if accepted else "rejected",
        },
    )


def emit_pricing(master: Any, mode: str, phase: int, n_added: int) -> None:
    """Record one Farkas or ordinary pricing step. No-op without a bound trace."""
    trace = getattr(master, "_exec_trace", None)
    if trace is None:
        return
    trace.add(
        name="pricing",
        kind="operator",
        parent_id=_parent(master),
        changed_solution=False,
        arguments={
            "mode": mode,
            "phase": int(phase),
            "n_added": int(n_added),
            "node_id": getattr(master, "_exec_trace_node", None),
        },
    )


def emit_pricing_on(
    trace: Any,
    parent_id: Optional[str],
    mode: str,
    phase: int,
    n_added: int,
    node_id: Any = None,
) -> None:
    """Record a pricing step on an explicit trace. No-op when ``trace`` is None."""
    if trace is None:
        return
    trace.add(
        name="pricing",
        kind="operator",
        parent_id=parent_id,
        changed_solution=False,
        arguments={
            "mode": mode,
            "phase": int(phase),
            "n_added": int(n_added),
            "node_id": node_id if node_id is not None else parent_id,
        },
    )


def emit_column(master: Any, nodes: Sequence[int], reduced_cost: float, pricing: str, phase: int) -> None:
    """Record a column that was actually injected, with its reduced cost."""
    trace = getattr(master, "_exec_trace", None)
    if trace is None:
        return
    route = [int(node) for node in nodes]
    trace.add(
        name="column",
        kind="operator",
        parent_id=_parent(master),
        changed_solution=True,
        arguments={
            "reduced_cost": float(reduced_cost),
            "pricing": pricing,
            "phase": int(phase),
            "nodes": route,
            "node_id": getattr(master, "_exec_trace_node", None),
        },
        after=solution_record([route]),
    )


def emit_column_on(
    trace: Any,
    parent_id: Optional[str],
    nodes: Sequence[int],
    reduced_cost: float,
    pricing: str,
    phase: int,
    node_id: Any = None,
) -> None:
    """Record an injected column on an explicit trace. No-op when ``trace`` is None."""
    if trace is None:
        return
    route = [int(node) for node in nodes]
    trace.add(
        name="column",
        kind="operator",
        parent_id=parent_id,
        changed_solution=True,
        arguments={
            "reduced_cost": float(reduced_cost),
            "pricing": pricing,
            "phase": int(phase),
            "nodes": route,
            "node_id": node_id if node_id is not None else parent_id,
        },
        after=solution_record([route]),
    )


def emit_phase(master: Any, phase: int, mode: str) -> None:
    """Record the Phase I / Phase II the master is already in. No-op without a trace."""
    trace = getattr(master, "_exec_trace", None)
    if trace is None:
        return
    trace.add(
        name="phase_i" if int(phase) == 1 else "phase_ii",
        kind="stage",
        parent_id=_parent(master),
        changed_solution=False,
        arguments={"phase": int(phase), "pricing": mode},
    )


def emit_node(master: Any, depth: int, node_seq: int) -> Optional[str]:
    """Open a B&B node stage and point later events at it. No-op without a trace."""
    trace = getattr(master, "_exec_trace", None)
    if trace is None:
        return None
    event_id = trace.add(
        name="node",
        kind="stage",
        parent_id=getattr(master, "_exec_trace_stage", None),
        arguments={"depth": int(depth), "node_id": int(node_seq)},
    )
    master._exec_trace_parent = event_id
    master._exec_trace_node = event_id
    return event_id


def emit_node_bound(
    master: Any,
    depth: int,
    bound: float,
    incumbent: Optional[float],
    root: bool = False,
) -> None:
    """Record a node LP bound. ``root=True`` names the event ``root_lp``."""
    trace = getattr(master, "_exec_trace", None)
    if trace is None:
        return
    inc = None if incumbent is None or not math.isfinite(float(incumbent)) else float(incumbent)
    bound_value = float(bound) if bound is not None and math.isfinite(float(bound)) else None
    trace.add(
        name="root_lp" if root else "node_bound",
        kind="operator",
        parent_id=_parent(master),
        changed_solution=False,
        arguments={
            "depth": int(depth),
            "bound": bound_value,
            "incumbent": inc,
            "node_id": getattr(master, "_exec_trace_node", None),
        },
    )


def _branch_side(child: Any) -> Dict[str, Any]:
    constraints = getattr(child, "constraints", None) or []
    payload: Dict[str, Any] = {}
    rule = getattr(child, "branching_rule", None)
    if isinstance(rule, str):
        payload["branching_rule"] = rule
    if not constraints:
        return payload
    last = constraints[-1]
    payload["type"] = type(last).__name__
    for key in ("u", "v", "must_use", "node", "forced", "node_r", "node_s", "together"):
        if hasattr(last, key):
            value = getattr(last, key)
            if value is None or isinstance(value, (bool, int, float, str)):
                payload[key] = value
    return payload


def emit_branch(master: Any, left: Any, right: Any) -> None:
    """Record the branch decision already chosen for this node."""
    trace = getattr(master, "_exec_trace", None)
    if trace is None:
        return
    trace.add(
        name="branch",
        kind="operator",
        parent_id=getattr(master, "_exec_trace_node", None) or _parent(master),
        changed_solution=False,
        arguments={"left": _branch_side(left), "right": _branch_side(right)},
    )


def emit_lagrangian(
    master: Any,
    bound: float,
    lp_obj: float,
    max_rc: float,
    incumbent: float,
) -> None:
    """Record a Lagrangian bound the solver already computed after CG convergence."""
    trace = getattr(master, "_exec_trace", None)
    if trace is None:
        return
    inc = float(incumbent) if math.isfinite(float(incumbent)) else None
    trace.add(
        name="lagrangian_bound",
        kind="operator",
        parent_id=_parent(master),
        changed_solution=False,
        arguments={
            "bound": float(bound),
            "lp_obj": float(lp_obj),
            "max_reduced_cost": float(max_rc),
            "incumbent": inc,
            "node_id": getattr(master, "_exec_trace_node", None),
        },
    )


def selected_route_nodes(master: Any, route_values: Dict[int, float]) -> List[List[int]]:
    """Customer nodes of columns selected above 0.5. Call only while tracing."""
    routes: List[List[int]] = []
    stored = master.routes
    for idx, val in route_values.items():
        index = int(idx)
        if val > 0.5 and 0 <= index < len(stored):
            routes.append([int(node) for node in stored[index].nodes])
    return routes


def emit_incumbent(
    trace: Any,
    parent_id: Optional[str],
    routes: Sequence[Sequence[int]],
    objective: Optional[float],
) -> None:
    """Record the route set that became the incumbent. No-op when ``trace`` is None."""
    if trace is None:
        return
    copied = [[int(node) for node in route] for route in routes]
    finite = None if objective is None or not math.isfinite(float(objective)) else float(objective)
    trace.add(
        name="incumbent",
        kind="incumbent",
        parent_id=parent_id,
        changed_solution=True,
        arguments={"objective": finite},
        after=solution_record(copied, objective=finite, feasible=True),
    )


def close_solver(
    trace: Any,
    parent_id: Optional[str],
    routes: Sequence[Sequence[int]],
    objective: Optional[float],
) -> None:
    """Record the returned incumbent and store ``trace.final``. No-op when untraced."""
    if trace is None:
        return
    copied = [[int(node) for node in route] for route in routes]
    finite = None if objective is None or not math.isfinite(float(objective)) else float(objective)
    emit_incumbent(trace, parent_id, copied, finite)
    trace.add(
        name="routes",
        kind="output",
        parent_id=parent_id,
        changed_solution=True,
        after=solution_record(copied, objective=finite, feasible=True),
    )
    trace.set_final(copied, objective=finite)


def note_unavailable(trace: Any, name: str, parent_id: Optional[str] = None) -> Optional[str]:
    """Record that a solver license or optional dependency is missing, then the caller still raises."""
    if trace is None:
        return None
    return trace.add(
        name=name,
        kind="stage",
        parent_id=parent_id,
        changed_solution=False,
        arguments={"reason": "unavailable"},
        interior="not_instrumented",
    )
