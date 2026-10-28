"""Execution-trace events for exact solvers (issue 170).

The recorder test always runs: it feeds the shared trace helpers and the
branch-and-cut cut callback and checks the event names. A live Gurobi solve
is skipped when gurobipy or a solver license is unavailable.
"""

from typing import Any, List

import numpy as np
import pytest
from logic.src.configs.policies.routing.exact_and_decomposition_solvers.bpc import BPCConfig
from logic.src.policies.helpers.solvers_and_matheuristics.exact_trace import (
    attach_master,
    close_solver,
    emit_branch,
    emit_column,
    emit_cut,
    emit_lagrangian,
    emit_node,
    emit_node_bound,
    emit_phase,
    emit_pricing,
    note_unavailable,
)
from logic.src.policies.routing.base.sub_problem import build_subproblem
from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_cut.cut_callbacks import (
    BCCutSeparationMixin,
)
from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine import run_bpc
from logic.src.tracking.execution_trace import ExecutionTrace, validate_trace

pytestmark = [pytest.mark.unit, pytest.mark.fast]

_PRICING_OR_CUT = {
    "pricing",
    "column",
    "capacity",
    "sec",
    "sri",
    "lci",
    "edge_clique",
    "multistar",
}


class _Master:
    """Trace binding target. Attributes are set by ``attach_master``."""


class _BranchChild:
    """Branch child with no constraint payload."""

    def __init__(self) -> None:
        self.branching_rule = "edge"
        self.constraints: List[Any] = []


def _trace() -> ExecutionTrace:
    return ExecutionTrace(
        problem="vrpp",
        policies={"routing": "bpc"},
        seed=0,
        instance={"n_nodes": 3, "capacity": 10.0},
    )


def test_exact_trace_recorder_event_names() -> None:
    """Fake master and branch-and-cut callback emit the exact-solver event names.

    This does not run Gurobi. The live solve below is skipped when the exact
    stack cannot run.
    """
    trace = _trace()
    parent = trace.add(name="instance", kind="input", arguments={"n_nodes": 3})
    host = _Master()
    attach_master(host, trace, parent)
    emit_phase(host, 1, "farkas")
    emit_phase(host, 2, "ordinary")
    emit_node(host, 0, 0)
    emit_node_bound(host, 0, 1.5, None, root=True)
    emit_node_bound(host, 1, 2.0, 4.0, root=False)
    emit_pricing(host, "farkas", 1, 1)
    emit_column(host, [1, 2], 0.25, "farkas", 1)
    emit_pricing(host, "ordinary", 2, 0)
    emit_cut(host, "capacity", frozenset({1, 2}), True)
    child = _BranchChild()
    emit_branch(host, child, child)
    emit_lagrangian(host, 3.0, 1.5, 0.25, 4.0)
    close_solver(trace, parent, [[1, 2]], 4.0)

    callback = BCCutSeparationMixin.__new__(BCCutSeparationMixin)
    callback.trace = trace
    callback.trace_parent_id = parent
    callback._record_separated_cut("sec", frozenset({1}), False)
    unavailable = note_unavailable(trace, "gurobi", parent_id=parent)

    # Untraced calls allocate no events.
    silent = ExecutionTrace("vrpp", {"routing": "bpc"}, seed=0, instance={})
    before = len(silent.events)
    emit_column({}, [1], 1.0, "ordinary", 2)
    emit_pricing({}, "ordinary", 2, 0)
    emit_cut({}, "capacity", frozenset({1}), False)
    close_solver(None, None, [[1]], 1.0)
    assert len(silent.events) == before

    document = trace.to_dict()
    validate_trace(document)
    names = [event["name"] for event in document["events"]]
    for expected in (
        "phase_i",
        "phase_ii",
        "pricing",
        "column",
        "capacity",
        "node",
        "root_lp",
        "node_bound",
        "branch",
        "lagrangian_bound",
        "incumbent",
        "routes",
        "sec",
        "gurobi",
        "bc",
    ):
        assert expected in names

    columns = [event for event in document["events"] if event["name"] == "column"]
    assert columns
    for event in columns:
        assert "reduced_cost" in event["arguments"]
        assert "profit" not in event["arguments"]
        assert event["arguments"]["pricing"] in {"farkas", "ordinary"}

    capacity = next(event for event in document["events"] if event["name"] == "capacity")
    assert capacity["arguments"]["decision"] == "added"
    assert capacity["arguments"]["cut_id"] == "S:1,2"
    rejected = next(event for event in document["events"] if event["name"] == "sec")
    assert rejected["arguments"]["decision"] == "rejected"

    phases = {event["name"]: event["arguments"]["pricing"] for event in document["events"] if event["name"] in {"phase_i", "phase_ii"}}
    assert phases["phase_i"] == "farkas"
    assert phases["phase_ii"] == "ordinary"

    unavailable_event = next(event for event in document["events"] if event["id"] == unavailable)
    assert unavailable_event["arguments"]["reason"] == "unavailable"
    assert unavailable_event["interior"] == "not_instrumented"
    assert document["final"]["routes"] == [[1, 2]]


def _require_gurobi() -> None:
    """Skip when gurobipy is missing or a license check cannot optimize."""
    gp = pytest.importorskip("gurobipy")
    try:
        model = gp.Model("exact-trace-license")
        model.Params.OutputFlag = 0
        model.addVar(obj=1.0)
        model.optimize()
        ready = model.Status == gp.GRB.OPTIMAL
    except Exception as exc:
        pytest.skip(f"live exact solve skipped; Gurobi cannot run: {exc}")
    if not ready:
        pytest.skip("live exact solve skipped; Gurobi did not solve a trivial model")


def test_traced_bpc_matches_untraced_tiny_instance() -> None:
    """A tiny BPC writes a pricing or cut event and matches the untraced solve.

    Skipped when Gurobi or the exact stack cannot run. Event names are still
    checked by ``test_exact_trace_recorder_event_names``.
    """
    _require_gurobi()
    dist = np.array([[0.0, 10.0, 10.0], [10.0, 0.0, 5.0], [10.0, 5.0, 0.0]])
    wastes = {1: 5.0, 2: 5.0}
    cfg = BPCConfig(
        max_cg_iterations=8,
        max_cuts_per_iteration=2,
        max_bb_nodes=8,
        time_limit=15.0,
        branching_strategy="edge",
        lr_pre_pruning=False,
    )
    plain = run_bpc(build_subproblem(dist, wastes, 10.0, 10.0, 1.0), cfg)

    trace = _trace()
    parent = trace.add(name="instance", kind="input", arguments={"n_nodes": 3})
    traced = run_bpc(
        build_subproblem(dist, wastes, 10.0, 10.0, 1.0, trace=trace, trace_parent_id=parent),
        cfg,
    )

    assert [list(route) for route in traced.routes] == [list(route) for route in plain.routes]
    assert traced.profit == pytest.approx(plain.profit)
    assert traced.cost == pytest.approx(plain.cost)

    document = trace.to_dict()
    validate_trace(document)
    names = {event["name"] for event in document["events"]}
    assert names & _PRICING_OR_CUT
    assert "pricing" in names or names & {"capacity", "sec", "sri", "lci", "edge_clique", "multistar"}
    for event in document["events"]:
        if event["name"] == "column":
            assert "reduced_cost" in event["arguments"]
            assert "profit" not in event["arguments"]
    assert document["final"]["routes"] == [list(route) for route in traced.routes]
