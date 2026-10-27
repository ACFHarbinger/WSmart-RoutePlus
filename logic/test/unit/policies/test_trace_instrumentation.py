"""Unit tests for execution trace instrumentation (Issue 174).

Covers:
- ALNS search routing (destroy, repair, acceptance, incumbent) with exact seed parity.
- Packing policies: last_minute, regular, service_level, lookahead.
- Route improvement: fast_tsp, classical_local_search.
- Acceptance criteria: oi, bmc.
"""

import numpy as np
import pytest
from logic.src.configs.policies.acceptance_criteria import AcceptanceConfig, OnlyImprovingConfig
from logic.src.configs.policies.routing.meta_heuristics.alns import ALNSConfig
from logic.src.interfaces.context import PackingContext
from logic.src.policies.acceptance_criteria.boltzmann_metropolis_criterion import BoltzmannAcceptance
from logic.src.policies.acceptance_criteria.only_improving import OnlyImproving
from logic.src.policies.packing.last_minute import LastMinuteSelection
from logic.src.policies.packing.lookahead import LookaheadSelection
from logic.src.policies.packing.regular import RegularSelection
from logic.src.policies.packing.service_level import ServiceLevelSelection
from logic.src.policies.route_improvement.fast_tsp import FastTSPRouteImprover
from logic.src.policies.route_improvement.local_search import ClassicalLocalSearchRouteImprover
from logic.src.policies.routing.base.sub_problem import build_subproblem
from logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search.dispatcher import (
    run_alns,
)
from logic.src.policies.trace_runner import make_instance, routes_equal, solve_alns
from logic.src.tracking.execution_trace import ExecutionTrace, validate_trace

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_traced_alns_exact_seed_parity_and_events():
    """An instrumented ALNS run emits real search events and matches untraced solve exactly."""
    graph_size = 5
    seed = 42
    acceptance = "oi"
    iterations = 10
    dist, wastes, coords = make_instance(graph_size, seed)

    # Untraced solve
    plain = solve_alns(dist, wastes, seed, acceptance, iterations)

    # Traced solve
    trace = ExecutionTrace(
        problem="ptp",
        policies={"packing": None, "routing": "alns", "route_improvement": None, "acceptance": acceptance},
        seed=seed,
        instance={
            "n_nodes": graph_size,
            "depot": [float(coords[0, 0]), float(coords[0, 1])],
            "capacity": 10.0,
        },
    )
    input_id = trace.add("instance", "input", arguments={"n_nodes": graph_size})

    problem = build_subproblem(
        dist, wastes, capacity=10.0, revenue=1.0, cost_unit=1.0, trace=trace, trace_parent_id=input_id
    )
    config = ALNSConfig(
        seed=seed,
        max_iterations=iterations,
        acceptance_criterion=AcceptanceConfig(method="oi", params=OnlyImprovingConfig()),
        time_limit=30.0,
    )
    traced_res = run_alns(problem, config)

    # 1. Exact parity assertion
    assert routes_equal(plain.routes, traced_res.routes)
    assert plain.profit == pytest.approx(traced_res.profit)

    # 2. Schema validation
    document = trace.to_dict()
    validate_trace(document)
    assert document["final"]["routes"] == [list(r) for r in traced_res.routes]

    # 3. Check emitted event kinds and names
    event_names = [e["name"] for e in document["events"]]

    assert "alns" in event_names
    assert "initial_solution" in event_names
    assert "incumbent" in event_names
    assert "oi" in event_names

    # Check destroy and repair operators exist
    destroy_operators = {"RandomRemoval", "WorstRemoval", "ShawRemoval"}
    repair_operators = {"GreedyProfitClean", "GreedyProfitNoisy", "Regret2ProfitClean", "Regret2ProfitNoisy"}

    found_destroys = [name for name in event_names if name in destroy_operators]
    found_repairs = [name for name in event_names if any(name.startswith(r[:6]) for r in repair_operators)]

    assert len(found_destroys) > 0, f"Expected destroy operators, found: {event_names}"
    assert len(found_repairs) > 0, f"Expected repair operators, found: {event_names}"

    # Verify every event has a valid parent_id
    seen_ids = set()
    for e in document["events"]:
        if e["parent_id"] is not None:
            assert e["parent_id"] in seen_ids
        seen_ids.add(e["id"])


def test_traced_packing_policies():
    """Packing policies (last_minute, regular, service_level, lookahead) emit operator events."""
    fills = np.array([30.0, 85.0, 92.0, 10.0], dtype=np.float64)
    rates = np.array([5.0, 10.0, 15.0, 2.0], dtype=np.float64)
    stds = np.array([1.0, 2.0, 3.0, 0.5], dtype=np.float64)

    trace = ExecutionTrace("cptp", {}, seed=1, instance={"n_nodes": 5})
    input_id = trace.add("instance", "input")

    # 1. last_minute
    lm_ctx = PackingContext(
        bin_ids=np.array([1, 2, 3, 4], dtype=np.int32),
        current_fill=fills,
        threshold=80.0,
        trace=trace,
        trace_parent_id=input_id,
    )
    lm_sel, _ = LastMinuteSelection().pack(lm_ctx)
    assert lm_sel == [2, 3]  # 1-based IDs for indices 1 and 2

    # 2. regular
    reg_ctx = PackingContext(
        bin_ids=np.array([1, 2, 3, 4], dtype=np.int32),
        current_fill=fills,
        current_day=1,
        threshold=1.0,  # every day
        trace=trace,
        trace_parent_id=input_id,
    )
    reg_sel, _ = RegularSelection().pack(reg_ctx)
    assert reg_sel == [1, 2, 3, 4]

    # 3. service_level
    sl_ctx = PackingContext(
        bin_ids=np.array([1, 2, 3, 4], dtype=np.int32),
        current_fill=fills,
        accumulation_rates=rates,
        std_deviations=stds,
        threshold=1.0,  # z=1.0
        horizon_days=1,
        trace=trace,
        trace_parent_id=input_id,
    )
    sl_sel, _ = ServiceLevelSelection().pack(sl_ctx)
    assert sl_sel == [3]

    # 4. lookahead
    la_ctx = PackingContext(
        bin_ids=np.array([1, 2, 3, 4], dtype=np.int32),
        current_fill=fills,
        accumulation_rates=rates,
        current_collection_day=0,
        trace=trace,
        trace_parent_id=input_id,
    )
    la_sel, _ = LookaheadSelection().pack(la_ctx)
    assert len(la_sel) > 0

    # Validate trace events
    events = trace.events
    assert len(events) == 5  # input + 4 packing operators
    packing_events = events[1:]
    for ev in packing_events:
        assert ev["kind"] == "operator"
        assert ev["parent_id"] == input_id
        assert ev["changed_solution"] is True
        assert ev["before"] is not None
        assert ev["after"] is not None
        assert "selected_bins" in ev["arguments"]


def test_traced_route_improvement():
    """Route improvement policies (fast_tsp, classical_local_search) record changes."""
    dist_matrix = np.array([
        [0.0, 1.0, 5.0, 1.0],
        [1.0, 0.0, 1.0, 5.0],
        [5.0, 1.0, 0.0, 1.0],
        [1.0, 5.0, 1.0, 0.0],
    ])
    # Sub-optimal tour: 0 -> 1 -> 3 -> 2 -> 0 (cost: 1 + 5 + 1 + 5 = 12)
    # Optimal tour: 0 -> 1 -> 2 -> 3 -> 0 (cost: 1 + 1 + 1 + 1 = 4)
    suboptimal_tour = [0, 1, 3, 2, 0]

    trace = ExecutionTrace("ptp", {}, seed=1, instance={"n_nodes": 4})
    input_id = trace.add("instance", "input")

    # Fast TSP
    improver = FastTSPRouteImprover(config={})
    refined, metrics = improver.process(
        suboptimal_tour,
        distance_matrix=dist_matrix,
        trace=trace,
        parent_id=input_id,
    )
    assert refined == [0, 1, 2, 3, 0] or refined == [0, 3, 2, 1, 0]

    # Classical Local Search (2opt)
    cls = ClassicalLocalSearchRouteImprover(config={"ls_operator": "2opt", "iterations": 50})
    cls_refined, _ = cls.process(
        suboptimal_tour,
        distance_matrix=dist_matrix,
        trace=trace,
        parent_id=input_id,
    )
    assert cls_refined == [0, 1, 2, 3, 0] or cls_refined == [0, 3, 2, 1, 0]

    # Check trace
    events = [e for e in trace.events if e["kind"] == "operator"]
    assert len(events) == 2
    assert events[0]["name"] == "fast_tsp"
    assert events[0]["changed_solution"] is True
    assert events[1]["name"] == "classical_local_search"
    assert events[1]["changed_solution"] is True


def test_traced_acceptance_criteria():
    """Acceptance criteria (oi, bmc) record acceptance decisions."""
    trace = ExecutionTrace("ptp", {}, seed=1, instance={"n_nodes": 3})
    input_id = trace.add("instance", "input")

    # OnlyImproving
    oi = OnlyImproving()
    oi.setup(10.0)
    acc1, _ = oi.accept(10.0, 12.0, trace=trace, parent_id=input_id)
    assert acc1 is True
    acc2, _ = oi.accept(12.0, 11.0, trace=trace, parent_id=input_id)
    assert acc2 is False

    # Boltzmann
    bmc = BoltzmannAcceptance(initial_temp=100.0, alpha=0.95, seed=42)
    bmc.setup(10.0)
    b_acc1, _ = bmc.accept(10.0, 12.0, trace=trace, parent_id=input_id)
    assert b_acc1 is True
    b_acc2, _ = bmc.accept(12.0, 8.0, trace=trace, parent_id=input_id)

    # Check events
    acc_events = [e for e in trace.events if e["name"] in ("oi", "bmc")]
    assert len(acc_events) == 4
    assert acc_events[0]["arguments"]["accepted"] is True
    assert acc_events[0]["changed_solution"] is True
    assert acc_events[1]["arguments"]["accepted"] is False
    assert acc_events[1]["changed_solution"] is False
    assert "temperature" in acc_events[2]["arguments"]
