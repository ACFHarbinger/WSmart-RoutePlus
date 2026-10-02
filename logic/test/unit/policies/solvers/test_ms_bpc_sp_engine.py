

from unittest.mock import MagicMock

import numpy as np
import pytest
from logic.src.policies.helpers.solvers_and_matheuristics import (
    Route,
    VRPPMasterProblem,
)
from logic.src.policies.helpers.solvers_and_matheuristics.branching import (
    FleetSizeBranchingConstraint,
    NodeVisitationBranchingConstraint,
)
from logic.src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine import (
    _apply_branching_to_master,
    _compute_lr_bound_at_node,
    _detect_cycles,
    _is_solution_integer,
    _perform_strong_branching,
    _reset_master_constraints,
    run_ms_bpc_sp,
)
from logic.src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params import (
    MSBPCSPParams,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]



def _has_gurobi_license() -> bool:
    try:
        import gurobipy as _gp



        _m = _gp.Model("_check")
        _m.dispose()
        return True
    except Exception:
        return False


_needs_license = pytest.mark.skipif(
    not _has_gurobi_license(),
    reason="Requires a valid Gurobi license",
)


@pytest.fixture
def ms_bpc_instance():
    dist_matrix = np.array([[0, 10, 10, 20], [10, 0, 5, 15], [10, 5, 0, 10], [20, 15, 10, 0]])
    wastes = {1: 5.0, 2: 5.0, 3: 5.0}  # Total 15
    capacity = 10.0
    R = 10.0
    C = 1.0
    return dist_matrix, wastes, capacity, R, C


@_needs_license
def test_ms_bpc_default(ms_bpc_instance):
    dist, wastes, cap, R, C = ms_bpc_instance
    params = MSBPCSPParams(max_cg_iterations=5, max_bb_nodes=5)
    routes, cost = run_ms_bpc_sp(dist, wastes, cap, R, C, params=params)
    assert len(routes) >= 0
    assert isinstance(cost, float)


def test_ms_bpc_ryan_foster_invalid_cover():
    master = MagicMock(spec=VRPPMasterProblem)
    master.model = MagicMock()
    master.strict_set_partitioning = False
    with pytest.raises(
        RuntimeError, match="Mathematical Exactness Violation: Ryan-Foster branching requires strict Set Partitioning"
    ):
        _apply_branching_to_master(master, [], branching_strategy="ryan_foster")


@_needs_license
def test_ms_bpc_cg_at_root_only(ms_bpc_instance):
    dist, wastes, cap, R, C = ms_bpc_instance
    params = MSBPCSPParams(max_cg_iterations=5, max_bb_nodes=5, cg_at_root_only=True)
    routes, cost = run_ms_bpc_sp(dist, wastes, cap, R, C, params=params)
    assert len(routes) >= 0


def test_detect_cycles():
    assert _detect_cycles([0, 1, 2, 3, 0]) == []
    assert _detect_cycles([0, 1, 2, 1, 3, 0]) == [(1, 2, 1)]
    assert _detect_cycles([0, 1, 2, 3, 2, 0]) == [(2, 3, 2)]


def test_is_solution_integer():
    r1 = Route(nodes=[1, 2], cost=2.0, revenue=12.0, load=5.0, node_coverage={1, 2})
    r2 = Route(nodes=[3], cost=1.0, revenue=6.0, load=5.0, node_coverage={3})
    routes = [r1, r2]
    # Integer solution
    assert _is_solution_integer(routes, {0: 1.0, 1: 1.0}) is True
    # Fractional solution
    assert _is_solution_integer(routes, {0: 0.5, 1: 1.0}) is False
    # Route with cycle (non-elementary)
    r_cycle = Route(nodes=[1, 2, 1], cost=3.0, revenue=18.0, load=5.0, node_coverage={1, 2})
    assert _is_solution_integer([r_cycle], {0: 1.0}) is False
    # Overlapping nodes visited more than once in sum (binary variables cover node twice)
    r3 = Route(nodes=[2, 3], cost=2.0, revenue=10.0, load=5.0, node_coverage={2, 3})
    assert _is_solution_integer([r1, r3], {0: 1.0, 1: 1.0}) is False


def test_reset_master_constraints():
    master = MagicMock(spec=VRPPMasterProblem)
    master.model = MagicMock()
    master.vehicle_limit = 2
    master.n_nodes = 3
    master.mandatory_nodes = {1}
    master.strict_set_partitioning = True

    # Mock constraints
    temp_c = MagicMock()
    v_c = MagicMock()
    c1 = MagicMock()
    c2 = MagicMock()
    c3 = MagicMock()

    master.model.getConstrByName.side_effect = lambda name: {
        "temp_min_vehicles": temp_c,
        "vehicle_limit": v_c,
        "coverage_1": c1,
        "coverage_2": c2,
        "coverage_3": c3,
    }.get(name)

    _reset_master_constraints(master)
    master.model.remove.assert_called_with(temp_c)
    assert v_c.RHS == 2.0
    assert c1.RHS == 1.0
    assert c2.RHS == 1.0


def test_apply_branching_to_master():
    master = MagicMock(spec=VRPPMasterProblem)
    master.model = MagicMock()
    master.vehicle_limit = 2
    master.n_nodes = 3
    master.mandatory_nodes = {1}
    master.strict_set_partitioning = True
    master.routes = [
        Route(nodes=[1, 2], cost=2.0, revenue=10.0, load=5.0, node_coverage={1, 2}),
        Route(nodes=[2, 3], cost=2.0, revenue=10.0, load=5.0, node_coverage={2, 3}),
    ]

    v1 = MagicMock()
    v2 = MagicMock()
    master.lambda_vars = [v1, v2]

    # Mock getConstrByName
    master.model.getConstrByName.return_value = MagicMock()

    # Apply empty branching constraints
    _apply_branching_to_master(master, [], branching_strategy="divergence")
    assert v1.UB == 1.0
    assert v2.UB == 1.0

    # Apply some branching constraints
    bc1 = FleetSizeBranchingConstraint(limit=2, is_upper=True)
    bc2 = NodeVisitationBranchingConstraint(node=2, forced=True)
    _apply_branching_to_master(master, [bc1, bc2], branching_strategy="divergence")
    master.model.update.assert_called()


def test_perform_strong_branching():
    master = MagicMock(spec=VRPPMasterProblem)
    master.model = MagicMock()
    master.model.ObjVal = 100.0
    master.model.Status = 2  # GRB.OPTIMAL

    v1 = MagicMock()
    v1.UB = 1.0
    master.lambda_vars = [v1]
    master.routes = [Route(nodes=[1, 2], cost=2.0, revenue=10.0, load=5.0, node_coverage={1, 2})]

    master.save_basis.return_value = ("basis",)

    candidates = [
        (1, [(1, 2)], [(2, 3)], 0.5),
        (2, [(0, 1)], [(1, 0)], 0.8),
    ]

    res = _perform_strong_branching(master, candidates, strong_branching_size=2)
    assert res is not None
    assert res[0] in (1, 2)
    master.restore_basis.assert_called()


@_needs_license
def test_compute_lr_bound_at_node():
    dist_matrix = np.array([[0, 10, 15], [10, 0, 5], [15, 5, 0]])
    wastes = {1: 5.0, 2: 5.0}
    params = MSBPCSPParams()
    ub, lam, visited = _compute_lr_bound_at_node(
        dist_matrix=dist_matrix,
        wastes=wastes,
        capacity=10.0,
        R=10.0,
        C=1.0,
        mandatory=set(),
        forced_out=set(),
        params=params,
        time_budget=2.0,
        env=None,
        recorder=None,
    )
    assert isinstance(ub, float)
    assert isinstance(lam, float)
    assert isinstance(visited, set)


def test_ms_cg_loop_is_local_and_uses_local_pricing():
    """M-kimi-01 revision (Codex review): the MS CG loop must stay local.

    The imported shared loop resolves its pricing/cycle helpers differently
    (e.g. cycle tuples ([5,6,5] locally vs arc-flavoured entries through the
    imported loop) while consumers read them as node ids. Guard the boundary:
    the engine's loop is the local definition and calls the local diverged
    pricing twins.
    """
    import inspect

    from logic.src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition import (
        ms_bpc_sp_engine,
    )

    src = inspect.getsource(ms_bpc_sp_engine._column_generation_loop)
    assert "_solve_pricing_step(" in src, "MS loop must call the local diverged pricing step"
    assert "_solve_farkas_pricing_step(" in src, "MS loop must call the local diverged Farkas step"
    assert ms_bpc_sp_engine.MSBPCSPPruningException is ms_bpc_sp_engine.BPCPruningException


def _partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for p in _partitions(rest):
        yield [[first]] + p
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1 :]


def _route_cost(dist_matrix, nodes):
    import itertools

    best = float("inf")
    for perm in itertools.permutations(nodes):
        path = (0,) + perm + (0,)
        best = min(best, sum(dist_matrix[a][b] for a, b in zip(path, path[1:], strict=False)))
    return best


def _brute_force_optimum(dist_matrix, wastes, capacity, R, C, vehicle_limit):
    n = len(wastes)
    ids = list(range(1, n + 1))
    best = -float("inf")
    for r in range(0, n + 1):
        for subset in __import__("itertools").combinations(ids, r):
            load = sum(wastes[i] for i in subset)
            for part in _partitions(list(subset)):
                if vehicle_limit is not None and len(part) > vehicle_limit:
                    continue
                if any(sum(wastes[i] for i in route) > capacity + 1e-9 for route in part):
                    continue
                km = sum(_route_cost(dist_matrix, route) for route in part)
                best = max(best, R * load - C * km)
    return best


@_needs_license
def test_ms_bpc_sp_enforces_vehicle_limit_in_phase3_and_phase5():
    """#90: vehicle_limit must survive Phase 3 (regret greedy) and Phase 5 (SP).

    Four bins, each with waste 6 and capacity 10 -> every bin needs its own
    route. The unlimited fleet serves all four; a 2-vehicle fleet can serve at
    most two. Before the fix, Phase 3 opened extra routes and Phase 5's SP
    re-optimization had no fleet constraint, so the engine returned the
    unlimited-fleet profit on limited-fleet instances.
    """
    dist_matrix = np.array(
        [
            [0.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 2.0, 2.0, 2.0],
            [1.0, 2.0, 0.0, 2.0, 2.0],
            [1.0, 2.0, 2.0, 0.0, 2.0],
            [1.0, 2.0, 2.0, 2.0, 0.0],
        ]
    )
    wastes = {1: 6.0, 2: 6.0, 3: 6.0, 4: 6.0}
    capacity, R, C = 10.0, 10.0, 1.0
    params = MSBPCSPParams(
        optimality_gap=1e-9,
        early_termination_gap=1e-9,
        exact_mode=True,
        max_bb_nodes=1000,
        time_limit=60,
        enable_strong_branching_heuristic=False,
    )

    routes_limited, profit_limited = run_ms_bpc_sp(
        dist_matrix, wastes, capacity, R, C, params=params, vehicle_limit=2
    )
    assert len(routes_limited) <= 2, f"fleet limit violated: {len(routes_limited)} routes"
    expected_limited = _brute_force_optimum(dist_matrix.tolist(), wastes, capacity, R, C, 2)
    assert abs(profit_limited - expected_limited) <= 1e-6 * max(1.0, abs(expected_limited))

    routes_unlimited, profit_unlimited = run_ms_bpc_sp(
        dist_matrix, wastes, capacity, R, C, params=params, vehicle_limit=None
    )
    expected_unlimited = _brute_force_optimum(dist_matrix.tolist(), wastes, capacity, R, C, None)
    assert len(routes_unlimited) == 4
    assert abs(profit_unlimited - expected_unlimited) <= 1e-6 * max(1.0, abs(expected_unlimited))


@_needs_license
def test_ms_bpc_sp_empty_fleet_result_relaxes_to_unconstrained_when_it_fits():
    """#90: an empty fleet-constrained result must not stand when the
    unconstrained relaxation's optimum fits the fleet limit.

    Gate instance s5_n5_m[]_v2: the only profitable structure is the pair
    route [1, 5]; standalone gains are all negative, the capacity-budgeted
    pre-selection drops node 5, and the fleet-constrained root CG does not
    converge within the limit. Before the retry passes the engine returned
    0.0 while the brute-force optimum is 6.1556 (one route, within the
    2-vehicle limit).
    """
    rng = np.random.default_rng(5)
    coords = rng.uniform(0, 20, size=(6, 2))
    dist_matrix = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    wastes = {i + 1: float(rng.uniform(3, 9)) for i in range(5)}
    capacity, R, C = 12.0, 3.0, 1.0
    params = MSBPCSPParams(
        optimality_gap=1e-9,
        early_termination_gap=1e-9,
        exact_mode=True,
        max_bb_nodes=100000,
        time_limit=120,
        enable_strong_branching_heuristic=False,
    )

    routes, profit = run_ms_bpc_sp(
        dist_matrix, wastes, capacity, R, C, params=params, vehicle_limit=2
    )
    expected = _brute_force_optimum(dist_matrix.tolist(), wastes, capacity, R, C, 2)
    assert len(routes) <= 2, f"fleet limit violated: {len(routes)} routes"
    assert abs(profit - expected) <= 1e-6 * max(1.0, abs(expected)), (
        f"expected brute-force optimum {expected}, got {profit}"
    )


def test_pricing_status_and_duplicate_timeout_safeguard():
    """Verify explicit pricing certification and incomplete-node safeguard."""
    import time
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from logic.src.policies.helpers.solvers_and_matheuristics import PricingStatus
    from logic.src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition import (
        ms_bpc_sp_engine as engine,
    )

    master = MagicMock()
    master.n_nodes = 6
    master.active_sri_cuts = {}
    master.model.Status = engine.GRB.OPTIMAL
    master.phase = 2
    master.solve_lp_relaxation.return_value = (10.0, {})
    master.add_route.return_value = False

    pricing = MagicMock()
    pricing.pricing_status = PricingStatus.TIMED_OUT
    pricing._timed_out = True
    pricing.last_max_rc = 5.0
    pricing.solve.return_value = [SimpleNamespace(reduced_cost=5.0, profit=5.0)]

    cut_engine = MagicMock()
    cut_engine.separate_and_add_cuts.return_value = 0

    result = engine._column_generation_loop(
        master, pricing, cut_engine, [], 1, 0, 60, time.perf_counter(), exact_mode=True
    )
    assert result[3], "Incomplete pricing with duplicates must be flagged as timed_out/incomplete node"
    assert pricing.solve.call_args.kwargs.get("exact_mode") is True, "exact_mode must be passed to pricing pricer"


@_needs_license
def test_fleet_dual_integration_in_rcspp():
    """Verify fleet-limit dual enters pricing solver and penalizes routes."""
    from logic.src.policies.helpers.solvers_and_matheuristics import RCSPPSolver, Route, VRPPMasterProblem

    dist_matrix = np.array([
        [0.0, 5.0, 5.0],
        [5.0, 0.0, 2.0],
        [5.0, 2.0, 0.0],
    ])
    wastes = {1: 5.0, 2: 5.0}
    master = VRPPMasterProblem(
        n_nodes=2,
        mandatory_nodes=set(),
        cost_matrix=dist_matrix,
        wastes=wastes,
        capacity=10.0,
        revenue_per_kg=10.0,
        cost_per_km=1.0,
        vehicle_limit=1,
    )
    r1 = Route(nodes=[1], cost=10.0, revenue=50.0, load=5.0, node_coverage={1})
    master.add_route(r1)
    master.build_model()
    master.solve_lp_relaxation()
    duals = master.get_reduced_cost_coefficients()
    assert "vehicle_limit" in duals
    assert "fleet_dual" in duals

    pricing = RCSPPSolver(
        n_nodes=2,
        cost_matrix=dist_matrix,
        wastes=wastes,
        capacity=10.0,
        revenue_per_kg=10.0,
        cost_per_km=1.0,
    )
    pricing.solve(dual_values={"node_duals": {1: 10.0, 2: 10.0}, "vehicle_limit": 25.0}, exact_mode=True)
    assert abs(pricing.vehicle_dual - 25.0) < 1e-6


def test_master_solve_ip_time_limit_and_incumbent_recovery():
    """Verify solve_ip respects time_limit parameter and extracts incumbent on TIME_LIMIT."""
    from unittest.mock import MagicMock

    from gurobipy import GRB
    from logic.src.policies.helpers.solvers_and_matheuristics import Route, VRPPMasterProblem

    r1 = Route(nodes=[1], cost=10.0, revenue=50.0, load=5.0, node_coverage={1})
    master = VRPPMasterProblem.__new__(VRPPMasterProblem)
    master.routes = [r1]
    mock_var = MagicMock()
    mock_var.X = 1.0
    master.lambda_vars = [mock_var]
    master.model = MagicMock()
    master.model.NumVars = 1
    master.model.Params = MagicMock()
    master.model.Params.TimeLimit = 123.0
    master.model.Status = GRB.OPTIMAL
    master.model.SolCount = 1
    master.model.ObjVal = 40.0

    # Case 1: Normal solve with custom time_limit sets and restores old time_limit
    obj, routes = master.solve_ip(time_limit=2.5)
    assert master.model.Params.TimeLimit == 123.0
    assert obj == 40.0
    assert len(routes) == 1

    # Case 2: TIME_LIMIT status with SolCount > 0 extracts incumbent
    master.model.Status = GRB.TIME_LIMIT
    master.model.Params.TimeLimit = 50.0
    obj, routes = master.solve_ip(time_limit=1.0)
    assert obj == 40.0
    assert len(routes) == 1
    assert master.model.Params.TimeLimit == 50.0

    # Case 3: TIME_LIMIT status with SolCount == 0 raises RuntimeError
    master.model.SolCount = 0
    with pytest.raises(RuntimeError):
        master.solve_ip(time_limit=1.0)


@_needs_license
def test_master_solve_ip_real_gurobi():
    """Verify solve_ip on a live Gurobi model with license."""
    from logic.src.policies.helpers.solvers_and_matheuristics import Route, VRPPMasterProblem

    dist_matrix = np.array([
        [0.0, 5.0, 5.0],
        [5.0, 0.0, 2.0],
        [5.0, 2.0, 0.0],
    ])
    wastes = {1: 5.0, 2: 5.0}
    master = VRPPMasterProblem(
        n_nodes=2,
        mandatory_nodes=set(),
        cost_matrix=dist_matrix,
        wastes=wastes,
        capacity=10.0,
        revenue_per_kg=10.0,
        cost_per_km=1.0,
        vehicle_limit=1,
    )
    r1 = Route(nodes=[1], cost=10.0, revenue=50.0, load=5.0, node_coverage={1})
    master.add_route(r1)
    master.build_model()
    master.model.Params.TimeLimit = 123.0
    obj, routes = master.solve_ip(time_limit=2.5)
    assert master.model.Params.TimeLimit == 123.0
    assert obj == 40.0
    assert len(routes) == 1


def test_cg_at_root_only_descendant_leaves_node_incomplete():
    """Skipping pricing at descendant nodes under cg_at_root_only must flag node timed_out."""
    import time
    from unittest.mock import MagicMock

    from logic.src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition import (
        ms_bpc_sp_engine as e,
    )

    m = MagicMock()
    m.n_nodes = 2
    m.active_sri_cuts = {}
    m.model.Status = e.GRB.OPTIMAL
    m.phase = 2
    m.solve_lp_relaxation.return_value = (10.0, {})
    p = MagicMock()
    p._timed_out = False
    c = MagicMock()
    c.separate_and_add_cuts.return_value = 0
    r = e._column_generation_loop(m, p, c, [], 1, 0, 60, time.perf_counter(), node_depth=1, cg_at_root_only=True)
    assert r[3], "Skipped descendant pricing cannot certify its restricted LP bound"
    assert p.solve.call_count == 0


def test_restricted_pricing_certification():
    """Restricted-neighbor pricing must not certify EXHAUSTIVE when exact_mode=False."""
    from logic.src.policies.helpers.solvers_and_matheuristics import PricingStatus, RCSPPSolver

    p = RCSPPSolver(
        n_nodes=6,
        cost_matrix=np.ones((7, 7)) - np.eye(7),
        wastes={i: 1 for i in range(1, 7)},
        capacity=2,
        revenue_per_kg=0,
        cost_per_km=1,
    )
    p.solve({}, exact_mode=False)
    assert p.pricing_status != PricingStatus.EXHAUSTIVE, "Restricted-neighbor pricing falsely certified exhaustive"
    p.solve({}, exact_mode=True)
    assert p.pricing_status == PricingStatus.EXHAUSTIVE


def test_farkas_unexhausted_preserves_incomplete_node():
    """Unexhausted Farkas pricing returning no columns must flag node timed_out/incomplete."""
    import time
    from unittest.mock import MagicMock

    from logic.src.policies.helpers.solvers_and_matheuristics import PricingStatus
    from logic.src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition import (
        ms_bpc_sp_engine as engine,
    )

    master = MagicMock()
    master.n_nodes = 2
    master.active_sri_cuts = {}
    master.model.Status = engine.GRB.INFEASIBLE
    master.phase = 1
    master.farkas_duals = {1: 1.0}
    master.solve_lp_relaxation.return_value = (0.0, {})
    pricing = MagicMock()
    pricing._timed_out = False
    pricing.pricing_status = PricingStatus.PARTIAL
    pricing.last_max_rc = 0.0
    pricing.solve.return_value = []
    result = engine._column_generation_loop(master, pricing, MagicMock(), [], 1, 0, 60, time.perf_counter())
    assert result[3], "Unexhausted Farkas pricing must leave the node incomplete"

