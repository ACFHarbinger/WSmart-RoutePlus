

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
        best = min(best, sum(dist_matrix[a][b] for a, b in zip(path, path[1:])))
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
