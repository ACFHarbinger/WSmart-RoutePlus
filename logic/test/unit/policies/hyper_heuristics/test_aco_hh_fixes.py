"""ACO-HH regressions from the 2026-09-26 logic review (B-kimi-44..49)."""

import random

import numpy as np
import pytest
from logic.src.policies.helpers.operators.solution_initialization.greedy_si import build_greedy_routes
from logic.src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco import (
    HyperHeuristicACO,
)
from logic.src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators import (
    OPERATOR_NAMES,
)
from logic.src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.params import (
    HyperACOParams,
)

pytestmark = [pytest.mark.unit]


def _instance(seed, n, cap, lo, hi, mandatory_all=False):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 10, size=(n, 2))
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    wastes = {i: float(rng.uniform(lo, hi)) for i in range(1, n)}
    mandatory = list(range(1, n)) if mandatory_all else [1, 2, 3]
    init = build_greedy_routes(
        dist_matrix=dist, wastes=wastes, capacity=cap, R=1.0, C=1.0, mandatory_nodes=mandatory, rng=random.Random(42)
    )
    return dist, wastes, cap, init, mandatory


def _solve(params, inst):
    dist, wastes, cap, init, mandatory = inst
    return HyperHeuristicACO(dist, wastes, cap, 1.0, 1.0, params, init, mandatory).solve()


def test_same_seed_same_plan():
    """B-kimi-44/45: perturb reseeded from OS entropy and visibility used wall-clock time."""
    inst = _instance(0, 9, 60.0, 5, 40)
    params = HyperACOParams(n_ants=6, max_iterations=8, time_limit=30, seed=42, profit_aware_operators=True)
    runs = {(tuple(map(tuple, r[0])), round(r[1], 9)) for r in (_solve(params, inst) for _ in range(4))}
    assert len(runs) == 1


def test_returned_plan_is_capacity_feasible_under_oscillation():
    """B-kimi-46: strategic oscillation could return an over-capacity plan as the best."""
    inst = _instance(7, 13, 50.0, 15, 45, mandatory_all=True)
    params = HyperACOParams(
        n_ants=6, max_iterations=60, time_limit=25, seed=42, profit_aware_operators=True, stagnation_limit=5
    )
    routes, _profit, _cost = _solve(params, inst)
    _dist, wastes, cap, _init, _m = inst
    assert all(r for r in routes), "empty routes must not be returned"
    assert all(sum(wastes[x] for x in r) <= cap + 1e-9 for r in routes)


def test_operator_pool_and_sequence_length_are_honoured():
    """B-kimi-47/48: the yaml operators list and sequence_length were ignored."""
    inst = _instance(0, 9, 60.0, 5, 40)
    dist, wastes, cap, init, mandatory = inst
    params = HyperACOParams(n_ants=2, max_iterations=1, seed=1, operators=["swap", "relocate"], sequence_length=3)
    solver = HyperHeuristicACO(dist, wastes, cap, 1.0, 1.0, params, init, mandatory)
    assert solver.operator_names == ["swap", "relocate"]
    sequence, _end = solver._select_sequence(0)
    assert len(sequence) == 3 and set(sequence) <= {"swap", "relocate"}

    default = HyperHeuristicACO(dist, wastes, cap, 1.0, 1.0, HyperACOParams(seed=1), init, mandatory)
    assert default.operator_names == OPERATOR_NAMES and default.sequence_length == len(OPERATOR_NAMES)

    with pytest.raises(ValueError):
        HyperACOParams(operators=["no_such_operator"])


def test_first_hop_deposits_on_real_rows_only():
    """B-kimi-49: the first hop of an improving journey was deposited on the virtual row."""
    inst = _instance(0, 9, 60.0, 5, 40)
    params = HyperACOParams(n_ants=6, max_iterations=8, time_limit=30, seed=42, profit_aware_operators=True)
    dist, wastes, cap, init, mandatory = inst
    solver = HyperHeuristicACO(dist, wastes, cap, 1.0, 1.0, params, init, mandatory)
    solver.solve()
    virtual = solver.tau[solver.n_operators]
    assert np.allclose(virtual, virtual[0]), "only evaporation may touch the virtual row"
    assert virtual[0] <= params.tau_0
