"""ACO-HH regressions from the 2026-09-26 logic review (B-kimi-44..49)."""

import random

import numpy as np
import pytest
from logic.src.configs.policies import HyperHeuristicACOConfig
from logic.src.policies.helpers.operators.solution_initialization.greedy_si import build_greedy_routes
from logic.src.policies.routing.base.sub_problem import build_subproblem
from logic.src.policies.routing.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco import (
    HyperHeuristicACO,
)
from logic.src.policies.routing.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators import (
    OPERATOR_NAMES,
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


def _problem(inst):
    dist, wastes, cap, _init, mandatory = inst
    return build_subproblem(dist, wastes, cap, 1.0, 1.0, mandatory_nodes=mandatory)


def _solve(params, inst):
    dist, wastes, cap, init, mandatory = inst
    return HyperHeuristicACO(_problem(inst), params, init).solve()


def test_same_seed_same_plan():
    """B-kimi-44/45: perturb reseeded from OS entropy and visibility used wall-clock time."""
    inst = _instance(0, 9, 60.0, 5, 40)
    params = HyperHeuristicACOConfig(n_ants=6, max_iterations=8, time_limit=30, seed=42, profit_aware_operators=True)
    runs = {(tuple(map(tuple, r[0])), round(r[1], 9)) for r in (_solve(params, inst) for _ in range(4))}
    assert len(runs) == 1


def test_returned_plan_is_capacity_feasible_under_oscillation():
    """B-kimi-46: strategic oscillation could return an over-capacity plan as the best."""
    inst = _instance(7, 13, 50.0, 15, 45, mandatory_all=True)
    params = HyperHeuristicACOConfig(
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
    params = HyperHeuristicACOConfig(
        n_ants=2, max_iterations=1, seed=1, operators=["swap", "relocate"], sequence_length=3
    )
    solver = HyperHeuristicACO(_problem(inst), params, init)
    assert solver.operator_names == ["swap", "relocate"]
    sequence, _end = solver._select_sequence(0)
    assert len(sequence) == 3 and set(sequence) <= {"swap", "relocate"}

    default = HyperHeuristicACO(_problem(inst), HyperHeuristicACOConfig(seed=1), init)
    assert default.operator_names == OPERATOR_NAMES and default.sequence_length == len(OPERATOR_NAMES)

    with pytest.raises(ValueError):
        HyperHeuristicACO(
            _problem(inst),
            HyperHeuristicACOConfig(operators=["no_such_operator"]),
            init,
        )


def test_first_hop_deposits_on_real_rows_only():
    """B-kimi-49: the first hop of an improving journey was deposited on the virtual row."""
    inst = _instance(0, 9, 60.0, 5, 40)
    params = HyperHeuristicACOConfig(n_ants=6, max_iterations=8, time_limit=30, seed=42, profit_aware_operators=True)
    _dist, _wastes, _cap, init, _mandatory = inst
    solver = HyperHeuristicACO(_problem(inst), params, init)
    solver.solve()
    virtual = solver.tau[solver.n_operators]
    assert np.allclose(virtual, virtual[0]), "only evaporation may touch the virtual row"
    assert virtual[0] <= params.tau_0


def test_elitism_sync_count_uses_ceiling():
    """B-kimi-51: elitism syncs ceil(n_ants * ratio) ants, matching the flowchart."""
    from logic.src.policies.routing.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco_hooks import (
        _elitism_sync_count,
    )

    assert _elitism_sync_count(10, 0.35) == 4  # int() floor gave 3
    assert _elitism_sync_count(10, 0.5) == 5
    assert _elitism_sync_count(10, 1.0) == 10
    assert _elitism_sync_count(10, 0.01) == 1  # floor of 0.1 would be 0 without the max


def test_operator_wrappers_swallow_data_errors_only():
    """B-kimi-50: operator wrappers return False on data errors but let programming errors propagate."""
    import logic.src.policies.routing.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators as ops

    dist, wastes, cap, init, mandatory = _instance(0, 9, 60.0, 5, 40)
    ctx = ops.HyperOperatorContext(
        routes=init, dist_matrix=dist, waste=wastes, capacity=cap, R=1.0, C=1.0, mandatory_nodes=mandatory
    )

    real_random_removal = ops.random_removal
    try:

        def raise_value_error(routes, n, rng=None):
            raise ValueError("simulated node/matrix mismatch")

        ops.random_removal = raise_value_error
        assert ops.apply_random_removal(ctx) is False

        def raise_type_error(routes, n, rng=None):
            raise TypeError("simulated programming error")

        ops.random_removal = raise_type_error
        with pytest.raises(TypeError):
            ops.apply_random_removal(ctx)
    finally:
        ops.random_removal = real_random_removal


@pytest.mark.parametrize("operators", [None, ["kick"]])
def test_mandatory_nodes_are_never_dropped(operators):
    """The profit-aware kick left unprofitable mandatory nodes unvisited, and the cost-only objective kept the plan."""
    rng = np.random.default_rng(3)
    coords = rng.uniform(0, 20, size=(31, 2))
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    wastes = {i: float(rng.uniform(0, 30)) for i in range(1, 31)}
    mandatory = list(range(1, 31, 2))
    cap = sum(wastes.values()) / 4
    init = build_greedy_routes(
        dist_matrix=dist, wastes=wastes, capacity=cap, R=0.58, C=1.0, mandatory_nodes=mandatory, rng=random.Random(3)
    )
    params = HyperHeuristicACOConfig(
        n_ants=6, max_iterations=10, time_limit=20, seed=3, profit_aware_operators=True, operators=operators
    )
    problem = build_subproblem(dist, wastes, cap, 0.58, 1.0, mandatory_nodes=mandatory)
    routes = HyperHeuristicACO(problem, params, init).solve().routes
    assert set(mandatory) <= {n for r in routes for n in r}


def test_profit_kick_and_perturb_keep_mandatory_nodes():
    from logic.src.policies.helpers.operators.perturbation_shaking import kick_profit, perturb_profit
    from logic.src.policies.routing.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators import (
        HyperOperatorContext,
    )

    rng = np.random.default_rng(5)
    dist = rng.uniform(5, 15, size=(11, 11))
    np.fill_diagonal(dist, 0.0)
    wastes = {i: 0.01 for i in range(1, 11)}  # every node is unprofitable
    for op in (kick_profit, perturb_profit):
        ctx = HyperOperatorContext(
            [[1, 2, 3], [4, 5]], dist, wastes, 100.0, 1.0, 1.0, mandatory_nodes=[1, 2, 3, 4, 5], rng=random.Random(5)
        )
        for _ in range(20):
            if op is kick_profit:
                op(ctx, destroy_ratio=0.6, rng=ctx.rng)
            else:
                op(ctx, k=5, prob_unvisited=1.0, rng=ctx.rng)
        assert {1, 2, 3, 4, 5} <= {n for r in ctx.routes for n in r}
