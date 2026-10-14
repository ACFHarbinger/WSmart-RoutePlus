

import numpy as np
import pytest
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.pipeline.simulations.bins.prediction import ScenarioGenerator
from logic.src.policies.routing.exact_and_decomposition_solvers.logic_based_benders_decomposition.policy_lbbd import (
    LBBDPolicy,
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


@_needs_license
def test_lbbd_basic():
    # Toy problem: 3 nodes (0=depot, 1, 2)
    dist_matrix = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.5],
        [1.0, 1.5, 0.0]
    ])

    wastes = {1: 0.1, 2: 0.1}

    config = {
        "num_days": 2,
        "max_iterations": 5,
        "seed": 42,
        "waste_weight": 10.0,
        "cost_weight": 1.0,
        "overflow_penalty": 100.0,
        "mean_increment": 0.5
    }

    gen = ScenarioGenerator(horizon=2, method="stochastic")
    bin_stats = {"means": np.array([5.0, 5.0]), "stds": np.array([1.0, 1.0])}
    tree = gen.generate(current_wastes=np.array([0.1, 0.1]), bin_stats=bin_stats)
    md_ctx = MultiDayContext(day_index=1)

    policy = LBBDPolicy(config=config)

    kwargs = {
        "distance_matrix": dist_matrix,
        "wastes": wastes,
        "capacity": 100.0,
        "day": 1,
        "scenario_tree": tree,
        "multi_day_context": md_ctx
    }

    route, cost, profit, search_ctx, updated_md_ctx = policy.execute(**kwargs)

    assert len(route) >= 2
    assert route[0] == 0
    assert route[-1] == 0
    assert updated_md_ctx is not None
    assert "iterations" in updated_md_ctx.extra

@_needs_license
def test_lbbd_infeasibility_nogood():
    # Make nodes very far or capacity very small to trigger nogood cuts
    dist_matrix = np.array([
        [0.0, 10.0, 10.0],
        [10.0, 0.0, 1.5],
        [10.0, 1.5, 0.0]
    ])

    wastes = {1: 0.9, 2: 0.9}

    config = {
        "num_days": 1,
        "max_iterations": 10,
        "cost_weight": 1.0,
        "subproblem_timeout": 5.0
    }

    gen = ScenarioGenerator(horizon=1, method="stochastic")
    bin_stats = {"means": np.array([5.0, 5.0]), "stds": np.array([1.0, 1.0])}
    tree = gen.generate(current_wastes=np.array([0.9, 0.9]), bin_stats=bin_stats)
    md_ctx = MultiDayContext(day_index=1)

    policy = LBBDPolicy(config=config)

    kwargs = {
        "distance_matrix": dist_matrix,
        "wastes": wastes,
        "capacity": 1.0,
        "day": 1,
        "scenario_tree": tree,
        "multi_day_context": md_ctx
    }

    route, cost, profit, search_ctx, updated_md_ctx = policy.execute(**kwargs)
    assert route[0] == 0
    # It should still find a route, just maybe only visiting one or zero if profit < cost
