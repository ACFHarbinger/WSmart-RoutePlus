

import numpy as np
import pytest
from logic.src.pipeline.simulations.bins.prediction import ScenarioGenerator
from logic.src.policies.routing.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.policy_cp_sat import (
    CPSATPolicy,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]



def test_cp_sat_basic():
    # Toy problem: 3 nodes (0=depot, 1, 2)
    dist_matrix = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.5],
        [1.0, 1.5, 0.0]
    ])

    wastes = {1: 0.1, 2: 0.1}

    config = {
        "num_days": 2,
        "time_limit": 30.0,
        "seed": 42,
        "waste_weight": 10.0,
        "cost_weight": 1.0,
        "overflow_penalty": 100.0,
        "mean_increment": 0.5
    }

    gen = ScenarioGenerator(horizon=2, method="stochastic")
    bin_stats = {"means": np.array([5.0, 5.0]), "stds": np.array([1.0, 1.0])}
    tree = gen.generate(current_wastes=np.array([0.1, 0.1]), bin_stats=bin_stats)

    policy = CPSATPolicy(config=config)

    from logic.src.interfaces.context.multi_day_context import MultiDayContext



    md_ctx = MultiDayContext(day_index=1)
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
    assert "expected_profit" in updated_md_ctx.extra

def test_cp_sat_multi_day():
    # Verify that it plans for future days
    dist_matrix = np.array([
        [0.0, 1.0],
        [1.0, 0.0]
    ])

    # Bin 1 is almost full
    wastes = {1: 0.9}

    config = {
        "num_days": 1,
        "cost_weight": 1.0,
        "waste_weight": 10.0
    }

    gen = ScenarioGenerator(horizon=1, method="stochastic")
    bin_stats = {"means": np.array([0.1]), "stds": np.array([0.01])}
    tree = gen.generate(current_wastes=np.array([0.9]), bin_stats=bin_stats)

    policy = CPSATPolicy(config=config)

    kwargs = {
        "distance_matrix": dist_matrix,
        "wastes": wastes,
        "capacity": 1.0,
        "day": 1,
        "scenario_tree": tree
    }

    route, cost, profit, search_ctx, md_ctx = policy.execute(**kwargs)
    # Should visit bin 1
    assert 1 in route
