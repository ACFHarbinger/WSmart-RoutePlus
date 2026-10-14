"""
Unit tests for the Scenario-Tree Extensive Form (ST-EF) policy.
"""

import numpy as np
import pytest
from logic.src.policies.routing.exact_and_decomposition_solvers.scenario_tree_extensive_form import (
    ScenarioTreeExtensiveFormPolicy,
)
from logic.src.policies.routing.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine import (
    GUROBI_AVAILABLE,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]



def _has_gurobi_license() -> bool:
    if not GUROBI_AVAILABLE:
        return False
    try:
        import gurobipy as _gp
        _m = _gp.Model("_check")
        _m.dispose()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_gurobi_license(), reason="Gurobi license not available for ST-EF solver")
def test_st_ef_basic():
    """Test ST-EF policy on a tiny instance."""
    n_nodes = 3
    # 0 -> Depot, 1/2 -> Customers
    dist_matrix = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.5],
        [1.0, 1.5, 0.0]
    ])

    # Initial wastes at the nodes (Day 0)
    wastes = {1: 50.0, 2: 50.0}
    capacity = 100.0

    # ST-EF Configuration
    config = {
        "num_days": 2,
        "num_realizations": 2,
        "mean_increment": 0.3,
        "time_limit": 30.0,
        "mip_gap": 0.01,
        "waste_weight": 1.0,
        "cost_weight": 0.1,
        "overflow_penalty": 10.0,
        "use_mtz": True,
        "seed": 42
    }

    policy = ScenarioTreeExtensiveFormPolicy(config=config)

    # Inject ScenarioTree and MultiDayContext for Multi-Period Policy
    from logic.src.interfaces.context.multi_day_context import (
        MultiDayContext,
    )
    from logic.src.pipeline.simulations.bins.prediction import ScenarioGenerator

    generator = ScenarioGenerator(seed=42, horizon=2)
    # n_nodes is 3 (0=depot, 1, 2)
    wastes_arr = np.zeros(n_nodes)
    for i, w in wastes.items():
        wastes_arr[i] = w

    bin_stats = {"means": np.zeros(n_nodes) + 0.3}
    scenario_tree = generator.generate(current_wastes=wastes_arr, bin_stats=bin_stats)

    kwargs = {
        "distance_matrix": dist_matrix,
        "wastes": wastes,
        "capacity": capacity,
        "mandatory": [],
        "R": 1.0,
        "C": 0.1,
        "number_vehicles": 1,
        "coords": np.array([[0, 0], [1, 1], [2, 2]]),
        "day": 1,
        "scenario_tree": scenario_tree,  # CRITICAL INJECTION
        "multi_day_context": MultiDayContext(dist_matrix, wastes, capacity), # pyrefly: ignore [bad-argument-type]
    }

    tour, cost, profit, search_ctx, multi_day_ctx = policy.execute(**kwargs)

    # Basic validity checks
    assert isinstance(tour, list)
    assert len(tour) >= 2
    assert tour[0] == 0
    assert tour[-1] == 0
    assert cost >= 0
    assert "expected_profit" in search_ctx.construction_metrics # pyrefly: ignore [missing-attribute]

    print(f"\nExtracted Tour (Day 1): {tour}")
    print(f"Total Cost: {cost:.2f}")
    print(f"Expected Value: {search_ctx.construction_metrics['expected_profit']:.2f}") # pyrefly: ignore [bad-index]

@pytest.mark.skipif(not _has_gurobi_license(), reason="Gurobi license not available for ST-EF solver")
def test_st_ef_no_waste():
    """Test ST-EF policy when there's nothing to collect."""
    n_nodes = 3
    dist_matrix = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.5],
        [1.0, 1.5, 0.0]
    ])

    wastes = {1: 0.0, 2: 0.0}
    capacity = 100.0

    config = {
        "num_days": 2,
        "num_realizations": 2,
        "mean_increment": 0.0, # NO GROWTH
        "time_limit": 10.0,
        "use_mtz": True,
        "seed": 42
    }

    policy = ScenarioTreeExtensiveFormPolicy(config=config)

    # Inject ScenarioTree and MultiDayContext for Multi-Period Policy
    from logic.src.interfaces.context.multi_day_context import (
        MultiDayContext,
    )
    from logic.src.pipeline.simulations.bins.prediction import ScenarioGenerator



    generator = ScenarioGenerator(seed=42, horizon=2)
    wastes_arr = np.zeros(n_nodes)
    for i, w in wastes.items():
        wastes_arr[i] = w

    bin_stats = {"means": np.zeros(n_nodes) + 0.0}
    scenario_tree = generator.generate(current_wastes=wastes_arr, bin_stats=bin_stats)

    kwargs = {
        "distance_matrix": dist_matrix,
        "wastes": wastes,
        "capacity": capacity,
        "day": 1,
        "R": 1.0,
        "C": 0.1,
        "scenario_tree": scenario_tree,
        "multi_day_context": MultiDayContext(dist_matrix, wastes, capacity), # pyrefly: ignore [bad-argument-type]
    }

    tour, cost, profit, search_ctx, multi_day_ctx = policy.execute(**kwargs)

    # Should probably stay at depot since cost > gain (gain is 0)
    assert tour == [0, 0]
    assert cost == 0.0
