"""
Unit tests for Tabu Search (TS) policy.
"""

import numpy as np
import pandas as pd
import pytest
from logic.src.configs.policies.ts import TSConfig
from logic.src.policies.routing.meta_heuristics.tabu_search.policy_ts import TSPolicy
from logic.src.policies.routing.meta_heuristics.tabu_search.solver import TSSolver

pytestmark = [pytest.mark.unit, pytest.mark.fast]





def test_ts_solver_instantiation():
    """Test that TSSolver can be instantiated without import errors."""
    params = TSConfig(
        max_iterations=10,
        tabu_tenure=5,
        use_swap=True,
        use_relocate=True,
        use_2opt=True
    )
    dist_matrix = np.zeros((5, 5))
    wastes = {1: 10., 2: 10., 3: 10., 4: 10.}
    capacity = 50.0

    solver = TSSolver(
        dist_matrix=dist_matrix,
        wastes=wastes,
        capacity=capacity,
        R=1.0,
        C=0.5,
        params=params
    )
    assert solver is not None

def test_ts_policy_execution(mocker):
    """Test that TSPolicy can be executed on a small instance."""
    dist_matrix = np.array([
        [0, 10, 20, 15],
        [10, 0, 15, 25],
        [20, 15, 0, 10],
        [15, 25, 10, 0]
    ])

    config = TSConfig(
        max_iterations=5,
        tabu_tenure=2
    )
    policy = TSPolicy(config=config)

    coords = pd.DataFrame({"Lat": [0, 1, 2, 3], "Lng": [0, 1, 2, 3]})

    # Mock bins object
    class MockBins:
        def __init__(self, c):
            self.c = c
    bins = MockBins([30.0, 40.0, 50.0])

    tour, cost, profit, stats, _ = policy.execute(
        coords=coords,
        mandatory=[2],
        distance_matrix=dist_matrix,
        bins=bins,
        area="Figueira da Foz",
        waste_type="plastic",
        config={"revenue": 10.0}
    )

    assert isinstance(tour, list)
    assert 0 in tour
    assert 2 in tour
    assert "profit" in stats.construction_metrics
    assert stats.construction_metrics["profit"] >= 0
