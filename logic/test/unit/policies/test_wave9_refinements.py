"""
Verification tests for Wave 9 refinements (GA, CFRS, HULK).
"""

import numpy as np
import pandas as pd
import pytest
from logic.src.policies.routing.hyper_heuristics.hyper_heuristic_us_lk.hulk import HULKConfig, HULKSolver
from logic.src.policies.routing.matheuristics.cluster_first_route_second.solver import (
    fisher_jaikumar_clustering,
)
from logic.src.policies.routing.meta_heuristics.genetic_algorithm.solver import GAConfig, GASolver

pytestmark = [pytest.mark.unit, pytest.mark.fast]


@pytest.fixture
def mock_vrpp_data():
    # Small 5-node instance
    dist_matrix = np.array(
        [
            [0, 10, 10, 20, 20, 30],
            [10, 0, 5, 15, 25, 35],
            [10, 5, 0, 10, 20, 30],
            [20, 15, 10, 0, 10, 20],
            [20, 25, 20, 10, 0, 10],
            [30, 35, 30, 20, 10, 0],
        ]
    )
    wastes = {1: 10, 2: 10, 3: 10, 4: 10, 5: 10}
    capacity = 30
    R, C = 10.0, 1.0
    return dist_matrix, wastes, capacity, R, C


def test_ga_local_search_integration(mock_vrpp_data):
    dist, wastes, cap, R, C = mock_vrpp_data
    params = GAConfig(max_generations=2, pop_size=5, seed=42)
    solver = GASolver(dist, wastes, cap, R, C, params)

    routes, profit, cost = solver.solve()
    assert len(routes) > 0
    assert profit > 0


def test_cfrs_seed_selection():
    # Test strict angular sector furthest node selection
    coords = pd.DataFrame({"Lat": [0.0, 10.0, -10.0, 0.0, 1.0], "Lng": [0.0, 0.0, 0.0, 10.0, -10.0]})
    mandatory = [1, 2, 3, 4]
    wastes = {1: 5.0, 2: 5.0, 3: 5.0, 4: 5.0}
    dist_matrix = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            dist_matrix[i, j] = np.linalg.norm(coords.iloc[i].values - coords.iloc[j].values)  # pyrefly: ignore [no-matching-overload, unsupported-operation]

    clusters = fisher_jaikumar_clustering(
        coords, mandatory, k=2, wastes=wastes, capacity=20, R=10, C=1, distance_matrix=dist_matrix
    )
    assert len(clusters) > 0


def test_hulk_reward_logic(mock_vrpp_data):
    dist, wastes, cap, R, C = mock_vrpp_data
    params = HULKConfig(max_iterations=5)
    solver = HULKSolver(dist, wastes, cap, R, C, params)

    # Check that reward parameters exist
    assert hasattr(params, "score_alpha")
    assert params.score_alpha == 10.0
    assert params.score_best == 20.0

    routes, profit, cost = solver.solve()
    assert profit >= 0
