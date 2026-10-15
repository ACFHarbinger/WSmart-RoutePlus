import numpy as np
import pandas as pd
import pytest
from logic.src.configs.policies.alns import ALNSConfig
from logic.src.policies.routing.matheuristics.cluster_first_route_second.solver import run_cf_rs
from logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search.alns import ALNSSolver

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_alns_segment_logic():
    # Small synthetic instance
    dist = np.array([[0, 10, 10], [10, 0, 5], [10, 5, 0]])
    wastes = {1: 10.0, 2: 10.0}
    capacity = 15.0
    params = ALNSConfig(max_iterations=150, start_temp=100.0, regret_pool="regret2")  # > 1 segment

    solver = ALNSSolver(dist, wastes, capacity, R=100.0, C=1.0, params=params)

    # Check initialization
    assert solver.segment_size == 100
    assert len(solver.destroy_weights) == 3
    assert len(solver.repair_weights) == 4

    routes, profit, cost = solver.solve()

    # Ensure it ran and produced something valid
    assert len(routes) > 0
    assert profit > 0


def test_cfrs_fisher_jaikumar():
    coords = pd.DataFrame({"Lat": [0, 1, -1, 0.1, -0.1], "Lng": [0, 0, 0, 1, 1]})
    mandatory = [1, 2, 3, 4]
    dist = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            dist[i, j] = np.linalg.norm(coords.iloc[i].values - coords.iloc[j].values)

    wastes = {1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0}
    capacity = 15.0

    routes, cost, metadata = run_cf_rs(coords, mandatory, dist, wastes, capacity, 1.0, 1.0, n_vehicles=2)

    assert len(metadata["clusters"]) >= 2
    assert len(routes) > 2  # Depot -> N1 -> ... -> Depot
