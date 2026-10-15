"""
Integration tests for GENIUS solver integrity.
"""

import numpy as np
import pytest
from logic.src.configs.policies.genius import GENIUSConfig
from logic.src.policies.routing.meta_heuristics.genius.solver import GENIUSSolver

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_genius_node_conservation():
    """Verify that nodes are not lost during GENIUS US iterations."""
    dist_matrix = np.array([
        [0, 1, 1, 1], # 0 (depot)
        [1, 0, 1, 1], # 1
        [1, 1, 0, 1], # 2
        [1, 1, 1, 0], # 3
    ], dtype=float)

    wastes = {1: 10.0, 2: 10.0, 3: 10.0}
    # Tight capacity: only 2 nodes fit in a route
    capacity = 25.0
    R = 100.0
    C = 1.0
    params = GENIUSConfig(
        n_iterations=5,
        neighborhood_size=5,
        random_us_sampling=False, # Use deterministic for reproducibility
        profit_aware_operators=True,
    )

    solver = GENIUSSolver(dist_matrix, wastes, capacity, R, C, params)
    routes, profit, cost = solver.solve()

    # All nodes should be routed in VRPP mode with high revenue
    routed_nodes = [n for r in routes for n in r]
    assert len(routed_nodes) == 3
    assert set(routed_nodes) == {1, 2, 3}

def test_genius_mandatory_conservation():
    """Explicitly test mandatory node conservation."""
    dist_matrix = np.array([
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
    ], dtype=float)

    wastes = {1: 10.0, 2: 10.0, 3: 10.0}
    capacity = 15.0 # Only 1 node fit per route
    R = 1.0 # Low revenue
    C = 1000.0 # High cost -> would prefer to drop nodes

    mandatory_nodes = [1, 2, 3]
    params = GENIUSConfig(
        n_iterations=5,
        neighborhood_size=5,
        random_us_sampling=False,
        profit_aware_operators=True,
    )

    solver = GENIUSSolver(dist_matrix, wastes, capacity, R, C, params, mandatory_nodes=mandatory_nodes)
    routes, profit, cost = solver.solve()

    # Even if revenue is low and cost is high, mandatory nodes MUST stay
    routed_nodes = [n for r in routes for n in r]
    assert set(routed_nodes) == {1, 2, 3}
