"""
Simple test for Branch-and-Cut VRPP solver.

Run with: python -m pytest logic/src/policies/branch_and_cut/test_bc.py -v
"""

import numpy as np
import pandas as pd
import pytest
from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_cut import (
    BranchAndCutPolicy,
    VRPPModel,
)
from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_cut.bc import GUROBI_AVAILABLE
from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_cut.heuristics import (
    construct_initial_solution,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]



@pytest.mark.skipif(not GUROBI_AVAILABLE, reason="Gurobi not available")
def test_vrpp_model_creation():
    """Test VRPP model initialization."""
    n_nodes = 6
    cost_matrix = np.random.rand(n_nodes, n_nodes)
    cost_matrix = (cost_matrix + cost_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(cost_matrix, 0)

    wastes = {1: 10., 2: 15., 3: 8., 4: 20., 5: 12.}
    capacity = 50.0

    model = VRPPModel(
        n_nodes=n_nodes,
        cost_matrix=cost_matrix,
        wastes=wastes,
        capacity=capacity,
        revenue_per_kg=1.0,
        cost_per_km=0.5,
        mandatory_nodes={1, 3},
    )

    assert model.n_nodes == n_nodes
    assert model.capacity == capacity
    assert len(model.mandatory_nodes) == 2
    assert model.depot == 0


@pytest.mark.skipif(not GUROBI_AVAILABLE, reason="Gurobi not available")
def test_heuristic_solution():
    """Test heuristic solution construction."""
    n_nodes = 6
    cost_matrix = np.array([
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        [1.0, 0.0, 1.5, 2.5, 3.5, 4.5],
        [2.0, 1.5, 0.0, 1.0, 2.0, 3.0],
        [3.0, 2.5, 1.0, 0.0, 1.0, 2.0],
        [4.0, 3.5, 2.0, 1.0, 0.0, 1.0],
        [5.0, 4.5, 3.0, 2.0, 1.0, 0.0],
    ])

    wastes = {1: 10., 2: 15., 3: 8., 4: 20., 5: 12.}
    capacity = 50.0

    model = VRPPModel(
        n_nodes=n_nodes,
        cost_matrix=cost_matrix,
        wastes=wastes,
        capacity=capacity,
        revenue_per_kg=2.0,
        cost_per_km=1.0,
        mandatory_nodes={1},
    )

    tour, profit = construct_initial_solution(model)

    assert tour is not None
    assert len(tour) >= 2
    assert tour[0] == 0  # Starts at depot
    assert tour[-1] == 0  # Ends at depot
    assert 1 in tour  # Mandatory node included


@pytest.mark.skipif(not GUROBI_AVAILABLE, reason="Gurobi not available")
def test_small_instance_solve():
    """Test solving a small VRPP instance."""
    # Create a small 5-node instance
    n_nodes = 5
    coords = pd.DataFrame({
        "Lat": [0.0, 1.0, 0.5, -0.5, 1.5],
        "Lng": [0.0, 1.0, 1.5, 1.0, 0.5],
    })

    # Compute Euclidean distance matrix
    dist_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            dist_matrix[i, j] = np.sqrt(
                (coords.iloc[i]["Lat"] - coords.iloc[j]["Lat"]) ** 2
                + (coords.iloc[i]["Lng"] - coords.iloc[j]["Lng"]) ** 2
            )

    wastes = {1: 20, 2: 15, 3: 25, 4: 18}
    R = 2.0  # High revenue
    C = 1.0

    policy = BranchAndCutPolicy(
        config={
            "time_limit": 10.0,
            "mip_gap": 0.05,
            "verbose": True,
        }
    )

    tour, cost, profit, search_ctx, multi_day_ctx = policy.execute(
        coords=coords,
        mandatory=[1],
        distance_matrix=dist_matrix,
        wastes=wastes,
        R=R,
        C=C,
        bins=type("obj", (), {"c": [20, 15, 25, 18]}),
    )

    # profit is already extracted from the 5-tuple

    # Verify solution
    assert tour is not None
    assert len(tour) >= 2
    assert tour[0] == 0
    assert tour[-1] == 0
    assert 1 in tour  # Mandatory node
    assert profit > 0  # Should be profitable

    print("\nSolution found:")
    print(f"  Tour: {tour}")
    print(f"  Profit: {profit:.2f}")
    print(f"  Nodes visited: {len(set(tour)) - 1}")
