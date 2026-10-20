"""
Unit tests for Pure Branch-and-Price Solver.
"""

import numpy as np
import pytest
from logic.src.configs.policies.bp import BPConfig
from logic.src.policies.helpers.solvers_and_matheuristics import Route, VRPPMasterProblem
from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_price.bp import (
    BranchAndPriceSolver,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]



def test_master_problem_dual_signs():
    """Test that Master Problem returns non-negative duals for Set Packing."""
    n_nodes = 3
    mandatory_nodes = {1}
    cost_matrix = np.zeros((4, 4))
    wastes = {1: 10., 2: 10., 3: 10.}
    capacity = 100.

    mp = VRPPMasterProblem(
        n_nodes=n_nodes,
        mandatory_nodes=mandatory_nodes,
        cost_matrix=cost_matrix,
        wastes=wastes,
        capacity=capacity,
        revenue_per_kg=1.0,
        cost_per_km=1.0,
        vehicle_limit=2
    )

    # Create some routes
    r1 = Route(nodes=[1], cost=0, revenue=10, load=10, node_coverage={1})
    r2 = Route(nodes=[2], cost=0, revenue=10, load=10, node_coverage={2})
    r3 = Route(nodes=[3], cost=0, revenue=10, load=10, node_coverage={3})

    mp.add_route(r1)
    mp.add_route(r2)
    mp.add_route(r3)

    mp.build_model()
    obj, values = mp.solve_lp_relaxation()

    duals = mp.get_reduced_cost_coefficients()

    # Check node duals (should be >= 0 for packing/partitioning in MAX)
    node_duals = duals["node_duals"]
    for node in [1, 2, 3]:
        assert node_duals[node] >= 0, f"Dual for node {node} should be non-negative"

    # Check vehicle dual
    assert node_duals["vehicle_limit"] >= 0, "Vehicle limit dual should be non-negative"

def test_bp_solver_small_instance():
    """Test Branch-and-Price on a small instance."""
    n_nodes = 3
    # Distance matrix: 0-1: 10, 0-2: 20, 0-3: 30, etc.
    cost_matrix = np.array([
        [0, 10, 20, 30],
        [10, 0, 5, 15],
        [20, 5, 0, 10],
        [30, 15, 10, 0]
    ])
    wastes = {1: 10., 2: 20., 3: 15.}
    capacity = 50.
    R = 2.0
    C = 1.0

    # Case 1: No vehicle limit, all nodes profitable
    solver = BranchAndPriceSolver(
        n_nodes=n_nodes,
        cost_matrix=cost_matrix,
        wastes=wastes,
        capacity=capacity,
        revenue_per_kg=R,
        cost_per_km=C,
        mandatory_nodes={1},
        params=BPConfig(use_exact_pricing=True),
    )

    tour, profit, stats = solver.solve()

    assert 1 in tour
    assert profit > 0
    assert tour[0] == 0
    assert tour[-1] == 0

def test_bp_vehicle_limit():
    """Test that vehicle limit is respected and affects duals."""
    n_nodes = 3
    cost_matrix = np.zeros((4, 4))
    wastes = {1: 100., 2: 100., 3: 100.} # Very profitable
    capacity = 200.
    R = 10.0
    C = 1.0

    # Limit to 1 vehicle
    solver = BranchAndPriceSolver(
        n_nodes=n_nodes,
        cost_matrix=cost_matrix,
        wastes=wastes,
        capacity=capacity,
        revenue_per_kg=R,
        cost_per_km=C,
        mandatory_nodes=set(),
        params=BPConfig(vehicle_limit=1, use_exact_pricing=True),
    )

    tour, profit, stats = solver.solve()

    # Should only have one route (separated by depot returns)
    # tour like [0, 1, 2, 0] or [0, 1, 3, 0] etc.
    # Total count of 0s in middle should be 0 if only 1 vehicle
    depot_indices = [i for i, node in enumerate(tour) if node == 0]
    assert len(depot_indices) == 2, "Should only have one route due to vehicle limit"
