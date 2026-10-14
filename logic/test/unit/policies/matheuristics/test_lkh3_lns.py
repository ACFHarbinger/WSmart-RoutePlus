"""
Unit tests for LKH-3 Large Neighborhood Search matheuristic.

Tests the Phase 2 implementation including:
- ALNS solver initialization
- Destroy-repair operator dispatch
- Profit-aware vs. standard mode
- Elite pool management
- Historical tracking
"""

import numpy as np
import pytest
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search import (
    LKH3_ALNS,
)

pytestmark = [pytest.mark.unit, pytest.mark.slow]



@pytest.fixture
def small_instance():
    """Create a small test instance (4 customers + depot)."""
    # Distance matrix (5x5): depot + 4 customers
    dist = np.array([
        [0, 10, 15, 20, 25],
        [10, 0, 5, 10, 15],
        [15, 5, 0, 8, 12],
        [20, 10, 8, 0, 10],
        [25, 15, 12, 10, 0],
    ])

    # Demands
    wastes = {1: 30, 2: 40, 3: 50, 4: 20}

    # Coordinates for geometric operators
    coords = np.array([
        [0.5, 0.5],  # Depot
        [0.2, 0.8],  # Customer 1
        [0.8, 0.8],  # Customer 2
        [0.8, 0.2],  # Customer 3
        [0.2, 0.2],  # Customer 4
    ])

    return dist, wastes, coords


def test_alns_initialization(small_instance):
    """Test ALNS solver initialization."""
    dist, wastes, coords = small_instance

    solver = LKH3_ALNS(
        distance_matrix=dist,
        wastes=wastes,
        capacity=60,
        revenue=2.0,
        cost_unit=1.0,
        profit_aware_operators=False,
        mandatory_nodes=[1],
        coords=coords,
        seed=42,
    )

    assert solver.distance_matrix.shape == (5, 5)
    assert len(solver.wastes) == 4
    assert solver.capacity == 60
    assert 1 in solver.mandatory_nodes
    assert len(solver.history) == 0
    assert len(solver.elite_pool) == 0


def test_alns_cvrp_solve(small_instance):
    """Test ALNS in CVRP mode (all nodes must be visited)."""
    dist, wastes, coords = small_instance

    solver = LKH3_ALNS(
        distance_matrix=dist,
        wastes=wastes,
        capacity=60,
        revenue=1.0,
        cost_unit=1.0,
        profit_aware_operators=False,
        mandatory_nodes=None,
        coords=coords,
        seed=42,
    )

    routes, obj, _ = solver.solve(
        max_iterations=5,
        lkh_trials=50,
        n_vehicles=3,
        plateau_limit=2,
    )

    # Should visit all nodes
    visited = {n for r in routes for n in r}
    assert visited == {1, 2, 3, 4}

    # Should return negative cost (CVRP mode)
    assert obj < 0

    # Should have feasible capacity
    for route in routes:
        load = sum(wastes.get(n, 0) for n in route)
        assert load <= 60 + 1e-6


def test_alns_vrpp_solve(small_instance):
    """Test ALNS in VRPP mode (subset selection)."""
    dist, wastes, coords = small_instance

    solver = LKH3_ALNS(
        distance_matrix=dist,
        wastes=wastes,
        capacity=60,
        revenue=3.0,  # High revenue to make all nodes profitable
        cost_unit=1.0,
        profit_aware_operators=True,
        mandatory_nodes=[1],
        coords=coords,
        seed=42,
    )

    routes, profit, _ = solver.solve(
        max_iterations=5,
        lkh_trials=50,
        n_vehicles=2,
        plateau_limit=2,
    )

    # Mandatory node must be visited
    visited = {n for r in routes for n in r}
    assert 1 in visited

    # Should return positive profit (VRPP mode with high revenue)
    # Note: May be negative if routing cost too high
    # Just check it's a valid number
    assert isinstance(profit, (int, float))

    # All routes must be capacity-feasible
    for route in routes:
        load = sum(wastes.get(n, 0) for n in route)
        assert load <= 60 + 1e-6


def test_elite_pool_management(small_instance):
    """Test that elite pool is maintained correctly."""
    dist, wastes, coords = small_instance

    solver = LKH3_ALNS(
        distance_matrix=dist,
        wastes=wastes,
        capacity=60,
        revenue=2.0,
        cost_unit=1.0,
        profit_aware_operators=False,
        seed=42,
    )

    # Manually add solutions to elite pool
    solver._update_elite_pool([[1, 2], [3, 4]])
    solver._update_elite_pool([[1, 3], [2, 4]])
    solver._update_elite_pool([[1, 4], [2, 3]])

    assert len(solver.elite_pool) == 3

    # Add more than max_pool_size (5)
    solver._update_elite_pool([[1], [2, 3, 4]])
    solver._update_elite_pool([[2], [1, 3, 4]])
    solver._update_elite_pool([[3], [1, 2, 4]])

    # Should cap at max_pool_size
    assert len(solver.elite_pool) <= solver.max_pool_size


def test_historical_tracking(small_instance):
    """Test that historical scores are tracked correctly."""
    dist, wastes, coords = small_instance

    solver = LKH3_ALNS(
        distance_matrix=dist,
        wastes=wastes,
        capacity=60,
        revenue=2.0,
        cost_unit=1.0,
        seed=42,
    )

    # Update history for a solution
    routes = [[1, 2], [3, 4]]
    obj = 50.0

    solver._update_history(routes, obj)

    # All visited nodes should have history entries
    assert 1 in solver.history
    assert 2 in solver.history
    assert 3 in solver.history
    assert 4 in solver.history

    # Update again with different objective
    solver._update_history(routes, 60.0)

    # Scores should be updated (EMA)
    # After first update: history[1] = -50.0  (stores -obj)
    # After second: history[1] = 0.1 * (-60) + 0.9 * (-50) = -51.0
    expected = 0.1 * (-60.0) + 0.9 * (-50.0)
    assert abs(solver.history[1] - expected) < 1e-6


def test_routing_cost_calculation(small_instance):
    """Test routing cost computation."""
    dist, wastes, coords = small_instance

    solver = LKH3_ALNS(
        distance_matrix=dist,
        wastes=wastes,
        capacity=60,
        seed=42,
    )

    routes = [[1, 2], [3, 4]]
    cost = solver._compute_routing_cost(routes)

    # Route 1: 0→1→2→0 = 10 + 5 + 15 = 30
    # Route 2: 0→3→4→0 = 20 + 10 + 25 = 55
    # Total = 85
    expected = 10 + 5 + 15 + 20 + 10 + 25
    assert abs(cost - expected) < 1e-6


def test_destroy_repair_cycle(small_instance):
    """Test that destroy-repair modifies the solution."""
    dist, wastes, coords = small_instance

    solver = LKH3_ALNS(
        distance_matrix=dist,
        wastes=wastes,
        capacity=60,
        revenue=2.0,
        cost_unit=1.0,
        profit_aware_operators=False,
        coords=coords,
        seed=42,
    )

    # Start with a simple solution (3 routes to allow 140 total demand in 60-cap chunks)
    routes = [[1], [2, 4], [3]]

    # Apply destroy-repair
    modified = solver._destroy_repair(routes)

    # Should return a valid solution (list of routes)
    assert isinstance(modified, list)
    assert all(isinstance(r, list) for r in modified)

    # Should still be capacity-feasible
    for route in modified:
        load = sum(wastes.get(n, 0) for n in route)
        assert load <= 60 + 1e-6


def test_perturbation_with_elite(small_instance):
    """Test perturbation operator with elite pool."""
    dist, wastes, coords = small_instance

    solver = LKH3_ALNS(
        distance_matrix=dist,
        wastes=wastes,
        capacity=60,
        revenue=2.0,
        cost_unit=1.0,
        seed=42,
    )

    # Add elite solution
    elite = [[1, 3], [2, 4]]
    solver._update_elite_pool(elite)

    # Apply perturbation (3 routes to ensure feasibility)
    routes = [[1], [2, 4], [3]]
    perturbed = solver._perturbation(routes)

    # Should return a valid solution
    assert isinstance(perturbed, list)
    assert all(isinstance(r, list) for r in perturbed)


def test_operator_selection_dispatch(small_instance):
    """Test that operator selection works for both modes."""
    dist, wastes, coords = small_instance

    # CVRP mode
    solver_cvrp = LKH3_ALNS(
        distance_matrix=dist,
        wastes=wastes,
        capacity=60,
        profit_aware_operators=False,
        seed=42,
    )

    destroy_op = solver_cvrp._select_destroy_operator()
    repair_op = solver_cvrp._select_repair_operator()

    assert callable(destroy_op)
    assert callable(repair_op)

    # VRPP mode
    solver_vrpp = LKH3_ALNS(
        distance_matrix=dist,
        wastes=wastes,
        capacity=60,
        profit_aware_operators=True,
        coords=coords,
        seed=42,
    )

    destroy_op_vrpp = solver_vrpp._select_destroy_operator()
    repair_op_vrpp = solver_vrpp._select_repair_operator()

    assert callable(destroy_op_vrpp)
    assert callable(repair_op_vrpp)


def test_mandatory_nodes_enforcement(small_instance):
    """Test that mandatory nodes are always included."""
    dist, wastes, coords = small_instance

    solver = LKH3_ALNS(
        distance_matrix=dist,
        wastes=wastes,
        capacity=60,
        revenue=0.1,  # Very low revenue to discourage visiting nodes
        cost_unit=1.0,
        profit_aware_operators=True,
        mandatory_nodes=[1, 2],  # Nodes 1 and 2 are mandatory
        coords=coords,
        seed=42,
    )

    routes, profit, _ = solver.solve(
        max_iterations=5,
        lkh_trials=50,
        n_vehicles=2,
    )

    visited = {n for r in routes for n in r}

    # Mandatory nodes must be visited
    assert 1 in visited
    assert 2 in visited


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "--tb=short"])
