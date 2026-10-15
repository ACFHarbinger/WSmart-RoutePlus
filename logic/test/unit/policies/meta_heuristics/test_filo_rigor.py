

import numpy as np
import pytest
from logic.src.configs.policies.filo import FILOConfig
from logic.src.policies.routing.meta_heuristics.fast_iterative_localized_optimization.filo import FILOSolver

pytestmark = [pytest.mark.unit, pytest.mark.slow]



@pytest.fixture
def filo_inputs():
    """Small instance for FILO testing."""
    n_nodes = 20
    dist_matrix = np.random.randint(1, 100, size=(n_nodes + 1, n_nodes + 1))
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = (dist_matrix + dist_matrix.T) // 2 # Symmetric

    wastes = {i: 10.0 for i in range(1, n_nodes + 1)}
    capacity = 50.0
    R = 1.0
    C = 0.1
    params = FILOConfig(
        max_iterations=100,
        time_limit=10.0,
        seed=42
    )
    return dist_matrix, wastes, capacity, R, C, params

@pytest.mark.unit
def test_filo_initialization(filo_inputs):
    """Test Clarke & Wright and Route Minimization."""
    dist_matrix, wastes, capacity, R, C, params = filo_inputs
    solver = FILOSolver(dist_matrix, wastes, capacity, R, C, params)

    # Test Clarke & Wright
    routes = solver._clarke_wright_initialization()
    assert isinstance(routes, list)
    assert len(routes) > 0

    # Verify all nodes are visited exactly once
    visited = [n for r in routes for n in r]
    assert sorted(visited) == list(range(1, len(dist_matrix)))

    # Test Route Minimization
    min_routes = solver._route_minimization(routes)
    assert len(min_routes) <= len(routes)
    visited_min = [n for r in min_routes for n in r]
    assert sorted(visited_min) == list(range(1, len(dist_matrix)))

@pytest.mark.unit
def test_filo_ruin_random_walk(filo_inputs):
    """Test Random Walk ruin mechanism."""
    dist_matrix, wastes, capacity, R, C, params = filo_inputs
    solver = FILOSolver(dist_matrix, wastes, capacity, R, C, params)

    routes = solver._clarke_wright_initialization()
    seed_node = routes[0][0] # Just some seed node

    new_routes, _, ruined = solver.ruin_recreate.apply(
        routes=routes,
        seed=seed_node,
        all_customers=solver.all_customers,
        mandatory_nodes=[],
        omega_intensity=1.0
    )

    assert isinstance(ruined, list)
    assert len(ruined) > 0

    # Verify resulting routes are valid (all mandatory nodes if any, capacity respected)
    visited_new = [n for r in new_routes for n in r]
    assert len(visited_new) > 0
    for r in new_routes:
        assert sum(wastes[n] for n in r) <= capacity + 1e-6

@pytest.mark.unit
def test_filo_solve_smoke(filo_inputs):
    """Smoke test for full FILO solver solve() loop."""
    dist_matrix, wastes, capacity, R, C, params = filo_inputs
    solver = FILOSolver(dist_matrix, wastes, capacity, R, C, params)

    best_routes, best_profit, best_cost = solver.solve()

    assert isinstance(best_routes, list)
    assert isinstance(best_profit, float)
    assert isinstance(best_cost, float)

    # Basic validity check
    visited = [n for r in best_routes for n in r]
    assert len(visited) <= solver.n_nodes # VRPP might not visit everyone
    for r in best_routes:
        load = sum(wastes[n] for n in r)
        assert load <= capacity + 1e-6
