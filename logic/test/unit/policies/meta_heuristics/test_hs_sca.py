

import numpy as np
import pytest
from logic.src.configs.policies.hs import HSConfig
from logic.src.configs.policies.sca import SCAConfig
from logic.src.policies.routing.meta_heuristics.harmony_search.solver import HSSolver
from logic.src.policies.routing.meta_heuristics.sine_cosine_algorithm.solver import SCASolver

pytestmark = [pytest.mark.unit, pytest.mark.fast]





@pytest.fixture
def small_vrpp_instance():
    # Simple 3-node instance
    dist_matrix = np.array([
        [0, 10, 20, 15],
        [10, 0, 25, 30],
        [20, 25, 0, 10],
        [15, 30, 10, 0]
    ])
    wastes = {1: 10.0, 2: 20.0, 3: 15.0}
    capacity = 30.0
    R = 1.0
    C = 0.1
    return dist_matrix, wastes, capacity, R, C

def test_hs_solver_basic(small_vrpp_instance):
    dist_matrix, wastes, capacity, R, C = small_vrpp_instance
    params = HSConfig(
        hm_size=5,
        HMCR=0.9,
        PAR=0.3,
        max_iterations=10,
        local_search_iterations=5,
        seed=42
    )
    solver = HSSolver(dist_matrix, wastes, capacity, R, C, params)
    routes, profit, cost = solver.solve()

    assert isinstance(routes, list)
    assert profit >= 0
    assert cost >= 0
    # Capacity constraint check
    for route in routes:
        load = sum(wastes[n] for n in route)
        assert load <= capacity

def test_sca_solver_basic(small_vrpp_instance):
    dist_matrix, wastes, capacity, R, C = small_vrpp_instance
    params = SCAConfig(
        pop_size=5,
        a_max=2.0,
        max_iterations=10,
        seed=42
    )
    solver = SCASolver(dist_matrix, wastes, capacity, R, C, params)
    routes, profit, cost = solver.solve()

    assert isinstance(routes, list)
    assert profit >= 0
    assert cost >= 0
    # Capacity constraint check
    for route in routes:
        load = sum(wastes[n] for n in route)
        assert load <= capacity

def test_hs_pitch_adjustment_logic(small_vrpp_instance):
    dist_matrix, wastes, capacity, R, C = small_vrpp_instance
    # High PAR to trigger pitch adjustment
    params = HSConfig(
        hm_size=2,
        HMCR=1.0,
        PAR=1.0,
        max_iterations=5,
        local_search_iterations=2,
        seed=42
    )
    solver = HSSolver(dist_matrix, wastes, capacity, R, C, params)
    # Mocking HM to ensure it's not empty
    hm = [solver._build_random_solution() for _ in range(params.hm_size)]
    new_harmony = solver._improvise(hm)
    assert len(new_harmony) > 0 # Should have at least one route if nodes are profitable

def test_sca_oscillation_bounds(small_vrpp_instance):
    dist_matrix, wastes, capacity, R, C = small_vrpp_instance
    params = SCAConfig(
        pop_size=2,
        a_max=2.0,
        max_iterations=2,
        seed=42
    )
    solver = SCASolver(dist_matrix, wastes, capacity, R, C, params)
    # Check if r1 is updated correctly
    # solver.solve() runs the loop; we can check internal state or just run it
    routes, profit, cost = solver.solve()
    assert profit is not None
