

import numpy as np
import pytest
from logic.src.policies.routing.meta_heuristics.firefly_algorithm.params import FAParams
from logic.src.policies.routing.meta_heuristics.firefly_algorithm.solver import FASolver
from logic.src.policies.routing.meta_heuristics.harmony_search.params import HSParams
from logic.src.policies.routing.meta_heuristics.harmony_search.solver import HSSolver
from logic.src.policies.routing.meta_heuristics.sine_cosine_algorithm.params import SCAParams
from logic.src.policies.routing.meta_heuristics.sine_cosine_algorithm.solver import SCASolver

pytestmark = [pytest.mark.unit, pytest.mark.fast]





@pytest.fixture
def profitable_problem():
    """Create a profitable problem instance."""
    dist_matrix = np.array([
        [0, 2, 2],
        [2, 0, 1],
        [2, 1, 0]
    ])
    wastes = {1: 10, 2: 10}
    capacity = 25
    R = 10.0
    C = 1.0
    return dist_matrix, wastes, capacity, R, C

def test_fa_refinements(profitable_problem):
    dist_matrix, wastes, capacity, R, C = profitable_problem
    params = FAParams(max_iterations=10, pop_size=5)
    solver = FASolver(dist_matrix, wastes, capacity, R, C, params)

    routes, profit, cost = solver.solve()
    assert profit > 0
    assert len(routes) > 0

def test_hs_refinements(profitable_problem):
    dist_matrix, wastes, capacity, R, C = profitable_problem
    params = HSParams(max_iterations=10, hm_size=5)
    solver = HSSolver(dist_matrix, wastes, capacity, R, C, params)

    routes, profit, cost = solver.solve()
    assert profit > 0
    assert len(routes) > 0

def test_sca_refinements(profitable_problem):
    dist_matrix, wastes, capacity, R, C = profitable_problem
    params = SCAParams(max_iterations=10, pop_size=5)
    solver = SCASolver(dist_matrix, wastes, capacity, R, C, params)

    routes, profit, cost = solver.solve()
    assert profit > 0
    assert len(routes) > 0
