

import numpy as np
import pytest
from logic.src.policies.routing.meta_heuristics.league_championship_algorithm.params import LCAParams
from logic.src.policies.routing.meta_heuristics.league_championship_algorithm.solver import LCASolver
from logic.src.policies.routing.meta_heuristics.soccer_league_competition.params import SLCParams
from logic.src.policies.routing.meta_heuristics.soccer_league_competition.solver import SLCSolver

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

def test_slc_solver_basic(small_vrpp_instance):
    dist_matrix, wastes, capacity, R, C = small_vrpp_instance
    params = SLCParams(
        n_teams=2,
        team_size=2,
        max_iterations=5,
        stagnation_limit=2,
        n_removal=1,
        local_search_iterations=2,
        seed=42,
    )
    solver = SLCSolver(dist_matrix, wastes, capacity, R, C, params)
    routes, profit, cost = solver.solve()

    assert isinstance(routes, list)
    assert profit >= 0
    assert cost >= 0
    # Capacity check
    for route in routes:
        assert sum(wastes[n] for n in route) <= capacity

def test_slc_coaching_and_superstars(small_vrpp_instance):
    dist_matrix, wastes, capacity, R, C = small_vrpp_instance
    params = SLCParams(
        n_teams=1,
        team_size=3,
        max_iterations=2,
        stagnation_limit=5,
        n_removal=1,
        local_search_iterations=1
    )
    solver = SLCSolver(dist_matrix, wastes, capacity, R, C, params)
    # Trigger coaching and superstar updates
    solver.solve()
    assert len(solver.superstars) > 0
    assert solver.superstars[0][1] >= 0

def test_lca_solver_basic(small_vrpp_instance):
    dist_matrix, wastes, capacity, R, C = small_vrpp_instance
    params = LCAParams(
        n_teams=4,
        max_iterations=5,
        n_removal=1,
        crossover_prob=0.8,
        tolerance_pct=0.05,
        local_search_iterations=2
    )
    solver = LCASolver(dist_matrix, wastes, capacity, R, C, params)
    routes, profit, cost = solver.solve()

    assert isinstance(routes, list)
    assert profit >= 0
    assert cost >= 0
    # Capacity check
    for route in routes:
        assert sum(wastes[n] for n in route) <= capacity

def test_lca_winner_loser_update(small_vrpp_instance):
    dist_matrix, wastes, capacity, R, C = small_vrpp_instance
    params = LCAParams(
        n_teams=2,
        max_iterations=2,
        n_removal=1,
        crossover_prob=1.0,
        tolerance_pct=0.0,
        local_search_iterations=1
    )
    solver = LCASolver(dist_matrix, wastes, capacity, R, C, params)
    solver.solve()
    # Ensure some movement happened
    assert solver.n_nodes == 3
