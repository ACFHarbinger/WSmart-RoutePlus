

import numpy as np
import pytest
from logic.src.configs.policies.routing.meta_heuristics.lca import LCAConfig
from logic.src.configs.policies.routing.meta_heuristics.slc import SLCConfig
from logic.src.configs.policies.routing.meta_heuristics.vpl import VPLConfig
from logic.src.policies.routing.meta_heuristics.league_championship_algorithm.solver import LCASolver
from logic.src.policies.routing.meta_heuristics.soccer_league_competition.solver import SLCSolver
from logic.src.policies.routing.meta_heuristics.volleyball_premier_league.solver import VPLSolver

pytestmark = [pytest.mark.unit, pytest.mark.fast]





@pytest.fixture
def profitable_problem():
    # Large profit nodes very close to depot
    dist_matrix = np.array([
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0]
    ])
    wastes = {1: 100.0, 2: 100.0, 3: 100.0}
    capacity = 1000.0
    return dist_matrix, wastes, capacity

def test_slc_superstar_influence(profitable_problem):
    dist_matrix, wastes, capacity = profitable_problem
    params = SLCConfig(n_teams=2, team_size=2, max_iterations=5, seed=42)
    solver = SLCSolver(dist_matrix, wastes, capacity, R=10.0, C=1.0, params=params)

    routes, profit, cost = solver.solve()
    assert len(solver.superstars) > 0
    assert profit > 0

def test_vpl_coaching_weighted(profitable_problem):
    dist_matrix, wastes, capacity = profitable_problem
    # Relaxed elite_size for testing
    params = VPLConfig(n_teams=4, max_iterations=5, elite_size=3, seed=42)
    solver = VPLSolver(dist_matrix, wastes, capacity, R=10.0, C=1.0, params=params)

    routes, profit, cost = solver.solve()
    assert profit > 0

def test_lca_round_robin(profitable_problem):
    dist_matrix, wastes, capacity = profitable_problem
    params = LCAConfig(n_teams=4, max_iterations=5)
    solver = LCASolver(dist_matrix, wastes, capacity, R=10.0, C=1.0, params=params)

    routes, profit, cost = solver.solve()
    assert profit > 0
