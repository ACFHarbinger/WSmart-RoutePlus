"""
Unit tests for Wave 3 Advanced Hyper-Heuristics:
GP-HH, ACO-HH, RL-GD-HH, HMM-GD-HH, SS-HH.
"""

import random

import numpy as np
import pytest
from logic.src.configs.policies import HyperHeuristicACOConfig
from logic.src.policies.helpers.operators.solution_initialization import build_greedy_routes
from logic.src.policies.routing.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco import (
    HyperHeuristicACO,
)
from logic.src.policies.routing.hyper_heuristics.genetic_programming_hyper_heuristic.params import GPHHParams
from logic.src.policies.routing.hyper_heuristics.genetic_programming_hyper_heuristic.solver import GPHHSolver
from logic.src.policies.routing.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params import (
    HMMGDHHParams,
)
from logic.src.policies.routing.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver import (
    HMMGDHHSolver,
)
from logic.src.policies.routing.hyper_heuristics.sequence_based_selection_hyper_heuristic.params import (
    SSHHParams,
)
from logic.src.policies.routing.hyper_heuristics.sequence_based_selection_hyper_heuristic.solver import (
    SSHHSolver,
)
from logic.src.policies.routing.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.params import (
    RLGDHHParams,
)
from logic.src.policies.routing.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver import (
    RLGDHHSolver,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]



@pytest.fixture
def sample_data():
    dist_matrix = np.array([
        [0, 10, 20],
        [10, 0, 15],
        [20, 15, 0]
    ])
    wastes = {1: 10.0, 2: 20.0}
    capacity = 50.0
    return dist_matrix, wastes, capacity

def test_gphh_solver(sample_data):
    dist, wastes, cap = sample_data
    params = GPHHParams(gp_pop_size=2, max_gp_generations=1)
    solver = GPHHSolver(dist, wastes, cap, R=10.0, C=1.0, params=params)
    routes, profit, cost = solver.solve()
    assert isinstance(routes, list)
    assert profit >= 0

def test_sshh_solver(sample_data):
    dist, wastes, cap = sample_data
    # Use the correct param name if different, checking SSHHParams
    params = SSHHParams(max_iterations=5, time_limit=1.0)
    solver = SSHHSolver(dist, wastes, cap, R=10.0, C=1.0, params=params)
    routes, profit, cost = solver.solve()
    assert isinstance(routes, list)
    assert profit >= 0

def test_hyper_aco_solver(sample_data):
    dist, wastes, cap = sample_data
    params = HyperHeuristicACOConfig(max_iterations=2, n_ants=2)
    solver = HyperHeuristicACO(dist, wastes, cap, R=10.0, C=1.0, params=params)
    routes, profit, cost = solver.solve()
    assert isinstance(routes, list)
    assert profit >= -1e9 # Allow negative if cost is high

def test_rl_gd_hh_solver(sample_data):
    dist, wastes, cap = sample_data
    params = RLGDHHParams(max_iterations=5, time_limit=1.0)
    solver = RLGDHHSolver(dist, wastes, cap, R=10.0, C=1.0, params=params)
    routes, profit, cost = solver.solve()
    assert isinstance(routes, list)
    assert profit >= 0

def test_hmm_gd_hh_solver(sample_data):
    dist, wastes, cap = sample_data
    params = HMMGDHHParams(max_iterations=5, time_limit=1.0)
    solver = HMMGDHHSolver(dist, wastes, cap, R=10.0, C=1.0, params=params)
    rng = random.Random(params.seed)
    heuristic_routes = build_greedy_routes(
        dist_matrix=dist,
        wastes=wastes,
        capacity=cap,
        R=10.0,
        C=1.0,
        mandatory_nodes=[],
        rng=rng,
    )
    routes, profit, cost = solver.solve(initial_routes=heuristic_routes)
    assert isinstance(routes, list)
    assert profit >= 0
