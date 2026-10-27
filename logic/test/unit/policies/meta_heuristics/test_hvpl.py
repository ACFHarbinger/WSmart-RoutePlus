import numpy as np
import pytest
from logic.src.configs.policies import HyperHeuristicACOConfig
from logic.src.configs.policies.acceptance_criteria import AcceptanceConfig, BoltzmannAcceptanceConfig
from logic.src.configs.policies.routing.meta_heuristics.alns import ALNSConfig
from logic.src.configs.policies.routing.meta_heuristics.hvpl import HVPLConfig
from logic.src.policies.routing.meta_heuristics.hybrid_volleyball_premier_league.solver import HVPLSolver

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_hvpl_solver():
    # Setup simple problem: 3 bins + depot
    dist_matrix = np.array(
        [[0.0, 10.0, 20.0, 10.0], [10.0, 0.0, 10.0, 20.0], [20.0, 10.0, 0.0, 10.0], [10.0, 20.0, 10.0, 0.0]]
    )
    waste = {1: 10.0, 2: 10.0, 3: 10.0}
    capacity = 50.0
    R = 10.0
    C = 1.0

    # Use very small params for fast test
    aco_params = HyperHeuristicACOConfig(
        n_ants=2,
        alpha=1.0,
        beta=2.0,
        rho=0.1,
        tau_0=1.0,
        eta_decay=0.5,
        max_iterations=1,  # Only one iteration per construction phase
        time_limit=60.0,
        elitism_ratio=1.0,  # ALNS handles local search
    )
    alns_params = ALNSConfig(
        max_iterations=5,
        acceptance_criterion=AcceptanceConfig(
            method="bmc", params=BoltzmannAcceptanceConfig(initial_temp=100.0, alpha=0.95, start_temp_control=0.05)
        ),
        reaction_factor=0.1,
        xi=0.3,
        time_limit=60.0,
    )
    params = HVPLConfig(
        n_teams=3,
        max_iterations=2,
        substitution_rate=0.3,
        time_limit=5.0,
        aco_params=aco_params,
        alns_params=alns_params,
    )

    solver = HVPLSolver(dist_matrix, waste, capacity, R, C, params)

    routes, profit, cost = solver.solve()

    # Basic sanity checks
    assert isinstance(routes, list)
    assert len(routes) > 0
    assert isinstance(profit, float)
    assert isinstance(cost, float)

    # Check that all nodes were visited (since it's a small problem)
    visited = set()
    for route in routes:
        for node in route:
            visited.add(node)
    assert visited == {1, 2, 3}


if __name__ == "__main__":
    test_hvpl_solver()
