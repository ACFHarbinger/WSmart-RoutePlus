

import numpy as np
import pytest
from logic.src.policies.routing.meta_heuristics.quantum_differential_evolution.params import QDEParams
from logic.src.policies.routing.meta_heuristics.quantum_differential_evolution.solver import QDESolver

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

def test_qde_solver_basic(sample_data):
    dist, wastes, cap = sample_data
    params = QDEParams(max_iterations=10, pop_size=5, seed=42)
    solver = QDESolver(dist, wastes, cap, R=1.0, C=1.0, params=params)

    routes, profit, cost = solver.solve()

    assert isinstance(routes, list)
    assert profit >= 0
    assert cost >= 0
    # Success criterion: it should at least finish and return consistent types

def test_qde_rotation_gate(sample_data):
    dist, wastes, cap = sample_data
    params = QDEParams(delta_theta=0.1, seed=42)
    solver = QDESolver(dist, wastes, cap, R=1.0, C=1.0, params=params)

    # Test lookup table logic
    # Case: xi=0, bi=1, worsening=True -> rotation should be +0.1
    rot = solver._rotation_gate(0, 1, True, 0.0)
    assert rot == pytest.approx(0.1)

    # Case: xi=1, bi=0, worsening=True -> rotation should be -0.1
    rot = solver._rotation_gate(1, 0, True, 0.0)
    assert rot == pytest.approx(-0.1)

    # Case: same bits -> 0
    assert solver._rotation_gate(0, 0, True, 0.0) == 0.0
    assert solver._rotation_gate(1, 1, True, 0.0) == 0.0

def test_bpc_custom_engine(sample_data):
    dist, wastes, cap = sample_data
    # Wrap in a tiny trial config
    values = {"time_limit": 10, "max_cg_iterations": 2}
    from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine import (
        run_bpc,
    )

    routes, cost = run_bpc(dist, wastes, cap, R=1.0, C=1.0, values=values)

    assert isinstance(routes, list)
    assert cost >= 0
