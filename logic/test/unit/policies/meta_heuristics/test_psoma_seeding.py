"""PSOMA is repeatable from its own seed, independent of the global RNGs (B-qwen-02)."""

import random

import numpy as np
import pytest
from logic.src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.params import (
    PSOMAParams,
)
from logic.src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver import (
    PSOMASolver,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _instance():
    rng = np.random.default_rng(3)
    pts = rng.random((9, 2)) * 10
    dist = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
    wastes = {i: float(v) for i, v in zip(range(1, 9), rng.uniform(5, 40, 8), strict=True)}
    return dist, wastes


def _run(seed, global_seed):
    random.seed(global_seed)
    np.random.seed(global_seed)
    dist, wastes = _instance()
    params = PSOMAParams.from_config({"pop_size": 4, "max_iterations": 3, "L": 2, "time_limit": 10.0, "seed": seed})
    routes, profit, _cost = PSOMASolver(dist, wastes, capacity=80.0, R=1.0, C=0.5, params=params).solve()
    return routes, profit


def test_same_seed_same_result_regardless_of_global_state():
    assert _run(seed=7, global_seed=1) == _run(seed=7, global_seed=999)
