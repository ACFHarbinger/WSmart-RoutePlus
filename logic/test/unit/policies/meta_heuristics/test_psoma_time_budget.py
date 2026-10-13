"""PSOMA honours its wall-clock budget through the training phase and the SA searches."""

import time

import numpy as np
from logic.src.policies.acceptance_criteria.base.factory import AcceptanceCriterionFactory
from logic.src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm import solver
from logic.src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.params import (
    PSOMAParams,
)


def test_solve_stops_near_time_limit_on_a_large_instance():
    n = 350
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 100, size=(n + 1, 2))
    dist = np.sqrt(((points[:, None] - points[None]) ** 2).sum(-1))
    wastes = {i: float(rng.uniform(5, 100)) for i in range(1, n + 1)}
    criterion = AcceptanceCriterionFactory.create(
        name="bmc", config={"initial_temp": 100.0, "alpha": 0.995, "seed": 42}, initial_temp=3.0
    )
    params = PSOMAParams(time_limit=2.0, seed=42, acceptance_criterion=criterion)
    start = time.perf_counter()
    solver.PSOMASolver(dist, wastes, 1000.0, 1.0, 1.0, params, []).solve()
    elapsed = time.perf_counter() - start
    assert elapsed < 6.0, f"solve took {elapsed:.1f}s against a 2s budget"
