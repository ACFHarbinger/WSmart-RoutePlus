"""ALNS calibrates its start temperature from the initial profit by default (owner decision 2026-09-27)."""

import math

import numpy as np
import pytest
from logic.src.configs.policies.alns import ALNSConfig
from logic.src.configs.policies.other.acceptance_criteria import AcceptanceConfig, BoltzmannAcceptanceConfig
from logic.src.policies.acceptance_criteria.boltzmann_metropolis_criterion import BoltzmannAcceptance
from logic.src.policies.routing.base.sub_problem import build_subproblem
from logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search.alns import ALNSSolver

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_default_config_calibrates():
    assert ALNSConfig().acceptance_criterion.params.initial_temp == 0.0


def test_calibrated_temperature_overrides_the_criterion_default():
    rng = np.random.default_rng(0)
    pts = rng.random((8, 2)) * 10
    dist = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
    wastes = {i: 40.0 for i in range(1, 8)}
    criterion = BoltzmannAcceptance(initial_temp=0.0, alpha=0.995, seed=0, start_temp_control=0.05)
    params = ALNSConfig(
        acceptance_criterion=AcceptanceConfig(
            method="bmc", params=BoltzmannAcceptanceConfig(initial_temp=0.0, alpha=0.995, start_temp_control=0.05)
        ),
        max_iterations=0,
        time_limit=5.0,
        seed=1,
    )
    params.acceptance_criterion = criterion
    solver = ALNSSolver(
        build_subproblem(dist, wastes, capacity=400.0, revenue=1.0, cost_unit=0.2),
        params,
    )
    _routes, profit, _cost = solver.solve()
    expected = 0.05 * max(abs(profit), 1.0) / math.log(2.0)
    assert pytest.approx(expected, rel=1e-6) == solver.acceptance_criterion.T
    assert criterion.T == 0.0
