"""Parameter loading must not depend on simulator initialization."""

from types import SimpleNamespace

import numpy as np
import pytest
from logic.src.configs.policies.acceptance_criteria import (
    AcceptanceConfig,
    BoltzmannAcceptanceConfig,
)
from logic.src.policies.routing.base.sub_problem import build_subproblem
from logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search.alns import (
    ALNSSolver,
)
from logic.src.policies.routing.meta_heuristics.hybrid_genetic_search.hgs import (
    HGSSolver,
)


@pytest.mark.parametrize("solver_cls", [ALNSSolver, HGSSolver])
def test_plain_mapping_honours_bmc_without_simulator(solver_cls):
    raw = {
        "acceptance_criterion": {"method": "bmc", "params": {"initial_temp": 137.0, "alpha": 0.91, "seed": 7}},
        "time_limit": 3.5,
        "seed": 7,
    }
    dist = np.zeros((2, 2))
    solver = solver_cls(build_subproblem(dist, {1: 1.0}, 10.0, 1.0, 1.0), raw)
    assert solver.cfg.time_limit == 3.5
    assert solver.acceptance_criterion.T == 137.0
    assert solver.acceptance_criterion.alpha == 0.91


@pytest.mark.parametrize("solver_cls", [ALNSSolver, HGSSolver])
def test_mapping_and_attribute_configs_have_parameter_parity(solver_cls):
    acceptance = AcceptanceConfig(
        method="bmc", params=BoltzmannAcceptanceConfig(initial_temp=137.0, alpha=0.91, seed=7)
    )
    raw = {
        "time_limit": 3.5,
        "max_iterations": 31,
        "start_temp": 2.0,
        "cooling_rate": 0.8,
        "reaction_factor": 0.3,
        "seed": 7,
        "acceptance_criterion": acceptance,
    }
    dist = np.zeros((2, 2))
    s_mapping = solver_cls(build_subproblem(dist, {1: 1.0}, 10.0, 1.0, 1.0), raw)
    s_attrs = solver_cls(build_subproblem(dist, {1: 1.0}, 10.0, 1.0, 1.0), SimpleNamespace(**raw))
    assert s_mapping.cfg.time_limit == s_attrs.cfg.time_limit
    assert s_mapping.cfg.seed == s_attrs.cfg.seed
    assert s_mapping.acceptance_criterion.T == s_attrs.acceptance_criterion.T == 137.0


def test_hgs_mapping_preserves_population_and_offspring_alias():
    dist = np.zeros((2, 2))
    solver = HGSSolver(
        build_subproblem(dist, {1: 1.0}, 10.0, 1.0, 1.0),
        {"mu": 7, "lambda_param": 11, "time_limit": 1.5},
    )
    assert (solver.cfg.mu, solver.cfg.n_offspring, solver.cfg.time_limit) == (7, 11, 1.5)
