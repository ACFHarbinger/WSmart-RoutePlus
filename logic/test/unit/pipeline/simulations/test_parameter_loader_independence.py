"""Parameter loading must not depend on simulator initialization."""

from types import SimpleNamespace

import pytest
from logic.src.configs.policies.other.acceptance_criteria import AcceptanceConfig, BoltzmannAcceptanceConfig
from logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.params import ALNSParams
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.params import HGSParams


@pytest.mark.parametrize("cls", [ALNSParams, HGSParams])
def test_plain_mapping_honours_bmc_without_simulator(cls):
    raw = {
        "acceptance_criterion": {"method": "bmc", "params": {"initial_temp": 137.0, "alpha": 0.91, "seed": 7}},
        "time_limit": 3.5,
        "seed": 7,
    }
    params = cls.from_config(raw)
    assert params.time_limit == 3.5
    assert params.acceptance_criterion.T == 137.0
    assert params.acceptance_criterion.alpha == 0.91


@pytest.mark.parametrize("cls", [ALNSParams, HGSParams])
def test_mapping_and_attribute_configs_have_parameter_parity(cls):
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
    mapping = cls.from_config(raw)
    attrs = cls.from_config(SimpleNamespace(**raw))
    assert mapping.time_limit == attrs.time_limit
    assert mapping.seed == attrs.seed
    assert mapping.acceptance_criterion.T == attrs.acceptance_criterion.T == 137.0


def test_hgs_mapping_preserves_population_and_offspring_alias():
    params = HGSParams.from_config({"mu": 7, "lambda_param": 11, "time_limit": 1.5})
    assert (params.mu, params.n_offspring, params.time_limit) == (7, 11, 1.5)
