"""The benchmark campaign runs the heuristics with their profit-aware operators on."""

import pytest
from logic.src.configs.policies.routing.hyper_heuristics.aco_hh import HyperHeuristicACOConfig
from logic.src.configs.policies.routing.meta_heuristics.alns import ALNSConfig
from logic.src.configs.policies.routing.meta_heuristics.hgs import HGSConfig
from logic.src.configs.policies.routing.meta_heuristics.pg_clns import PGCLNSConfig
from logic.src.configs.policies.routing.meta_heuristics.psoma import PSOMAConfig

pytestmark = [pytest.mark.unit, pytest.mark.fast]


@pytest.mark.parametrize("config", [ALNSConfig, HGSConfig, HyperHeuristicACOConfig, PGCLNSConfig, PSOMAConfig])
def test_profit_aware_operators_are_on_by_default(config):
    assert config().profit_aware_operators is True
