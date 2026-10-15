"""P-13: converted policy YAML loads into the typed dataclass; unknown keys fail."""

from pathlib import Path

import pytest
from logic.src.policies.routing.base import RoutingAlgorithmFactory
from logic.src.policies.routing.base.base_routing_policy import _structured_load, _unwrap_engine_section
from omegaconf import OmegaConf

POLICY_DIR = Path(__file__).resolve().parents[3] / "configs" / "policies"


@pytest.fixture(autouse=True, scope="module")
def _registered():
    RoutingAlgorithmFactory.ensure_registered()


def test_alns_yaml_loads_into_alns_config():
    section = OmegaConf.to_container(OmegaConf.load(POLICY_DIR / "policy_alns.yaml"), resolve=True)["alns"]
    config = RoutingAlgorithmFactory.get_adapter("alns", config={"alns": section, "seed": 1})._config
    assert config.time_limit == 60.0
    assert config.cooling_rate == pytest.approx(0.995)
    assert config.start_temp == 0


def test_unknown_key_fails_loudly():
    from logic.src.configs.policies.alns import ALNSConfig

    with pytest.raises(ValueError, match="not_a_real_field"):
        _structured_load(ALNSConfig, {"time_limit": 10.0, "not_a_real_field": 1})


def test_list_of_maps_is_rejected():
    with pytest.raises(TypeError, match="mapping"):
        _unwrap_engine_section([{"time_limit": 10.0}])
