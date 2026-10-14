"""An explicit null disables a policy yaml packing."""

from logic.src.configs import Config
from logic.src.pipeline.features.test.config import expand_policy_configs
from logic.src.pipeline.simulations.actions.base import _flatten_config
from omegaconf import OmegaConf


def test_custom_null_selection_overrides_yaml():
    cfg = Config()
    cfg.sim.data_distribution = "emp"
    cfg.sim.policies = OmegaConf.create([{"alns": {"custom": [{"packing": None}]}}])
    expand_policy_configs(cfg)
    assert cfg.sim.config_path
    for stored in cfg.sim.config_path.values():
        assert _flatten_config(stored).get("packing") is None
