"""A custom sim.policies entry keeps the selection it sets.

``expand_policy_configs`` fans the policy yaml out into packing variants
and then pins each variant's ``packing`` back onto the stored config.
A caller that already set a selection, such as lookahead, was overwritten by the
yaml's own last-minute variants, so that selection could not be expressed.
"""

import pytest
from logic.src.configs import Config
from logic.src.pipeline.features.test.config import expand_policy_configs
from logic.src.pipeline.simulations.actions.base import _as_policy_mapping
from logic.src.pipeline.simulations.day_context import find_policy_keys
from omegaconf import OmegaConf

pytestmark = [pytest.mark.unit, pytest.mark.fast]

_LOOKAHEAD = {"other/pk_lookahead.yaml": "lookahead"}


def _expand(policies):
    cfg = Config()
    cfg.sim.data_distribution = "emp"
    cfg.sim.policies = OmegaConf.create(policies)
    expand_policy_configs(cfg)
    return cfg


def _selections(cfg):
    """Selection values the day loop will run, one per stored policy."""
    found = []
    for stored in cfg.sim.config_path.values():
        flat = _as_policy_mapping(stored)
        found.append(flat.get("packing", find_policy_keys(stored).get("packing")))
    return found


def test_custom_dict_lookahead_is_not_replaced_by_the_yaml_variant():
    """The selection nested in the custom section survives variant pinning."""
    cfg = _expand(
        [
            {
                "alns": {
                    "custom": {
                        "time_limit": 2.0,
                        "engine": "custom",
                        "packing": _LOOKAHEAD,
                        "route_improvement": {"other/ri_cls.yaml": "default"},
                    }
                }
            }
        ]
    )

    selections = _selections(cfg)
    assert selections, "expand_policy_configs stored no policy config"
    for selection in selections:
        assert selection == [_LOOKAHEAD] or selection == _LOOKAHEAD


def test_resolved_custom_variant_keeps_its_own_packing():
    """A one-key override that already names the selection is stored as given."""
    cfg = _expand(
        [
            {
                "swc_tcf": {
                    "engine": "gurobi",
                    "time_limit": 60.0,
                    "packing": _LOOKAHEAD,
                    "route_improvement": {"other/ri_cls.yaml": "default"},
                }
            }
        ]
    )

    selections = _selections(cfg)
    assert len(selections) == 1
    assert selections[0] == [_LOOKAHEAD] or selections[0] == _LOOKAHEAD


def test_yaml_variant_is_still_pinned_when_the_caller_sets_no_selection():
    """A plain policy name still expands to the yaml's own selection variants."""
    cfg = _expand(["alns"])

    selections = _selections(cfg)
    assert len(selections) >= 2
    texts = [str(selection) for selection in selections]
    assert all("last_minute" in text for text in texts)
    assert all("lookahead" not in text for text in texts)
    assert len(set(texts)) >= 2
