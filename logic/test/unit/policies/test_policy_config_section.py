"""A policy reads its yaml section even when the simulator keys it by the test_sim entry name.

``expand_policy_configs`` stores each variant as ``{entry_name: section}``. Entries such as ``aco_hh``
(config key ``hyper_aco``) or ``sans_og_a`` (config key ``sans``) used to miss the lookup and silently
run on the dataclass defaults.
"""

from pathlib import Path

import pytest
from logic.src.policies.route_construction.base import RouteConstructorFactory
from omegaconf import OmegaConf

POLICY_DIR = Path(__file__).resolve().parents[3] / "configs" / "policies"


def _section(filename, *path):
    node = OmegaConf.to_container(OmegaConf.load(POLICY_DIR / filename), resolve=True)
    for key in path:
        node = node[key]
    return node


@pytest.fixture(autouse=True, scope="module")
def _registered():
    RouteConstructorFactory.ensure_registered()


def test_aco_hh_entry_name_reads_its_yaml():
    section = _section("policy_aco_hh.yaml", "aco_hh")
    config = RouteConstructorFactory.get_adapter("aco_hh", config={"aco_hh": section, "seed": 1})._config
    keyed = RouteConstructorFactory.get_adapter("aco_hh", config={"hyper_aco": section, "seed": 1})._config
    assert config == keyed
    assert (config.n_ants, config.rho, config.time_limit, config.profit_aware_operators) == (10, 0.5, 60.0, True)


@pytest.mark.parametrize("variant", ["og_a", "og_b"])
def test_sans_legacy_variant_entry_name_selects_the_og_engine(variant):
    section = _section("policy_sans.yaml", "sans", variant)
    for wrapped in (section, OmegaConf.create(section)):  # the simulator passes a DictConfig section
        config = RouteConstructorFactory.get_adapter("sans", config={f"sans_{variant}": wrapped, "seed": 1})._config
        assert config.engine == "og"
        assert len(config.combination) == 7


def test_flat_config_with_a_dict_field_is_not_mistaken_for_a_section():
    flat = {"engine": "og", "time_limit": 5, "mandatory_selection": {"other/ms_last_minute.yaml": ["x"]}}
    config = RouteConstructorFactory.get_adapter("sans", config=flat)._config
    assert (config.engine, config.time_limit) == ("og", 5)


@pytest.mark.parametrize(
    "entry, filename, expected",
    [
        ("aco_hh", "policy_aco_hh.yaml", {"n_ants": 10, "rho": 0.5, "time_limit": 60.0}),
        ("bpc", "policy_bpc.yaml", {"cutting_planes": "all", "max_cg_iterations": 150, "max_bb_nodes": 2000}),
        ("swc_tcf", "policy_swc_tcf.yaml", {"time_limit": 60.0, "framework": "gurobi"}),
    ],
)
def test_omegaconf_custom_lists_reach_the_typed_config(entry, filename, expected):
    """The simulator passes each section as a DictConfig whose 'custom'/'gurobi' block is a ListConfig."""
    section = OmegaConf.create(_section(filename, entry))
    config = RouteConstructorFactory.get_adapter(entry, config={entry: section, "seed": 1})._config
    assert {k: getattr(config, k) for k in expected} == expected
    assert all(not OmegaConf.is_config(v) for v in vars(config).values())
