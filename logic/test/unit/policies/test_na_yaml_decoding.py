"""Fail-before: policy_na.yaml decoding / selection / improver must reach the adapter.

The yaml stores decoding under a nested ``decoding:`` map inside the ``amgat``
list. ``BaseRoutingPolicy._build_config`` used to keep only dataclass field
names, so ``beam_width: 5`` became the dataclass default 1.
"""

from pathlib import Path
from typing import Any, Dict

import pytest
from logic.src.constants.paths import CONFIGS_DIR
from logic.src.pipeline.simulations.actions.node_selection import MandatorySelectionAction
from logic.src.pipeline.simulations.actions.route_improvement import RouteImprovementAction
from logic.src.policies.route_construction.base.factory import RouteConstructorFactory
from logic.src.policies.route_construction.learning_algorithms.neural_agent.params import (
    NeuralParams,
)
from omegaconf import OmegaConf

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _na_yaml() -> Any:
    return OmegaConf.load(Path(CONFIGS_DIR) / "policies" / "policy_na.yaml")


def _adapter_from_yaml() -> Any:
    cfg = _na_yaml()
    return RouteConstructorFactory.get_adapter(
        "na",
        config={"na": OmegaConf.to_container(cfg.na, resolve=True)},
    )


def test_from_config_reads_yaml_beam_width() -> None:
    """from_config already unpacked decoding; the adapter used not to call it."""
    cfg = _na_yaml()
    params = NeuralParams.from_config(OmegaConf.to_container(cfg.na, resolve=True))
    assert params.decoding_strategy == "greedy"
    assert params.beam_width == 5
    assert params.reward_weight == pytest.approx(0.0)
    assert params.length_penalty_alpha == pytest.approx(0.0)


def test_na_adapter_honours_yaml_decoding_not_dataclass_beam_width() -> None:
    """get_adapter('na', policy_na.yaml) must not silently use beam_width=1."""
    adapter = _adapter_from_yaml()
    assert adapter.config is not None
    assert adapter.config.decoding_strategy == "greedy"
    assert adapter.config.beam_width == 5, (
        f"yaml decoding.beam_width is 5; adapter has {adapter.config.beam_width} "
        "(dataclass default is 1)"
    )
    assert adapter.config.reward_weight == pytest.approx(0.0)
    assert adapter.config.length_penalty_alpha == pytest.approx(0.0)


def test_na_adapter_honours_omegaconf_amgat_section() -> None:
    cfg = _na_yaml()
    adapter = RouteConstructorFactory.get_adapter("na", config={"na": cfg.na})
    assert adapter.config.beam_width == 5
    assert adapter.config.decoding_strategy == "greedy"


def test_na_yaml_lookahead_list_variant_loads_day_zero() -> None:
    parsed = MandatorySelectionAction()._parse_strategy_item({"other/ms_lookahead.yaml": ["default"]})
    assert parsed[0]["name"] == "lookahead"
    assert parsed[0]["params"].get("current_collection_day") == 0


def test_na_yaml_route_improvement_is_empty() -> None:
    cfg = _na_yaml()
    amgat = OmegaConf.to_container(cfg.na.amgat, resolve=True)
    merged: Dict[str, Any] = {}
    for item in amgat:
        if isinstance(item, dict):
            merged.update(item)
    assert merged.get("route_improvement") == []
    action = RouteImprovementAction()
    configs = action._get_route_improvement_configs({"config": {"na": merged}})
    assert configs == []


def test_execute_records_consumer_decoding_from_yaml(tmp_path, monkeypatch) -> None:
    """The policy consumer JSONL must show beam_width 5, not the dataclass 1."""
    import json

    capture = tmp_path / "na.jsonl"
    monkeypatch.setenv("WSR_CAPTURE_STRATEGY_JSON", str(capture))
    adapter = _adapter_from_yaml()
    adapter.execute(mandatory=[], search_context=None, multi_day_context=None)
    records = [json.loads(line) for line in capture.read_text().splitlines() if line.strip()]
    decoding = next(row["payload"] for row in records if row["kind"] == "consumer_decoding")
    assert decoding["strategy"] == "greedy"
    assert decoding["beam_width"] == 5
    ri_records = [row for row in records if row["kind"] == "consumer_ri"]
    assert ri_records == []  # execute does not run the improver action
