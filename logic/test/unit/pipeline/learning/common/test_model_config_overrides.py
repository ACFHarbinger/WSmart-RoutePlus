"""Model constructor settings must survive Hydra's mapping representation."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
from logic.src.pipeline.features.train.model_factory.builder import _init_policy
from omegaconf import OmegaConf


@pytest.mark.parametrize("hydra", [True, False])
@pytest.mark.parametrize(
    "settings",
    [
        {"name": "temporal", "temporal_horizon": 7, "predictor_type": "lstm"},
        {"name": "moe", "num_experts": 8, "k": 3, "noisy_gating": False},
    ],
)
def test_top_level_model_settings_reach_constructor(hydra, settings):
    if hydra:
        cfg = OmegaConf.create({"task": "train", "train": {"env": {"name": "cptp"}, "policy": {"model": settings}}})
    else:
        cfg = SimpleNamespace(
            task="train",
            train=SimpleNamespace(
                env=SimpleNamespace(name="cptp"), policy=SimpleNamespace(model=SimpleNamespace(**settings))
            ),
        )

    with patch("logic.src.pipeline.features.train.model_factory.builder.import_target") as resolve:
        _init_policy(cast(Any, cfg), None)

    kwargs = resolve.return_value.call_args.kwargs
    assert kwargs == {"env_name": "cptp", **{k: v for k, v in settings.items() if k != "name"}}
