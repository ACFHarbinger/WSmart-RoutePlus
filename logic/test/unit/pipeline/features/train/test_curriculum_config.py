"""Regression coverage for task-scoped curriculum graph configuration."""

import pytest
from logic.src.pipeline.features.train.engine import _build_stage_config, _get_primary_graph
from omegaconf import OmegaConf

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_curriculum_helpers_use_train_scoped_environment():
    """Hydra's ``train.env`` graph is not silently replaced by a root key."""
    cfg = OmegaConf.create(
        {
            "task": "train",
            "train": {"env": {"curriculum_graphs": [{"num_loc": 20, "n_days": 4}]}},
        }
    )

    assert _get_primary_graph(cfg, "n_days") == 4
    stage = _build_stage_config(cfg, cfg.train.env.curriculum_graphs[0])
    assert stage.train.env.graph.num_loc == 20
    assert stage.train.env.graph.n_days == 4
    assert "env" not in stage
