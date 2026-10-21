"""Validation must preserve the configured economic objective."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from logic.src.pipeline.learning.common.base.data import _create_eval_env_and_gen
from omegaconf import OmegaConf


@pytest.mark.parametrize(
    "reward, expected",
    [
        ({"cost_weight": 0.0, "waste_weight": 27.7}, (0.0, 27.7)),
        ({"cost_weight": 1.0, "waste_weight": 0.0}, (1.0, 0.0)),
        ({"cost_weight": None, "waste_weight": None}, (1.0, 1.0)),
        ({}, (1.0, 1.0)),
        ({"cost_weight": 2.5, "waste_weight": 27.7}, (2.5, 27.7)),
    ],
)
@pytest.mark.parametrize("per_eval_graph", [True, False])
def test_eval_objective_preserves_zero_and_defaults_only_none(reward, expected, per_eval_graph):
    graph_reward = reward if not per_eval_graph else {"cost_weight": 99.0, "waste_weight": 99.0}
    cfg = OmegaConf.create(
        {"task": "train", "device": "cpu", "train": {"env": {"name": "cptp", "graph": {"reward": graph_reward}}}}
    )
    eval_graph = {"num_loc": 2, "n_samples": 2}
    if per_eval_graph:
        eval_graph["reward"] = reward
    with patch("logic.src.pipeline.learning.envs.get_env", return_value=SimpleNamespace(generator=None)) as factory:
        _create_eval_env_and_gen(cfg, eval_graph)
    kwargs = factory.call_args.kwargs
    assert (kwargs["cost_weight"], kwargs["waste_weight"]) == expected
