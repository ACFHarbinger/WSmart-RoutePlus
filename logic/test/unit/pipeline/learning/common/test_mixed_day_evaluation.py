"""Evaluation day blocks may share a batch and must retain each row's day."""

from types import SimpleNamespace

import pytest
import torch
from logic.src.envs.cptp import CPTPEnv
from logic.src.pipeline.learning.common.base.steps import StepMixin
from tensordict import TensorDict


class ThresholdSelector:
    def select(self, fill_levels, **kwargs):
        return fill_levels > 0.5


@pytest.mark.parametrize(
    "days,expected",
    [
        (None, [[0.1, 0.2], [0.1, 0.2]]),
        ([0, 0], [[0.1, 0.2], [0.1, 0.2]]),
        ([1, 1], [[0.7, 0.8], [0.7, 0.8]]),
        ([0, 1], [[0.1, 0.2], [0.7, 0.8]]),
        ([1, 0], [[0.7, 0.8], [0.1, 0.2]]),
        ([-4, 99], [[0.1, 0.2], [0.7, 0.8]]),
    ],
)
@pytest.mark.parametrize("consumer", ["packing", "reset"])
def test_reset_and_packing_select_each_instances_day(days, expected, consumer):
    td = TensorDict(
        {
            "locs": torch.tensor([[[1.0, 0.0], [2.0, 0.0]]] * 2),
            "depot": torch.zeros(2, 2),
            "waste": torch.tensor([[[0.1, 0.2], [0.7, 0.8]]] * 2),
            "capacity": torch.full((2,), 5.0),
        },
        batch_size=[2],
    )
    if days is not None:
        td["current_day"] = torch.tensor(days)
    env = CPTPEnv(generator_params={"num_loc": 2}, check_env_specs=False)
    model = SimpleNamespace(env=env, mandatory_selector=ThresholdSelector())
    expected_waste = torch.cat([torch.zeros(2, 1), torch.tensor(expected)], dim=1)
    if consumer == "packing":
        packed = StepMixin._apply_packing(model, td)
        assert torch.equal(packed["mandatory"], expected_waste > 0.5)
    else:
        reset = env.reset(td)
        torch.testing.assert_close(reset["waste"], expected_waste)
