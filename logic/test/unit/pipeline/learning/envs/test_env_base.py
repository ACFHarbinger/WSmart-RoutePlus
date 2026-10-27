"""Tests for RL4COEnvBase."""

from typing import cast

import pytest
import torch
from logic.src.envs.base import RL4COEnvBase
from tensordict import TensorDict, TensorDictBase

pytestmark = [pytest.mark.unit, pytest.mark.fast]


class ConcreteEnv(RL4COEnvBase):
    def _reset_instance(self, tensordict):
        tensordict["visited"] = torch.zeros((*tensordict.batch_size, 5), dtype=torch.bool)
        tensordict["current_node"] = torch.zeros((*tensordict.batch_size, 1), dtype=torch.long)
        tensordict["tour"] = tensordict["current_node"].clone()
        return tensordict

    def _get_reward(self, tensordict, actions=None):
        return torch.zeros(tensordict.batch_size)

    def _get_action_mask(self, tensordict):
        return torch.ones((*tensordict.batch_size, 5), dtype=torch.bool)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Perform a transition in the environment."""
        step_index = cast(torch.Tensor, tensordict["i"]) + 1
        tensordict["i"] = step_index
        action = cast(torch.Tensor, tensordict["action"])
        tensordict["current_node"] = action.unsqueeze(-1)
        tensordict["reward"] = self._get_reward(tensordict, action)
        return tensordict


def test_env_base_init():
    """Verify initialization of RL4COEnvBase."""
    env = ConcreteEnv(batch_size=4)
    assert env.batch_size == torch.Size([4])
    assert env.name == "base"


def test_env_base_reset():
    """Verify environment reset logic."""
    env = ConcreteEnv()
    td = TensorDict({"locs": torch.randn(2, 5, 2)}, batch_size=[2])

    out_td = env.reset(td)
    assert out_td.batch_size == torch.Size([2])
    assert "visited" in out_td.keys()  # type: ignore[not-iterable]
    assert "action_mask" in out_td.keys()  # type: ignore[not-iterable]
    assert out_td["i"].sum() == 0


def test_env_base_step():
    """Verify state transition steps."""
    env = ConcreteEnv()
    td = TensorDict(
        {
            "locs": torch.randn(2, 5, 2),
            "visited": torch.zeros(2, 5, dtype=torch.bool),
            "current_node": torch.zeros(2, 1, dtype=torch.long),
            "tour": torch.zeros(2, 1, dtype=torch.long),
            "i": torch.zeros(2, dtype=torch.long),
        },
        batch_size=[2],
    )

    # Simulate action
    td["action"] = torch.tensor([1, 2], dtype=torch.long)

    next_td = env._step(td)
    assert cast(torch.Tensor, next_td["i"]).tolist() == [1, 1]
    assert cast(torch.Tensor, next_td["current_node"]).flatten().tolist() == [1, 2]
    assert "reward" in next_td.keys()  # type: ignore[not-iterable]
