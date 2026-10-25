"""Regression checks for explicit 2-opt action contracts."""

import pytest
import torch
from logic.src.envs.tsp_kopt import TSPkoptEnv
from tensordict import TensorDict


def test_mixed_action_modes_are_resolved_per_instance():
    td = TensorDict({"solution": torch.tensor([[0, 3, 1, 4, 2], [0, 3, 1, 4, 2]]),
                     "action": torch.tensor([[3, 2], [1, 3]]),
                     "action_is_position": torch.tensor([False, True])}, batch_size=[2])
    result = TSPkoptEnv()._step_instance(td)["solution"]
    assert result.tolist() == [[0, 3, 2, 4, 1], [0, 3, 4, 1, 2]]


@pytest.mark.parametrize("action,position", [([0, 2, 4], False), ([0, 99], False), ([0, 99], True), ([-1, 3], True)])
def test_unsupported_or_invalid_action_is_rejected_without_mutation(action, position):
    solution = torch.tensor([[0, 3, 1, 4, 2]])
    td = TensorDict({"solution": solution.clone(), "action": torch.tensor([action]),
                     "action_is_position": torch.tensor([position])}, batch_size=[1])
    with pytest.raises(ValueError):
        TSPkoptEnv()._step_instance(td)
    assert torch.equal(td["solution"], solution)
