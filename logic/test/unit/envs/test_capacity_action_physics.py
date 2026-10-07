"""Capacity and action-shape physics for MVPTP."""

import pytest
import torch
from logic.src.envs.routing.mvptp import MVPTPEnv
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_mvptp_remaining_capacity_never_goes_negative() -> None:
    env = MVPTPEnv(generator_params={"num_loc": 2}, check_env_specs=False)
    td = TensorDict(
        {
            "locs": torch.tensor([[[0.0, 0.0], [1.0, 0.0]]]),
            "depot": torch.zeros(1, 2),
            "waste": torch.tensor([[0.0, 0.8]]),
            "capacity": torch.tensor([0.5]),
            "remaining_capacity": torch.tensor([0.3]),
            "collected_waste": torch.zeros(1),
            "visited": torch.tensor([[True, False]]),
            "current_node": torch.zeros(1, 1, dtype=torch.long),
            "tour": torch.zeros(1, 0, dtype=torch.long),
            "tour_length": torch.zeros(1),
            "action": torch.tensor([1]),
        },
        batch_size=[1],
    )

    next_td = env._step_instance(td)

    assert next_td["remaining_capacity"].item() == pytest.approx(0.0)
    assert next_td["remaining_capacity"].item() >= 0.0

