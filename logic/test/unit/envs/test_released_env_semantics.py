"""The released PTP and TCPTP keep the dynamics of the pre-WP-07 envs (review CR-12).

Checked against rollouts of ``envs/routing/{ptp,tcptp}.py`` at a5eb27eec: PTP never offers an
empty bin unless it is mandatory, and a TCPTP day continues after a depot return while a
customer is still reachable on a fresh trip.
"""

import pytest
import torch
from logic.src.envs.ptp import PTPEnv
from logic.src.envs.tcptp import TCPTPEnv
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _raw(waste, **extra):
    n = len(waste)
    return TensorDict(
        {
            "locs": torch.tensor([[[0.1 * (i + 1), 0.0] for i in range(n)]]),
            "depot": torch.zeros(1, 2),
            "waste": torch.tensor([waste]),
            "capacity": torch.ones(1),
            **extra,
        },
        batch_size=[1],
    )


def test_ptp_masks_empty_bins_unless_mandatory():
    env = PTPEnv(generator_params={"num_loc": 3}, check_env_specs=False)
    assert env.reset(_raw([0.5, 0.0, 0.5]))["action_mask"].tolist() == [[True, True, False, True]]
    mandatory = torch.tensor([[False, True, False]])
    td = env.reset(_raw([0.5, 0.0, 0.5], mandatory=mandatory))
    # The empty bin is offered because it is mandatory, and the depot waits for it.
    assert td["action_mask"].tolist() == [[False, True, True, True]]


def test_tcptp_day_continues_after_a_depot_return():
    env = TCPTPEnv(generator_params={"num_loc": 2}, check_env_specs=False)
    td = env.reset(_raw([0.6, 0.6]))
    # Capacity 1 fits one bin per trip: the first return to the depot must not end the day.
    for action, done in ((1, False), (0, False), (2, False), (0, True)):
        td["action"] = torch.tensor([action])
        td = env.step(td)["next"]
        assert bool(td["done"].item()) == done, action
    assert td["visited"].all()
