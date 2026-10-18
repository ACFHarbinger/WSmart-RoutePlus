"""CVRP uses the routing base's depot-aligned load vector across every trip."""

import torch
from logic.src.envs.tasks.cvrp import CVRPEnv
from tensordict import TensorDict


def test_customer_demands_align_with_depot_and_reset_between_trips():
    env = CVRPEnv(generator_params={"num_loc": 2}, check_env_specs=False)
    raw = TensorDict(
        {
            "locs": torch.tensor([[[1.0, 0.0], [2.0, 0.0]]]),
            "depot": torch.zeros(1, 2),
            "demand": torch.tensor([[0.6, 0.6]]),
            "vehicle_capacity": torch.ones(1),
            "capacity": torch.ones(1),
        },
        batch_size=[1],
    )
    td = env.reset(raw)
    assert td["action_mask"].tolist() == [[False, True, True]]
    for action, remaining, done in ((1, 0.4, False), (0, 1.0, False), (2, 0.4, False), (0, 1.0, True)):
        td["action"] = torch.tensor([action])
        td = env.step(td)["next"]
        torch.testing.assert_close(td["remaining_capacity"], torch.tensor([remaining]))
        assert bool(td["done"].item()) == done
    assert td["visited"].all()
    assert env.get_reward(td, td["tour"]).item() == -6
