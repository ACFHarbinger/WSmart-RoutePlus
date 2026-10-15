"""The unreleased envs keep their pre-WP-07 dynamics on the ESPPRC base (review CR-12, Q-8b).

OP and PDP use the reproductions from the review; IRP, ThOP, ATSP, PCTSP and SPCTSP run on their original
generators. Seeded rollouts of every env were also replayed step by step against
``envs/routing/*.py`` at a5eb27eec (masks, done flags, rewards) when this landed.
"""

import pytest
import torch
from logic.src.envs import get_env
from logic.src.envs.tasks.op import OPEnv
from logic.src.envs.tasks.pdp import PDPEnv
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_op_offers_only_customers_within_the_distance_budget():
    env = OPEnv(generator_params={"num_loc": 2}, check_env_specs=False)
    raw = TensorDict(
        {
            "locs": torch.tensor([[[1.0, 0.0], [2.0, 0.0]]]),
            "depot": torch.zeros(1, 2),
            "prize": torch.tensor([[1.0, 2.0]]),
            "max_length": torch.tensor([0.1]),
        },
        batch_size=[1],
    )
    assert env.reset(raw)["action_mask"].tolist() == [[True, False, False]]


def test_op_reward_is_the_collected_prize():
    env = OPEnv(generator_params={"num_loc": 2}, check_env_specs=False)
    raw = TensorDict(
        {
            "locs": torch.tensor([[[0.1, 0.0], [0.2, 0.0]]]),
            "depot": torch.zeros(1, 2),
            "prize": torch.tensor([[1.0, 2.0]]),
            "max_length": torch.tensor([1.0]),
        },
        batch_size=[1],
    )
    td = env.reset(raw)
    for action in (2, 0):
        td["action"] = torch.tensor([action])
        td = env.step(td)["next"]
    assert bool(td["done"].item())
    assert env.get_reward(td, td["tour"]).item() == pytest.approx(2.0)


def test_pdp_delivery_opens_only_after_its_pickup():
    env = PDPEnv(generator_params={"num_loc": 2}, check_env_specs=False)
    raw = TensorDict({"locs": torch.tensor([[[1.0, 0.0], [2.0, 0.0]]]), "depot": torch.zeros(1, 2)}, batch_size=[1])
    td = env.reset(raw)
    assert td["action_mask"].tolist() == [[False, True, False]]
    td["action"] = torch.tensor([1])
    td = env.step(td)["next"]
    assert td["action_mask"].tolist() == [[False, False, True]]
    td["action"] = torch.tensor([2])
    td = env.step(td)["next"]
    assert bool(td["done"].item())
    assert env.get_reward(td, td["tour"]).item() == pytest.approx(-4.0)


@pytest.mark.parametrize("name", ["irp", "thop", "atsp", "pctsp", "spctsp"])
def test_rollout_on_the_original_generator(name):
    torch.manual_seed(0)
    env = get_env(name, generator_params={"num_loc": 6})
    td = env.reset(env.generator(batch_size=[4]))
    g = torch.Generator().manual_seed(1)
    actions = []
    for _ in range(80):
        if bool(td["done"].all()):
            break
        mask = td["action_mask"]
        assert mask.any(dim=-1).all(), "every live instance has a valid action"
        action = torch.multinomial(mask.float(), 1, generator=g).squeeze(-1)
        actions.append(action)
        td["action"] = action
        td = env.step(td)["next"]
    assert bool(td["done"].all())
    assert torch.isfinite(env.get_reward(td, torch.stack(actions, 1))).all()
