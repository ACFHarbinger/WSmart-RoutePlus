"""CPTP day end: a depot visit unloads, depot-from-depot stops, and the stop is permanent."""

import pytest
import torch
from logic.src.pipeline.rl.envs.cptp import CPTPEnv
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _reset(batch: int = 2, num_loc: int = 4, capacity: float = 0.5) -> tuple[CPTPEnv, TensorDict]:
    env = CPTPEnv(generator_params={"num_loc": num_loc}, check_env_specs=False)
    gen = torch.Generator().manual_seed(0)
    td = TensorDict(
        {
            "locs": torch.rand(batch, num_loc, 2, generator=gen),
            "depot": torch.full((batch, 2), 0.5),
            "waste": torch.full((batch, num_loc), 0.2),
            "capacity": torch.full((batch,), capacity),
        },
        batch_size=[batch],
    )
    return env, env.reset(td)


def _step(env: CPTPEnv, td: TensorDict, actions: list[int]) -> TensorDict:
    td["action"] = torch.tensor(actions)
    return env.step(td)["next"]


def test_depot_return_unloads_and_the_day_continues() -> None:
    env, td = _reset()
    td = _step(env, td, [1, 1])
    td = _step(env, td, [2, 2])
    td = _step(env, td, [0, 0])

    assert not td["done"].any()
    assert torch.allclose(td["remaining_capacity"], td["capacity"])
    assert td["action_mask"][:, 3:].all()


def test_depot_from_depot_stops_one_instance_without_ending_the_other() -> None:
    env, td = _reset()
    td = _step(env, td, [1, 1])
    td = _step(env, td, [0, 0])
    td = _step(env, td, [0, 2])

    assert td["done"].reshape(-1).tolist() == [True, False]
    # The stopped instance can only stay at the depot.
    assert td["action_mask"][0].tolist() == [True, False, False, False, False]

    length_at_stop = td["tour_length"][0].clone()
    td = _step(env, td, [0, 3])
    td = _step(env, td, [0, 0])

    assert td["done"].reshape(-1).tolist() == [True, False]
    assert torch.allclose(td["tour_length"][0], length_at_stop)
    assert td["collected_waste"].tolist() == pytest.approx([0.2, 0.6])


def test_stopping_as_the_first_action_collects_nothing() -> None:
    env, td = _reset(batch=1)
    td = _step(env, td, [0])

    assert td["done"].all()
    assert td["collected_waste"].item() == 0.0


def test_day_ends_at_the_depot_when_no_bin_is_left() -> None:
    env, td = _reset(batch=1, num_loc=2, capacity=1.0)
    td = _step(env, td, [1])
    td = _step(env, td, [2])
    assert not td["done"].any()
    td = _step(env, td, [0])

    assert td["done"].all()
