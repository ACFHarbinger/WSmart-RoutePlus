"""Profitable nearest-neighbour expert: selective tours that the CPTP environment accepts."""

import pytest
import torch
from logic.src.envs.tasks.cptp import CPTPEnv
from logic.src.policies.vector.profitable_nearest_neighbour import ProfitableNearestNeighbourPolicy
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _reset(waste: list[list[float]], waste_weight: float = 10.0) -> tuple[CPTPEnv, TensorDict]:
    # Depot at the origin; bins 1-3 sit together one unit away, bin 4 is five units away.
    locs = torch.tensor([[1.0, 0.0], [1.0, 0.1], [1.1, 0.0], [0.0, 5.0]])
    batch = len(waste)
    env = CPTPEnv(generator_params={"num_loc": 4}, check_env_specs=False, waste_weight=waste_weight, cost_weight=1.0)
    td = TensorDict(
        {
            "locs": locs.expand(batch, 4, 2).clone(),
            "depot": torch.zeros(batch, 2),
            "waste": torch.tensor(waste),
            "capacity": torch.full((batch,), 10.0),
        },
        batch_size=[batch],
    )
    return env, env.reset(td)


def test_skips_the_far_bin_and_stays_home_when_no_tour_pays() -> None:
    env, td = _reset([[0.5, 0.5, 0.5, 0.1], [0.01, 0.0, 0.0, 0.01]])
    out = ProfitableNearestNeighbourPolicy("cptp")(td, env)

    visited = [set(a.tolist()) - {0} for a in out["actions"]]
    assert visited[0] == {1, 2, 3}
    assert visited[1] == set()
    assert out["actions"][1].tolist() == [0] * out["actions"].size(1)
    assert out["reward"][1].item() == pytest.approx(0.0)
    assert out["reward"][0].item() > 0


def test_every_expert_action_is_valid_so_teacher_forcing_is_finite() -> None:
    env, td = _reset([[0.5, 0.5, 0.5, 0.1], [0.01, 0.0, 0.0, 0.01], [0.9, 0.0, 0.9, 0.9]])
    actions = ProfitableNearestNeighbourPolicy("cptp")(td, env)["actions"]

    state = td.clone()
    for step in range(actions.size(1)):
        action = actions[:, step]
        assert state["action_mask"].gather(1, action.unsqueeze(-1)).all(), step
        state["action"] = action
        state = env.step(state)["next"]
    assert state["done"].all()


def test_ignores_bins_without_waste() -> None:
    env, td = _reset([[0.9, 0.0, 0.9, 0.0]])
    actions = ProfitableNearestNeighbourPolicy("cptp")(td, env)["actions"][0].tolist()

    assert set(actions) - {0} == {1, 3}
    # The tour returns to the depot and then ends the day there.
    assert actions[len([a for a in actions if a]) :][:2] == [0, 0]
