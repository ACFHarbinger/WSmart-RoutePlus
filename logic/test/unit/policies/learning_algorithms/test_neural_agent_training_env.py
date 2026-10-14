"""The simulator decodes neural routes in the training environment, with the simulator's capacity."""

import pytest
import torch
from logic.src.envs.routing.cptp import CPTPEnv
from logic.src.policies.route_construction.learning_algorithms.neural_agent.agent import NeuralAgent
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]

PROFIT_VARS = {"cost_km": 1.0, "revenue_kg": 0.5837, "bin_capacity": 47.5, "vehicle_capacity": 73.68}


class _RecordingModel(torch.nn.Module):
    """Stands in for a trained model: visits every valid bin in index order, then stops."""

    env_name = "cptp"
    problem = None

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Identity()
        self.calls = []

    def forward(self, td, env=None, strategy=None, **kwargs):
        self.calls.append((td.clone(), env))
        actions = []
        while not td["done"].all():
            mask = td["action_mask"]
            action = torch.where(mask[:, 1:].any(-1), mask[:, 1:].float().argmax(-1) + 1, torch.zeros_like(td["i"].view(-1)))
            actions.append(action)
            td["action"] = action
            td = env.step(td)["next"]
        return {"actions": torch.stack(actions, 1)}


def _input(waste):
    return {"locs": torch.rand(1, len(waste), 2), "depot": torch.zeros(1, 2), "waste": torch.tensor([waste])}


def test_simulator_day_uses_the_training_env_with_the_simulator_capacity() -> None:
    model = _RecordingModel()
    route, _, _ = NeuralAgent(model).compute_simulator_day(
        _input([0.5, 0.0, 0.3, 0.0]), (None, torch.zeros(5, 5)), torch.zeros(5, 5), PROFIT_VARS
    )

    td, env = model.calls[0]
    assert isinstance(env, CPTPEnv)
    assert td["capacity"].item() == pytest.approx(73.68)
    assert env.waste_weight == pytest.approx(0.5837 * 47.5)
    # Bins without waste are masked, so the route holds only bins 1 and 3.
    assert set(route) - {0} == {1, 3}


def test_mandatory_bins_are_kept_even_when_empty() -> None:
    model = _RecordingModel()
    mandatory = torch.tensor([[False, False, True, False, False]])
    route, _, _ = NeuralAgent(model).compute_simulator_day(
        _input([0.5, 0.0, 0.0, 0.0]), (None, torch.zeros(5, 5)), torch.zeros(5, 5), PROFIT_VARS, mandatory=mandatory
    )

    assert set(route) - {0} == {1, 2}


def test_empty_bin_mask_is_off_by_default() -> None:
    env = CPTPEnv(generator_params={"num_loc": 3}, check_env_specs=False)
    td = TensorDict(
        {"locs": torch.rand(1, 3, 2), "depot": torch.zeros(1, 2), "waste": torch.tensor([[0.5, 0.0, 0.2]]), "capacity": torch.tensor([5.0])},
        batch_size=[1],
    )
    assert env.reset(td.clone())["action_mask"][0].tolist() == [True, True, True, True]

    env.mask_empty_bins = True
    assert env.reset(td.clone())["action_mask"][0].tolist() == [True, True, False, True]
