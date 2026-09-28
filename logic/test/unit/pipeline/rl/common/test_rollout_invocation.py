"""Dataset/batch parity for both supported greedy policy interfaces."""

from types import SimpleNamespace

import pytest
import torch
from logic.src.data.datasets import TensorDictDataset
from logic.src.pipeline.rl.common.baselines import RolloutBaseline
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]


class ModernPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, td, env, strategy):
        assert strategy == "greedy" and env is not None
        return {"reward": td["x"].squeeze(-1) * self.weight}


class LegacyPolicy(ModernPolicy):
    def set_strategy(self, strategy):
        self.strategy = strategy

    def forward(self, td):
        assert self.strategy == "greedy"
        return td["x"].squeeze(-1) * self.weight, torch.zeros(len(td))


@pytest.mark.parametrize("policy_class", [ModernPolicy, LegacyPolicy])
def test_batch_dataset_parity_with_partial_last_batch(policy_class):
    policy = policy_class()
    baseline = RolloutBaseline(policy)
    data = TensorDict({"x": torch.tensor([[1.0], [2.0], [3.0]])}, [3])
    original = data.clone()
    env = SimpleNamespace(batch_size=[2], reset=lambda td: td)
    batch_rewards = baseline._rollout_batch(policy, data, env)
    dataset_rewards = baseline._rollout_dataset(policy, TensorDictDataset(data), env)
    assert torch.equal(batch_rewards, torch.tensor([2.0, 4.0, 6.0]))
    assert torch.equal(dataset_rewards, batch_rewards)
    assert torch.equal(data["x"], original["x"])
    assert not batch_rewards.requires_grad and not dataset_rewards.requires_grad


def test_modern_requires_environment_but_legacy_batch_does_not():
    data = TensorDict({"x": torch.ones(2, 1)}, [2])
    baseline = RolloutBaseline()
    with pytest.raises(ValueError, match="Environment"):
        baseline._rollout_batch(ModernPolicy(), data)
    assert torch.equal(baseline._rollout_batch(LegacyPolicy(), data), torch.full((2,), 2.0))
