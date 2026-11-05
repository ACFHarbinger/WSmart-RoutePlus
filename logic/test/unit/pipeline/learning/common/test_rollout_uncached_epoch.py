"""Skipping an unused cache preserves live baselines, RNG and policy gradients."""

import copy
from typing import Any, cast

import pytest
import torch
from logic.src.data.datasets import TensorDictDataset
from logic.src.envs import get_env
from logic.src.models.core.attention_model.policy import AttentionModelPolicy
from logic.src.pipeline.learning.common.epoch import prepare_epoch
from logic.src.pipeline.learning.core.reinforce import REINFORCE
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]


class _Mandatory:
    def select(self, fill, **kwargs):
        mask = torch.zeros_like(fill, dtype=torch.bool)
        mask[:, 1] = True
        return mask


@pytest.mark.parametrize("packed", [False, True])
def test_uncached_epoch_preserves_rng_values_actions_and_gradients(packed):
    torch.manual_seed(42)
    env = get_env("cmptp", generator_params={"num_loc": 5}, check_env_specs=False)
    policy = AttentionModelPolicy(
        env_name="cmptp", embed_dim=16, hidden_dim=32, n_encode_layers=1, n_heads=4, normalization="layer"
    )
    module = REINFORCE(env=env, policy=policy, baseline="rollout", mandatory_selector=_Mandatory() if packed else None)
    module.log = lambda *args, **kwargs: None
    module.train()
    env.batch_size = torch.Size([4])
    raw = env.generator(batch_size=4)
    raw["current_day"] = torch.zeros(4, dtype=torch.long)
    initial_rng = torch.get_rng_state().clone()
    initial_state = copy.deepcopy(module.state_dict())
    outcomes = []
    for precompute in (True, False):
        module.load_state_dict(initial_state)
        module.zero_grad(set_to_none=True)
        torch.set_rng_state(initial_rng)
        dataset = TensorDictDataset(raw.clone())
        prepared = prepare_epoch(policy, env, module.baseline, dataset, 3, precompute_baseline=precompute)
        after_prepare_rng = torch.get_rng_state().clone()
        assert torch.equal(dataset.data["current_day"], torch.full((4,), 3))
        # Same source rows as Lightning's already-constructed DataLoader; wrapped
        # values are discarded before shared_step on later reloads as well.
        if precompute:
            batch = TensorDict({"data": dataset.data, "baseline": cast(Any, prepared).baseline}, batch_size=[4])
        else:
            batch = dataset.data
        loss = module.training_step(batch, 0)
        loss.backward()
        outcomes.append(
            (
                after_prepare_rng,
                torch.get_rng_state().clone(),
                module.last_out["actions"].clone(),
                loss.detach().clone(),
                [p.grad.clone() if p.grad is not None else None for p in policy.parameters()],
            )
        )
    old, new = outcomes
    for left, right in zip(old[:4], new[:4], strict=True):
        assert torch.equal(left, right)
    for left, right in zip(old[4], new[4], strict=True):
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            assert torch.equal(left, right)
        else:
            assert left is None and right is None


def test_custom_policy_keeps_explicit_cache_contract():
    from types import SimpleNamespace

    from logic.src.pipeline.learning.common.baselines import RolloutBaseline

    class Custom(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, td, env, strategy):
            return {"reward": td["x"].squeeze(-1) * self.weight}

    baseline = RolloutBaseline(Custom())
    dataset = TensorDictDataset(TensorDict({"x": torch.ones(2, 1)}, [2]))
    prepared = baseline.prepare_uncached_dataset(dataset, SimpleNamespace(batch_size=[2], reset=lambda td: td))
    assert torch.equal(prepared.baseline, torch.full((2, 1), 2.0))


def test_stochastic_custom_dataset_keeps_precomputation_rng_consumption():
    from logic.src.pipeline.learning.common.baselines import RolloutBaseline
    from torch.utils.data import Dataset

    class StochasticDataset(Dataset):
        def __init__(self, data):
            self.data = data
            self.reads = 0

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            self.reads += 1
            row = self.data[index].clone()
            row["waste"] = row["waste"] * torch.rand(())
            return row

    torch.manual_seed(42)
    env = get_env("cmptp", generator_params={"num_loc": 2}, check_env_specs=False)
    env.batch_size = torch.Size([2])
    policy = AttentionModelPolicy(
        env_name="cmptp", embed_dim=16, hidden_dim=32, n_encode_layers=1, n_heads=4, normalization="layer"
    )
    baseline = RolloutBaseline(policy)
    raw = env.generator(batch_size=2)
    initial_rng = torch.get_rng_state().clone()
    results = []
    for uncached in (False, True):
        torch.set_rng_state(initial_rng)
        dataset = StochasticDataset(raw.clone())
        prepared = (
            baseline.prepare_uncached_dataset(dataset, env) if uncached else baseline.wrap_dataset(dataset, env=env)
        )
        assert prepared.dataset is dataset
        assert dataset.reads == 2
        results.append((prepared.baseline, torch.get_rng_state().clone()))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])


def test_rollout_dataset_pads_a_short_last_batch():
    """An instance count that is not a multiple of the batch size must not break the padding."""
    from logic.src.pipeline.learning.common.baselines import RolloutBaseline

    torch.manual_seed(42)
    env = get_env("cmptp", generator_params={"num_loc": 5}, check_env_specs=False)
    env.batch_size = torch.Size([4])
    policy = AttentionModelPolicy(
        env_name="cmptp", embed_dim=16, hidden_dim=32, n_encode_layers=1, n_heads=4, normalization="layer"
    )
    baseline = RolloutBaseline(policy)
    raw = env.generator(batch_size=6)  # six instances, batch size four: the last batch has two
    raw["current_day"] = torch.zeros(6, dtype=torch.long)
    rewards = baseline._rollout_dataset(policy, TensorDictDataset(raw), env)
    assert rewards.shape == (6,)
    assert torch.isfinite(rewards).all()
