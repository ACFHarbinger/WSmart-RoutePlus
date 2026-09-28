"""Promotion lifecycle regressions: validation is never a training decision pool."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from logic.src.pipeline.rl.common.baselines import RolloutBaseline, WarmupBaseline
from logic.src.pipeline.rl.core.reinforce import REINFORCE
from omegaconf import OmegaConf
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def generate(batch_size):
    return TensorDict({"x": torch.rand(batch_size, 1)}, [batch_size])


def make_baseline():
    policy = torch.nn.Linear(1, 1)
    baseline = RolloutBaseline(policy)
    env = SimpleNamespace(generator=generate, batch_size=[2], name="vrpp")
    baseline.configure_comparison(env, 3, seed=123)
    return policy, baseline, env


@pytest.mark.parametrize(
    "delta,accepted",
    [
        (torch.tensor([1.0, 1.0, 1.3]), True),
        (torch.tensor([-1.0, -1.0, -1.3]), False),
        (torch.tensor([0.1, -0.1, 0.0]), False),
    ],
)
def test_promotion_refresh_only_after_significant_improvement(delta, accepted):
    policy, baseline, env = make_baseline()
    previous = baseline.comparison_dataset
    frozen = baseline.baseline_policy
    reporting = [7, 8, 9]
    rewards = torch.tensor([1.1, 1.0, 1.2])
    with patch.object(baseline, "_rollout", side_effect=[rewards + delta, rewards]) as rollout:
        baseline.epoch_callback(policy, 0, reporting, env)
    assert (baseline.comparison_dataset is not previous) == accepted
    assert (baseline.baseline_policy is not frozen) == accepted
    assert baseline.comparison_generation == int(accepted)
    for call in rollout.call_args_list:
        assert call.args[1] is previous
        assert call.args[2] is baseline.comparison_env
        assert call.args[2] is not env
    assert reporting == [7, 8, 9]
    assert policy.training


def test_no_comparison_does_not_replace_frozen_policy():
    policy = torch.nn.Linear(1, 1)
    baseline = RolloutBaseline(policy)
    frozen = baseline.baseline_policy
    with torch.no_grad():
        policy.weight.add_(100)
    baseline.epoch_callback(policy, 0)
    assert baseline.baseline_policy is frozen
    assert not torch.equal(frozen.weight, policy.weight)


def test_seeded_refresh_and_global_rng_isolation():
    state = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()
    policy = torch.nn.Linear(1, 1)
    # Capture after policy initialization consumes its own random values.
    state = torch.random.get_rng_state().clone()
    env = SimpleNamespace(generator=generate)
    a, b = RolloutBaseline(policy), RolloutBaseline(policy)
    a.configure_comparison(env, 3, 123)
    b.configure_comparison(env, 3, 123)
    assert torch.equal(a.comparison_dataset.data["x"], b.comparison_dataset.data["x"])
    assert torch.equal(a._generate_comparison(1).data["x"], b._generate_comparison(1).data["x"])
    assert not torch.equal(a.comparison_dataset.data["x"], a._generate_comparison(1).data["x"])
    assert torch.equal(state, torch.random.get_rng_state())
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])


@pytest.mark.parametrize("warmup", [0, 2])
def test_multigraph_reporting_is_separate_and_worse_candidate_rejected(warmup):
    env = SimpleNamespace(name="vrpp", generator=generate, batch_size=[2])
    reporting_env = SimpleNamespace(name="other", generator=generate, batch_size=[2])
    cfg = OmegaConf.create(
        {
            "seed": 7,
            "task": "train",
            "train": {"env": {"graph": {"n_samples": 3}, "eval_graphs": [{"num_loc": 20, "n_samples": 5}]}},
        }
    )
    model = REINFORCE(env=env, policy=torch.nn.Linear(1, 1), baseline="rollout", cfg=cfg, bl_warmup_epochs=warmup)
    with patch(
        "logic.src.pipeline.rl.common.base.data._create_eval_env_and_gen", return_value=(reporting_env, generate)
    ):
        model.setup("fit")
    baseline = model.baseline.baseline if isinstance(model.baseline, WarmupBaseline) else model.baseline
    assert model.val_dataset is None
    reporting = model.val_datasets[0]
    assert len(reporting) == 5 and len(baseline.comparison_dataset) == 3
    assert baseline.comparison_env.name == "vrpp"
    frozen = baseline.baseline_policy
    with patch.object(baseline, "_rollout", side_effect=[torch.zeros(3), torch.ones(3)]):
        model.on_train_epoch_end()
    assert baseline.baseline_policy is frozen
    assert model.val_datasets[0] is reporting


def test_invalid_rewards_and_failed_generation_do_not_promote():
    policy, baseline, _ = make_baseline()
    frozen, dataset = baseline.baseline_policy, baseline.comparison_dataset
    with patch.object(baseline, "_rollout", side_effect=[torch.tensor([float("nan"), 2.0, 3.0]), torch.ones(3)]):
        baseline.epoch_callback(policy, 0)
    assert baseline.baseline_policy is frozen
    with (
        patch.object(baseline, "_rollout", side_effect=[torch.tensor([2.1, 2.0, 2.5]), torch.tensor([1.1, 1.0, 1.2])]),
        patch.object(baseline, "_generate_comparison", side_effect=RuntimeError("generator failed")),
        pytest.raises(RuntimeError, match="generator failed"),
    ):
        baseline.epoch_callback(policy, 0)
    assert baseline.baseline_policy is frozen and baseline.comparison_dataset is dataset


@pytest.mark.parametrize("load_before_setup", [False, True])
def test_checkpoint_restores_comparison_identity(load_before_setup):
    policy, original, env = make_baseline()
    original.comparison_generation = 3
    original.comparison_dataset = original._generate_comparison(3)
    restored = RolloutBaseline(policy)
    if load_before_setup:
        restored.load_state_dict(original.state_dict())
        restored.configure_comparison(env, 7, seed=99)
    else:
        restored.configure_comparison(env, 7, seed=99)
        restored.load_state_dict(original.state_dict())
    with patch.object(restored, "_rollout", side_effect=[torch.zeros(3), torch.ones(3)]):
        restored.epoch_callback(policy, 0)
    assert restored.comparison_generation == 3
    assert torch.equal(original.comparison_dataset.data["x"], restored.comparison_dataset.data["x"])
    old_checkpoint = {k: v for k, v in original.state_dict().items() if k != "_extra_state"}
    restored.load_state_dict(old_checkpoint, strict=True)


@pytest.mark.parametrize("promote", [False, True])
def test_repeated_setup_keeps_pool_identity(promote):
    policy, baseline, env = make_baseline()
    candidate = torch.tensor([2.1, 2.0, 2.5]) if promote else torch.zeros(3)
    with patch.object(baseline, "_rollout", side_effect=[candidate, torch.tensor([1.1, 1.0, 1.2])]):
        baseline.epoch_callback(policy, 0)
    dataset = baseline.comparison_dataset
    generation = baseline.comparison_generation
    baseline.configure_comparison(env, 3, seed=123)
    assert baseline.comparison_dataset is dataset
    assert baseline.comparison_generation == generation


def test_real_vrpp_pool_and_greedy_rollout():
    from logic.src.envs.routing.vrpp import VRPPEnv
    from logic.src.models.core.attention_model.policy import AttentionModelPolicy

    env = VRPPEnv(num_loc=5, batch_size=2, device="cpu")
    env.NAME = env.name
    policy = AttentionModelPolicy(env_name="vrpp", embed_dim=16, hidden_dim=32, n_encode_layers=1, n_heads=2)
    baseline = RolloutBaseline(policy)
    baseline.configure_comparison(env, 3, seed=123)
    data = baseline.comparison_dataset.data.clone()
    rewards = baseline._rollout(baseline.baseline_policy, baseline.comparison_dataset, baseline.comparison_env)
    assert rewards.shape == (3,)
    assert torch.isfinite(rewards).all()
    assert torch.equal(data["locs"], baseline.comparison_dataset.data["locs"])
