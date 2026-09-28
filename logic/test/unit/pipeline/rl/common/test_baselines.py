"""
Tests for RL baseline implementations.
"""

import pytest
import torch
import torch.nn as nn
from logic.src.pipeline.rl.common.baselines import (
    CriticBaseline,
    ExponentialBaseline,
    NoBaseline,
    POMOBaseline,
    RolloutBaseline,
    WarmupBaseline,
    get_baseline,
)
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]





class TestBaselines:
    """Tests for various baseline strategies."""

    @pytest.fixture
    def mock_env(self, mocker):
        """Mock environment."""
        env = mocker.MagicMock()
        env.batch_size = [2]
        env.reset.side_effect = lambda x: x
        return env

    def test_no_baseline(self):
        """Test NoBaseline returns zeros."""
        bl = NoBaseline()
        reward = torch.tensor([1.0, 2.0, 3.0])
        val = bl.eval(None, reward)
        assert torch.all(val == 0)
        assert val.shape == reward.shape

    def test_exponential_baseline(self):
        """Test ExponentialBaseline moving average."""
        bl = ExponentialBaseline(beta=0.5)

        # First call
        r1 = torch.tensor([10.0, 20.0])
        val1 = bl.eval(None, r1)
        assert val1.mean() == 15.0

        # Second call: 0.5 * 15 + 0.5 * 25 = 20
        r2 = torch.tensor([20.0, 30.0])
        val2 = bl.eval(None, r2)
        assert val2.mean() == 20.0

    def test_rollout_baseline_init(self, am_setup):
        """Test RolloutBaseline copies the policy."""
        policy = am_setup
        bl = RolloutBaseline(policy=policy)

        assert bl.baseline_policy is not None
        assert bl.baseline_policy is not policy  # Should be a copy

        # Check if requires_grad is False
        for param in bl.baseline_policy.parameters():
            assert not param.requires_grad

    def test_rollout_baseline_eval(self, am_setup, mock_env, mocker):
        """Test RolloutBaseline evaluation (greedy rollout)."""
        policy = am_setup
        bl = RolloutBaseline(policy=policy)

        td = TensorDict({"loc": torch.randn(2, 5, 2)}, batch_size=[2])
        reward = torch.tensor([1.0, 2.0])

        # Mock policy call in _rollout
        mocker.patch.object(bl.baseline_policy, "forward", return_value={"reward": torch.tensor([1.5, 2.5])})

        val = bl.eval(td, reward, env=mock_env)
        assert torch.all(val == torch.tensor([1.5, 2.5]))

    def test_rollout_baseline_callback(self, am_setup, mock_env, mocker):
        """Test RolloutBaseline significance-based update."""
        policy = am_setup
        bl = RolloutBaseline(policy=policy, update_every=1, bl_alpha=0.05)

        # We need to mock _rollout to return specific values
        # candidate outperforms baseline significantly
        mocker.patch.object(
            bl,
            "_rollout",
            side_effect=[
                torch.tensor([2.1, 2.0, 2.5]),  # candidate (mean 2.2)
                torch.tensor([1.1, 1.0, 1.2]),  # baseline (mean 1.1)
            ],
        )

        # Promotion uses a private generated pool, never reporting validation.
        from types import SimpleNamespace

        comparison_env = SimpleNamespace(
            generator=lambda batch_size: TensorDict({"x": torch.zeros(batch_size, 1)}, [batch_size])
        )
        bl.configure_comparison(comparison_env, 3, seed=7)
        mock_setup = mocker.patch.object(bl, "setup")

        val_dataset = [1, 2, 3]  # Dummy
        bl.epoch_callback(policy, 0, val_dataset=val_dataset, env=mock_env)

        mock_setup.assert_called_once_with(policy)

    def test_critic_baseline(self, mocker):
        """Test CriticBaseline integration."""
        mock_critic = mocker.MagicMock(spec=nn.Module)

        # Add mock parameter with device to avoid ensure_tensordict failure
        mock_param = mocker.MagicMock()
        mock_param.device = torch.device('cpu')
        mock_critic.parameters.return_value = iter([mock_param])

        mock_critic.side_effect = lambda x: x["reward_pred"].unsqueeze(-1)

        bl = CriticBaseline(critic=mock_critic)

        td = TensorDict({"reward_pred": torch.tensor([1.2, 1.3])}, batch_size=[2])
        reward = torch.tensor([1.0, 1.0])

        val = bl.eval(td, reward)
        assert torch.all(val == torch.tensor([1.2, 1.3]))

    def test_warmup_baseline(self, mocker):
        """Test transition in WarmupBaseline."""
        target_bl = NoBaseline()
        warmup_bl = WarmupBaseline(target_bl, warmup_epochs=2)

        torch.tensor([1.0])

        # Epoch 0 (warmup starts) - alpha will be updated in callback
        # Initially alpha is 0 (implicitly)
        # Base class init doesn't set alpha, let's check evaluation

        # Update to epoch 0
        warmup_bl.epoch_callback(nn.Module(), 0)  # alpha = (0+1)/2 = 0.5
        assert warmup_bl.alpha == 0.5

        # Update to epoch 1
        warmup_bl.epoch_callback(nn.Module(), 1)  # alpha = (1+1)/2 = 1.0
        assert warmup_bl.alpha == 1.0

    def test_pomo_baseline(self):
        """Test POMOBaseline mean across starts."""
        bl = POMOBaseline()
        # Reward shape: [batch, num_starts]
        reward = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
        val = bl.eval(None, reward)

        # Expected: [[2.0, 2.0], [3.0, 3.0]]
        expected = torch.tensor([[2.0, 2.0], [3.0, 3.0]])
        assert torch.all(val == expected)

    def test_get_baseline(self, am_setup):
        """Test factory function."""
        bl = get_baseline("exponential", beta=0.1)
        assert isinstance(bl, ExponentialBaseline)
        assert bl.beta == 0.1

        bl_rollout = get_baseline("rollout", policy=am_setup)
        assert isinstance(bl_rollout, RolloutBaseline)
