"""
Tests for RL algorithms (REINFORCE, PPO).
"""

from typing import Any, cast

import pytest
import torch
import torch.nn as nn
from logic.src.pipeline.learning.core.ppo import PPO
from logic.src.pipeline.learning.core.reinforce import REINFORCE
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]



class TestAlgorithms:
    """Tests for REINFORCE and PPO implementations."""

    @pytest.fixture
    def mock_env(self, mocker):
        """Mock environment."""
        env = mocker.MagicMock()
        env.reset.side_effect = lambda x: x
        return env

    def test_reinforce_loss(self, am_setup, mock_env, mocker):
        """Test REINFORCE loss calculation with advantage."""
        policy = am_setup
        model = REINFORCE(policy=policy, env=mock_env, baseline="exponential")

        td = TensorDict({"loc": torch.randn(2, 5, 2)}, batch_size=[2])
        out = {"reward": torch.tensor([1.5, 2.5]), "log_likelihood": torch.tensor([-0.5, -0.7], requires_grad=True)}

        # Mock baseline eval
        mocker.patch.object(model.baseline, "eval", return_value=torch.tensor([1.0, 1.0]))
        mocker.patch.object(model, "log")

        loss = model.calculate_loss(td, out, 0, env=mock_env)

        assert isinstance(loss, torch.Tensor)
        assert loss.grad_fn is not None  # Should be backpropagatable
        # advantage = [0.5, 1.5]
        # norm_advantage = [-1, 1] (roughly)
        # loss = -(-1 * -0.5 + 1 * -0.7) / 2 = -(0.5 - 0.7) / 2 = 0.1
        assert loss.item() != 0

    def test_ppo_init(self, am_setup, mock_env, mocker):
        """Test PPO initialization and parameters."""
        critic = mocker.MagicMock(spec=nn.Module)
        model = PPO(policy=am_setup, env=mock_env, critic=critic, ppo_epochs=5, mini_batch_size=1)

        assert model.ppo_epochs == 5
        assert model.mini_batch_size == 1
        assert not model.automatic_optimization  # PPO uses manual optimization

    def test_ppo_training_step(self, am_setup, mock_env, mocker):
        """Test PPO training step manual optimization loop."""
        critic = mocker.MagicMock(spec=nn.Module)
        model = PPO(policy=am_setup, env=mock_env, critic=critic, ppo_epochs=1, mini_batch_size=1)

        # Mock policy rollout
        mock_rollout_out = {
            "actions": torch.zeros(2, 5, dtype=torch.long),
            "log_likelihood": torch.tensor([-0.5, -0.6]),
            "reward": torch.tensor([1.0, 2.0]),
            "entropy": torch.tensor([0.1, 0.1], requires_grad=True),
        }
        mocker.patch.object(model.policy, "forward", return_value=mock_rollout_out)

        # Mock critic
        mock_val = torch.tensor([0.9, 1.9])
        critic.return_value = mock_val

        # Also need to handle sub-batch calls since it uses DataLoader
        def critic_side_effect(td):
            bs = td.batch_size[0]
            return torch.zeros(bs, 1)

        critic.side_effect = critic_side_effect

        # Mock Lightning internals for manual optimization
        from pytorch_lightning.core.optimizer import LightningOptimizer



        mock_opt = mocker.MagicMock(spec=LightningOptimizer)
        mock_opt.zero_grad = mocker.MagicMock()
        mock_opt.step = mocker.MagicMock()
        mocker.patch.object(model, "optimizers", return_value=mock_opt)
        mocker.patch.object(model, "manual_backward")
        mocker.patch.object(model, "clip_gradients")
        mocker.patch.object(model, "log")

        batch = TensorDict({"loc": torch.randn(2, 5, 2)}, batch_size=[2])

        loss = model.training_step(batch, 0)

        assert isinstance(loss, torch.Tensor)
        assert cast(Any, model.manual_backward).call_count >= 1
        assert mock_opt.step.call_count >= 1
        assert cast(Any, model.log).call_count >= 1
