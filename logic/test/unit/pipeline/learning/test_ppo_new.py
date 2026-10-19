"""
Tests for PPO algorithm implementation.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from logic.src.pipeline.learning.core.ppo import PPO
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]



class MockModel(nn.Module):
    """Mock model that is a valid nn.Module."""
    def __init__(self, out=None, input_dim=10):
        super().__init__()
        self.linear = nn.Linear(1, 1) # dummy param
        self.out = out

    def forward(self, x, *args, **kwargs):
        if self.out is not None:
            bs = x.batch_size[0] if hasattr(x, "batch_size") else (len(x) if hasattr(x, "__len__") else 1)
            if torch.is_tensor(self.out) and self.out.dim() > 0:
                return self.out[:bs]
            if isinstance(self.out, dict):
                return {k: (v[:bs] if torch.is_tensor(v) and v.dim() > 0 else v) for k, v in self.out.items()}
            return self.out
        return torch.randn(1)

class MockBaseline(nn.Module):
    """Mock baseline that is a valid nn.Module."""
    def __init__(self):
        super().__init__()
        self.val = torch.zeros(1)
        self.unwrap_side_effect = None

    def unwrap_batch(self, batch):
        if self.unwrap_side_effect:
            return self.unwrap_side_effect(batch)
        return batch, None

class TestPPO:
    """Tests for PPO algorithm."""

    @pytest.fixture
    def setup_ppo(self):
        """Setup PPO instance with mocks."""
        env = MagicMock()
        policy = MockModel() # Real nn.Module
        critic = MockModel() # Real nn.Module

        # Determine device
        torch.device("cpu")

        module = PPO(
            env=env,
            policy=policy,
            critic=critic,
            ppo_epochs=2,
            mini_batch_size=2,  # Small batch size for testing
            batch_size=4,
            lr_scheduler="step"
        )

        # Mock trainer
        module.trainer = MagicMock()
        module.trainer.max_epochs = 10
        module.trainer.gradient_clip_val = None
        module.trainer.gradient_clip_algorithm = None

        return module, env, policy, critic

    def test_init(self, setup_ppo):
        """Test initialization."""
        module, env, policy, critic = setup_ppo

        assert module.critic == critic
        assert module.ppo_epochs == 2
        assert module.automatic_optimization is False

    def test_calculate_loss_dummy(self, setup_ppo):
        """Verify dummy calculate_loss returns 0 tensor."""
        module, _, _, _ = setup_ppo
        td = TensorDict({}, batch_size=[1], device="cpu")
        loss = module.calculate_loss(td, {}, 0)
        assert torch.isclose(loss, torch.tensor(0.0))

    def test_training_step(self, setup_ppo):
        """Test full training step execution."""
        module, env, policy, critic = setup_ppo

        # Mock batch
        batch_size = 4
        batch = TensorDict({
            "loc": torch.randn(batch_size, 2)
        }, batch_size=[batch_size])

        # Mock env reset
        env.reset.side_effect = lambda x: x

        # Update policy mock output
        out = {
            "actions": torch.zeros(batch_size, 5),
            "log_likelihood": torch.zeros(batch_size),
            "reward": torch.ones(batch_size),
            "entropy": torch.zeros(batch_size),
        }
        policy.out = out

        # Update critic mock output
        critic.out = torch.zeros(batch_size, 1)

        # Mock optimizer
        from pytorch_lightning.core.optimizer import LightningOptimizer


        class DummyLO(LightningOptimizer):
            def zero_grad(self, *args, **kwargs): pass
            def step(self, *args, **kwargs): pass

        optimizer = MagicMock(spec=DummyLO)
        optimizer.optimizer = MagicMock() # Inner optimizer
        module.optimizers = MagicMock(return_value=[optimizer])

        # Mock baseline
        module.baseline = MockBaseline()

        # Test execution
        with patch("torch.utils.data.DataLoader") as mock_dl:
            # Mock dataloader
            sub_td = batch.clone()
            sub_td.set("logprobs", torch.zeros(batch_size))
            sub_td.set("reward", torch.ones(batch_size))
            sub_td.set("action", torch.zeros(batch_size, 5))

            mock_dl.return_value = [sub_td]

            with patch.object(module, "log") as mock_log:
                # We need to spy on optimizer
                module.training_step(batch, batch_idx=0)

                assert optimizer.step.call_count == 4
                assert optimizer.zero_grad.call_count == 4
                mock_log.assert_called()

    def test_calculate_advantages(self, setup_ppo):
        """Test advantage calculation."""
        module, _, _, _ = setup_ppo

        rewards = torch.tensor([10.0, 10.0, 5.0, 5.0])
        values = torch.tensor([5.0, 5.0, 5.0, 5.0])

        adv = module.calculate_advantages(rewards, values)

        assert torch.isclose(adv.mean(), torch.tensor(0.0), atol=1e-5)

    def test_calculate_ratio(self, setup_ppo):
        """Test importance sampling ratio."""
        module, _, _, _ = setup_ppo

        new_log_p = torch.tensor([-1.0, -1.0])
        old_log_p = torch.tensor([-1.0, -2.0])

        ratio = module.calculate_ratio(new_log_p, old_log_p)

        assert torch.isclose(ratio[0], torch.tensor(1.0))
        assert torch.isclose(ratio[1], torch.tensor(2.71828), atol=1e-4)

    def test_calculate_actor_loss(self, setup_ppo):
        """Test PPO clipped surrogate loss."""
        module, _, _, _ = setup_ppo
        module.eps_clip = 0.2

        ratio = torch.tensor([1.1])
        advantage = torch.tensor([1.0])
        loss = module.calculate_actor_loss(ratio, advantage)
        assert torch.isclose(loss, torch.tensor(-1.1))

        ratio = torch.tensor([1.5])
        advantage = torch.tensor([1.0])
        loss = module.calculate_actor_loss(ratio, advantage)
        assert torch.isclose(loss, torch.tensor(-1.2))

    def test_configure_optimizers(self, setup_ppo):
        """Test combined optimizer configuration."""
        module, _, _, _ = setup_ppo

        opt = module.configure_optimizers()

        # PPO implementation usually just returns the optimizer directly
        assert isinstance(opt, torch.optim.Adam)

    def test_training_step_dict_input(self, setup_ppo):
        """Test training_step with dict input instead of TensorDict."""
        module, env, policy, critic = setup_ppo

        batch = {"loc": torch.randn(4, 2)}

        module.baseline = MockBaseline()
        module.baseline.unwrap_side_effect = lambda x: (x, None)

        env.reset.side_effect = lambda x: x

        # We want to verify it gets converted. If we receive a dict, PPO converts it to TensorDict
        # Then calls env.reset(TensorDict)
        with patch.object(env, "reset") as mock_reset:
            # Make reset fail with a special error to stop execution
            mock_reset.side_effect = KeyError("Check passed")
            try:
                module.training_step(batch, 0)
            except KeyError as e:
                if str(e) == "'Check passed'":
                    # Check what reset was called with
                    call_args = mock_reset.call_args
                    assert isinstance(call_args[0][0], TensorDict)
                    return
                raise e
