"""
Tests for REINFORCE algorithm implementation.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from logic.src.pipeline.learning.core.reinforce import REINFORCE
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]





class MockBaseline(nn.Module):
    """Mock baseline that is a valid nn.Module."""
    def __init__(self):
        super().__init__()
        self.val = torch.zeros(1) # dummy param
        self.eval_return_value = None

    def eval(self, tensordict, reward, env=None):   # type: ignore[override]
        if self.eval_return_value is not None:
            return self.eval_return_value
        return torch.zeros(reward.shape[0])


class TestREINFORCE:
    """Tests for REINFORCE algorithm."""

    @pytest.fixture
    def setup_reinforce(self):
        """Setup REINFORCE instance with mocks."""
        env = MagicMock()

        # Policy must be nn.Module
        policy = MagicMock(spec=nn.Module)
        # But to be assigned as submodule it needs to BE an nn.Module instance
        # So let's use a simple Linear layer as mock policy
        policy = nn.Linear(1, 1)

        module = REINFORCE(
            env=env,
            policy=policy,
            baseline="rollout",  # Default baseline type
            entropy_weight=0.1,
            max_grad_norm=1.0,
        )
        return module, env, policy

    def test_init(self, setup_reinforce):
        """Test initialization."""
        module, env, policy = setup_reinforce

        assert module.env == env
        assert module.policy == policy
        assert module.entropy_weight == 0.1
        assert module.max_grad_norm == 1.0
        assert module.baseline is not None

    def test_calculate_loss_no_baseline(self, setup_reinforce):
        """Test loss calculation without baseline or entropy."""
        module, _, _ = setup_reinforce
        module.entropy_weight = 0.0

        # Replace baseline with MockBaseline
        module.baseline = MockBaseline()
        module.baseline.eval_return_value = torch.zeros(2)

        # Mock output
        reward = torch.tensor([10.0, 5.0])
        log_likelihood = torch.tensor([-0.5, -1.0])
        out = {
            "reward": reward,
            "log_likelihood": log_likelihood,
        }

        td = TensorDict({}, batch_size=[2])

        # Mock log
        with patch.object(module, "log") as mock_log:
            loss = module.calculate_loss(td, out, batch_idx=0)

            assert isinstance(loss, torch.Tensor)
            assert loss.ndim == 0
            mock_log.assert_called()

    def test_calculate_loss_with_entropy(self, setup_reinforce):
        """Test loss calculation with entropy bonus."""
        module, _, _ = setup_reinforce
        module.entropy_weight = 0.1

        # Mock baseline
        module.baseline = MockBaseline()
        module.baseline.eval_return_value = torch.zeros(2)

        # Mock output with entropy
        reward = torch.tensor([0.0, 0.0])
        log_likelihood = torch.tensor([-1.0, -1.0])
        entropy = torch.tensor([0.5, 0.5])
        out = {
            "reward": reward,
            "log_likelihood": log_likelihood,
            "entropy": entropy
        }

        td = TensorDict({}, batch_size=[2])

        with patch.object(module, "log"):
            loss = module.calculate_loss(td, out, batch_idx=0)
            assert torch.isclose(loss, torch.tensor(-0.05), atol=1e-6)

    def test_calculate_loss_with_baseline_val(self, setup_reinforce):
        """Test loss calculation when baseline value is pre-calculated."""
        module, _, _ = setup_reinforce

        # Set pre-calculated baseline value
        module._current_baseline_val = torch.tensor([5.0, 5.0]) # type: ignore[attr-defined]

        # Mock baseline eval to ensure it's NOT called
        module.baseline = MockBaseline()
        # We can't easily assert not called on real method without wrapper
        # But we can set a flag inside MockBaseline if we wanted, or just trust logic
        # Or wrap eval with MagicMock
        with patch.object(module.baseline, 'eval') as mock_eval:
             out = {
                "reward": torch.tensor([10.0, 10.0]),
                "log_likelihood": torch.tensor([-1.0, -1.0]),
            }
             td = TensorDict({}, batch_size=[2])

             with patch.object(module, "log"):
                module.calculate_loss(td, out, batch_idx=0)
                mock_eval.assert_not_called()

    def test_on_before_optimizer_step(self, setup_reinforce):
        """Test gradient clipping."""
        module, _, policy = setup_reinforce
        module.max_grad_norm = 1.0

        # Policy is real nn.Module, so parameters exist
        optimizer = MagicMock()

        with patch("torch.nn.utils.clip_grad_norm_") as mock_clip, patch.object(module, "log"):
            mock_clip.return_value = torch.tensor(0.5)
            module.on_before_optimizer_step(optimizer)

            mock_clip.assert_called_once()
            args, _ = mock_clip.call_args
            assert args[1] == 1.0

    def test_on_before_optimizer_step_no_clip(self, setup_reinforce):
        """Test no gradient clipping when max_grad_norm is 0."""
        module, _, _ = setup_reinforce
        module.max_grad_norm = 0.0

        optimizer = MagicMock()

        with patch("torch.nn.utils.clip_grad_norm_") as mock_clip:
            module.on_before_optimizer_step(optimizer)
            mock_clip.assert_not_called()
