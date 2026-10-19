"""
Tests for PyTorch Lightning RL modules (REINFORCE, PPO, GRPO).
"""

import pytest
import torch
from logic.src.pipeline.learning.core.reinforce import REINFORCE
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]





class TestRLCore:
    """Tests for core Lightning RL modules."""

    @pytest.fixture
    def mock_env(self, mocker):
        """Mock environment."""
        env = mocker.MagicMock()
        env.reset.side_effect = lambda x: x
        return env

    @pytest.fixture
    def mock_policy(self, am_setup):
        """Use am_setup as policy."""
        return am_setup

    def test_reinforce_init(self, mock_policy, mock_env):
        """Test REINFORCE initialization."""
        model = REINFORCE(policy=mock_policy, env=mock_env, baseline="none", entropy_weight=0.01)
        assert model.entropy_weight == 0.01
        assert model.policy == mock_policy
        assert model.env == mock_env

    def test_reinforce_calculate_loss(self, mock_policy, mock_env):
        """Test REINFORCE loss calculation."""
        model = REINFORCE(
            policy=mock_policy,
            env=mock_env,
            baseline="none",
        )

        td = TensorDict({"loc": torch.randn(2, 5, 2)}, batch_size=[2])
        out = {
            "reward": torch.tensor([1.0, 2.0]),
            "log_likelihood": torch.tensor([-0.5, -0.6], requires_grad=True),
        }

        loss = model.calculate_loss(td, out, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.grad_fn is not None  # Should be backpropagatable
