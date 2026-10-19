

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from logic.src.pipeline.learning.core.a2c import A2C
from logic.src.pipeline.learning.envs.base import RL4COEnvBase
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]





class MockPolicy(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.param = nn.Parameter(torch.randn(1))

    def forward(self, td, env=None, strategy="sampling"):
        batch_size = td.batch_size[0]
        return {
            "reward": torch.ones(batch_size, device=td.device),
            "log_likelihood": torch.zeros(batch_size, device=td.device),
            "entropy": torch.zeros(batch_size, device=td.device),
        }


class MockCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.randn(1))

    def forward(self, td):
        batch_size = td.batch_size[0]
        return torch.zeros(batch_size, 1, device=td.device)


@pytest.fixture
def mock_env():
    env = MagicMock(spec=RL4COEnvBase)
    env.device = torch.device("cpu")
    env.batch_size = torch.Size([2])
    return env


def test_a2c_init(mock_env):
    policy = MockPolicy()
    critic = MockCritic()

    a2c = A2C(env=mock_env, policy=policy, critic=critic, actor_lr=1e-4, critic_lr=1e-3)

    assert a2c.policy == policy
    assert a2c.critic == critic
    assert a2c.actor_lr == 1e-4
    assert a2c.critic_lr == 1e-3


def test_a2c_calculate_loss(mock_env):
    policy = MockPolicy()
    critic = MockCritic()
    a2c = A2C(env=mock_env, policy=policy, critic=critic)

    td = TensorDict({"locs": torch.randn(2, 5, 2)}, batch_size=[2])
    out = {
        "reward": torch.tensor([1.0, 2.0]),
        "log_likelihood": torch.tensor([-0.5, -0.1]),
        "entropy": torch.tensor([0.1, 0.2]),
    }

    # Mock critic to return constant value
    a2c.critic.forward = MagicMock(return_value=torch.tensor([[0.5], [0.5]]))

    loss = a2c.calculate_loss(td, out, 0)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_a2c_configure_optimizers(mock_env):
    policy = MockPolicy()
    critic = MockCritic()
    a2c = A2C(env=mock_env, policy=policy, critic=critic)

    optimizers = a2c.configure_optimizers()

    assert len(optimizers) == 2
    assert isinstance(optimizers[0], torch.optim.Adam)
    assert isinstance(optimizers[1], torch.optim.Adam)

    # Check that they optimize different parameters
    assert optimizers[0].param_groups[0]["params"][0] is policy.param
    assert optimizers[1].param_groups[0]["params"][0] is critic.param
