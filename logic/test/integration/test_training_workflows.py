"""
Integration tests for training workflows.
"""

import pytest
import torch
from logic.src.envs import PTPEnv
from logic.src.pipeline.rl.core import A2C, PPO, REINFORCE
from logic.src.policies.vector import AttentionModelPolicy
from pytorch_lightning import Trainer

pytestmark = [pytest.mark.integration, pytest.mark.slow]





class SimpleCritic(torch.nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.value = torch.nn.Linear(embed_dim, 1)
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, td):
        # Return scalar value per batch
        batch_size = td.batch_size[0]
        return torch.randn(batch_size, device=td.device)


@pytest.mark.integration
def test_reinforce_training_loop():
    """Test REINFORCE training loop for a few steps."""
    # Initialize env with batch size matching training
    env = PTPEnv(num_loc=10, batch_size=[2])
    policy = AttentionModelPolicy(
        env_name="ptp",
        embed_dim=128,
        hidden_dim=128,
        n_encode_layers=2,
        n_heads=8,
    )

    # REINFORCE with rollout baseline
    module = REINFORCE(
        env=env,
        policy=policy,
        baseline="rollout",
        optimizer_kwargs={"lr": 1e-4},
        batch_size=2,
    )

    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(module)
    assert trainer.global_step > 0


@pytest.mark.integration
def test_ppo_training_loop():
    """Test PPO training loop for a few steps."""
    env = PTPEnv(num_loc=10, batch_size=[1], check_env_specs=False)
    policy = AttentionModelPolicy(
        env_name="ptp",
        embed_dim=128,
        hidden_dim=128,
        n_encode_layers=2,
        n_heads=8,
    )
    critic = SimpleCritic(embed_dim=128)

    module = PPO(
        env=env,
        policy=policy,
        critic=critic,
        optimizer_kwargs={"lr": 1e-4},
        batch_size=1,
    )

    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(module)
    assert trainer.global_step > 0


@pytest.mark.integration
def test_a2c_training_loop():
    """Test A2C training loop for a few steps."""
    env = PTPEnv(num_loc=10, batch_size=[2], check_env_specs=False)
    policy = AttentionModelPolicy(
        env_name="ptp",
        embed_dim=128,
        hidden_dim=128,
    )
    # A2C will create its own critic if not provided

    module = A2C(
        env=env,
        policy=policy,
        batch_size=2,
    )

    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=0,
        accelerator="cpu",
        logger=False,
    )

    trainer.fit(module)
    assert trainer.global_step > 0


@pytest.mark.integration
def test_ppo_multistart_training_loop():
    """Test PPO with multistart decoding."""
    env = PTPEnv(num_loc=10, batch_size=[2], check_env_specs=False)
    policy = AttentionModelPolicy(
        env_name="ptp",
        embed_dim=128,
        hidden_dim=128,
    )
    critic = SimpleCritic(embed_dim=128)

    # Enable multistart in PPO kwargs via policy/strategy config if supported
    # Or just test that passing decode_kwargs works
    module = PPO(
        env=env,
        policy=policy,
        critic=critic,
        batch_size=2,
        decode_kwargs={"multistart": True, "num_starts": 5, "select_best": True},
    )

    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=1,
        accelerator="cpu",
        logger=False,
    )

    trainer.fit(module)
    assert trainer.global_step > 0
