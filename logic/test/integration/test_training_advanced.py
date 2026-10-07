

import os
import shutil

import pytest
from logic.src.envs.routing.ptp import PTPEnv
from logic.src.models.common.critic_network.policy import CriticNetwork as CriticNetworkPolicy
from logic.src.models.core.attention_model.policy import AttentionModelPolicy
from logic.src.models.subnets.factories.attention import AttentionComponentFactory
from logic.src.pipeline.rl.core.dr_grpo import DRGRPO
from logic.src.pipeline.rl.core.gdpo import GDPO
from logic.src.pipeline.rl.core.reinforce import REINFORCE
from pytorch_lightning import Trainer

pytestmark = [pytest.mark.integration, pytest.mark.slow]





@pytest.fixture
def clean_logs():
    if os.path.exists("lightning_logs"):
        shutil.rmtree("lightning_logs")
    yield
    if os.path.exists("lightning_logs"):
        shutil.rmtree("lightning_logs")


@pytest.mark.integration
def test_dr_grpo_training_loop(clean_logs):
    """Test the DR-GRPO algorithm training loop."""
    env = PTPEnv(num_loc=20)
    # Monkeypatch NAME for compatibility with CriticNetwork which expects .NAME (uppercase)
    # PTPEnv only has .name (lowercase)
    env.NAME = env.name

    policy = AttentionModelPolicy(
        env_name="ptp",
        embed_dim=128,
        hidden_dim=128,
        n_encode_layers=2,
        n_heads=8,
    )

    factory = AttentionComponentFactory()
    critic = CriticNetworkPolicy(
        env_name="ptp", component_factory=factory, embed_dim=128, hidden_dim=128, n_layers=2, n_sublayers=1, n_heads=8
    )

    # DR-GRPO configuration
    module = DRGRPO(
        env=env,
        policy=policy,
        critic=critic,
        optimizer="adam",
        lr=1e-4,
        max_grad_norm=1.0,
        env_name="ptp",
    )

    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(module)
    assert trainer.global_step > 0


@pytest.mark.integration
def test_gdpo_training_loop(clean_logs):
    """Test the GDPO algorithm training loop."""
    env = PTPEnv(num_loc=20)
    env.NAME = env.name

    policy = AttentionModelPolicy(
        env_name="ptp",
        embed_dim=128,
        hidden_dim=128,
        n_encode_layers=2,
    )

    factory = AttentionComponentFactory()
    critic = CriticNetworkPolicy(
        env_name="ptp", component_factory=factory, embed_dim=128, hidden_dim=128, n_layers=2, n_sublayers=1, n_heads=8
    )

    # GDPO configuration with objective keys
    module = GDPO(
        env=env,
        policy=policy,
        critic=critic,
        gdpo_objective_keys=["reward"],
        gdpo_objective_weights=[1.0],
        optimizer="adam",
        lr=1e-4,
        env_name="ptp",
    )

    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=0,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(module)
    assert trainer.global_step > 0


@pytest.mark.integration
def test_training_resume(tmp_path):
    """Test resuming training from a checkpoint."""
    env = PTPEnv(num_loc=10)
    policy = AttentionModelPolicy(env_name="ptp", embed_dim=64, hidden_dim=64, n_encode_layers=1)
    module = REINFORCE(
        env=env,
        policy=policy,
        optimizer="adam",
        lr=1e-3,
    )

    checkpoint_dir = tmp_path / "checkpoints"

    # Train 1 epoch
    trainer1 = Trainer(
        default_root_dir=str(checkpoint_dir),
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=0,
        accelerator="cpu",
        enable_checkpointing=True,
    )
    trainer1.fit(module)

    # Find checkpoint
    ckpt_path = list((checkpoint_dir / "lightning_logs" / "version_0" / "checkpoints").glob("*.ckpt"))[0]

    # Resume
    trainer2 = Trainer(
        default_root_dir=str(checkpoint_dir),
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=0,
        accelerator="cpu",
        enable_checkpointing=True,
    )

    # We load state dict manually to verify or pass ckpt_path to fit
    trainer2.fit(module, ckpt_path=str(ckpt_path))

    assert trainer2.current_epoch == 2
    assert trainer2.global_step > trainer1.global_step
