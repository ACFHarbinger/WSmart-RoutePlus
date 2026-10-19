"""
Tests for RL4COLitModule base class.
"""

from typing import Any, cast

import pytest
import torch
from logic.src.pipeline.learning.common.base import RL4COLitModule
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]





class MockLitModule(RL4COLitModule):
    """Mock implementation of RL4COLitModule for testing."""

    def calculate_loss(self, td, out, batch_idx, env=None):
        return (out["reward"] * out["log_likelihood"]).sum()


class TestRL4COLitModule:
    """Tests for RL4COLitModule logic."""

    @pytest.fixture
    def setup_data(self, am_setup, mocker):
        """Setup mock environment and policy."""
        env = mocker.MagicMock()
        env.reset.side_effect = lambda x: x
        policy = am_setup
        return env, policy

    def test_init(self, setup_data):
        """Test initialization and hyperparameter saving."""
        env, policy = setup_data
        model = MockLitModule(env=env, policy=policy, baseline="none", batch_size=32)

        assert model.env == env
        assert model.policy == policy
        assert model.batch_size == 32
        print(f"DEBUG: hparams keys: {model.hparams.keys()}")
        assert "batch_size" in model.hparams
        assert model.hparams["batch_size"] == 32

    def test_shared_step(self, setup_data, mocker):
        """Test shared_step execution flow."""
        env, policy = setup_data
        model = MockLitModule(env=env, policy=policy, baseline="none")

        # Mock policy output
        mock_out = {"reward": torch.tensor([1.0, 2.0]), "log_likelihood": torch.tensor([-0.5, -0.6])}
        mocker.patch.object(model.policy, "forward", return_value=mock_out)

        batch = TensorDict({"loc": torch.randn(2, 5, 2)}, batch_size=[2])

        # We need to mock self.log because it requires a trainer
        mocker.patch.object(model, "log")

        out = model.shared_step(batch, 0, phase="train")

        assert "reward" in out
        assert "loss" in out
        assert cast(Any, model.log).call_count >= 1

    def test_configure_optimizers(self, setup_data):
        """Test optimizer configuration."""
        env, policy = setup_data

        # Adam
        model = MockLitModule(env=env, policy=policy, optimizer="adam", optimizer_kwargs={"lr": 0.01})
        opt = model.configure_optimizers()
        assert isinstance(opt, torch.optim.Adam)
        assert opt.param_groups[0]["lr"] == 0.01

        # AdamW with scheduler
        model = MockLitModule(
            env=env, policy=policy, optimizer="adamw", lr_scheduler="cosine", lr_scheduler_kwargs={"T_max": 10}
        )
        config = model.configure_optimizers()
        assert isinstance(config["optimizer"], torch.optim.AdamW)
        assert isinstance(config["lr_scheduler"], torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_setup_fit(self, setup_data, mocker):
        """Test dataset pre-generation in setup('fit')."""
        env, policy = setup_data
        model = MockLitModule(env=env, policy=policy)

        # Mock generator
        mock_gen = mocker.MagicMock()

        def gen_side_effect(batch_size=10):
            return TensorDict({"data": torch.randn(batch_size, 2)}, batch_size=[batch_size])

        mock_gen.side_effect = gen_side_effect
        mock_gen.to.return_value = mock_gen
        env.generator = mock_gen

        model.setup(stage="fit")

        assert len(cast(Any, model.train_dataset)) == 10
        assert model.val_dataset is not None

    def test_dataloaders(self, setup_data, mocker):
        """Test dataloader creation."""
        env, policy = setup_data
        model = MockLitModule(env=env, policy=policy, batch_size=2)

        model.train_dataset = [1, 2, 3, 4]
        model.val_dataset = [1, 2]

        train_loader = model.train_dataloader()
        assert train_loader.batch_size == 2

        val_loader = model.val_dataloader()
        assert val_loader.batch_size == 2
