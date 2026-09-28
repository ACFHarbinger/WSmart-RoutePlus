"""Tests for RL4COLitModule."""

import os
from unittest.mock import MagicMock

import pytest
import torch
from logic.src.pipeline.rl.common.base import RL4COLitModule

pytestmark = [pytest.mark.unit, pytest.mark.fast]



class ConcreteLitModule(RL4COLitModule):
    def calculate_loss(self, td, out, batch_idx, env=None):
        return torch.tensor(0.0)


@pytest.fixture
def lit_setup():
    mock_env = MagicMock()
    mock_env.generator = MagicMock()
    # Ensure .to returns the mock itself
    mock_env.generator.to.return_value = mock_env.generator
    mock_policy = MagicMock(spec=torch.nn.Module)
    mock_policy.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]

    return ConcreteLitModule(env=mock_env, policy=mock_policy, baseline="rollout", batch_size=2)


def test_lit_module_init(lit_setup):
    """Verify initialization of RL4COLitModule."""
    assert lit_setup.batch_size == 2
    assert lit_setup.optimizer_name == "adam"


def test_lit_module_save_weights(lit_setup, tmp_path):
    """Verify weights saving logic."""
    path = str(tmp_path / "model.pt")
    lit_setup.save_weights(path)

    assert os.path.exists(path)
    assert os.path.exists(str(tmp_path / "args.json"))

    # Check loading to verify format
    data = torch.load(path, weights_only=False)
    assert "state_dict" in data
    assert "hparams" in data


def test_lit_module_configure_optimizers(lit_setup):
    """Verify optimizer configuration."""
    # Adam
    opt = lit_setup.configure_optimizers()
    assert isinstance(opt, torch.optim.Adam)

    # AdamW
    lit_setup.optimizer_name = "adamw"
    opt = lit_setup.configure_optimizers()
    assert isinstance(opt, torch.optim.AdamW)

    # RMSprop
    lit_setup.optimizer_name = "rmsprop"
    opt = lit_setup.configure_optimizers()
    assert isinstance(opt, torch.optim.RMSprop)


def test_lit_module_configure_schedulers(lit_setup):
    """Verify scheduler configuration."""
    lit_setup.lr_scheduler_name = "step"
    lit_setup.lr_scheduler_kwargs = {"step_size": 1, "gamma": 0.1}

    config = lit_setup.configure_optimizers()
    assert "optimizer" in config
    assert "lr_scheduler" in config
    assert isinstance(config["lr_scheduler"], torch.optim.lr_scheduler.StepLR)


def test_lit_module_setup_fit(lit_setup):
    """Verify setup phase for training."""
    lit_setup.env.graph.num_samples = 4

    # Mock generator to return TensorDict
    from tensordict import TensorDict



    mock_data_train = TensorDict({"loc": torch.randn(4, 5, 2)}, batch_size=[4])
    mock_data_val = TensorDict({"loc": torch.randn(2, 5, 2)}, batch_size=[2])

    # Training, fixed reporting validation, then the independent rollout pool.
    mock_data_comparison = TensorDict({"loc": torch.randn(10, 5, 2)}, batch_size=[10])
    lit_setup.env.generator.side_effect = [mock_data_train, mock_data_val, mock_data_comparison]

    lit_setup.setup("fit")
    assert lit_setup.train_dataset is not None
    assert len(lit_setup.train_dataset) == 4
    assert lit_setup.val_dataset is not None
    assert len(lit_setup.val_dataset) == 2
