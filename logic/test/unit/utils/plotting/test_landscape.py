"""Unit tests for landscape.py."""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch
from logic.src.utils.plotting.landscape import imitation_loss_fn, plot_loss_landscape, rl_loss_fn
from omegaconf import OmegaConf

pytestmark = [pytest.mark.unit, pytest.mark.fast]


@pytest.fixture
def mock_model():
    """Create a mock model that behaves like an nn.Module with necessary methods."""
    # Use spec to avoid having every attribute by default
    model = MagicMock(spec=["eval", "parameters", "cost_weights", "set_strategy", "model", "modules"])

    # Mock parameters to get a device
    param = MagicMock()
    param.device = torch.device("cpu")
    model.parameters.return_value = iter([param])
    model.cost_weights = {"w1": torch.tensor([1.0])}

    # By default, don't have modules or nested model
    del model.modules
    del model.model

    # Mock return value for __call__
    # (cost, log_p, entropy, pi, mask)
    res = (
        torch.tensor([10.0]),  # cost
        torch.tensor([-2.0]),  # log_p
        torch.tensor([0.5]),  # entropy
        torch.tensor([[1, 2]]),  # pi
        torch.tensor([0]),  # mask
    )
    model.return_value = res
    return model


def test_imitation_loss_fn(mock_model):
    """Test imitation loss calculation."""
    x_batch = {"data": torch.zeros(1)}
    pi_target = torch.tensor([[1, 2]])

    loss = imitation_loss_fn(mock_model, x_batch, pi_target)

    assert isinstance(loss, float)
    assert loss == 2.0  # -(-2.0)
    mock_model.eval.assert_called()


def test_rl_loss_fn(mock_model):
    """Test RL loss calculation."""
    x_batch = {"data": torch.zeros(1)}

    loss = rl_loss_fn(mock_model, x_batch)

    assert isinstance(loss, float)
    assert loss == 10.0
    mock_model.set_strategy.assert_called_with("greedy")


@patch("logic.src.utils.plotting.landscape.get_batch")
@patch("logic.src.policies.vector.local_search.vectorized_two_opt")
@patch("logic.src.utils.plotting.landscape.loss_landscapes.random_plane")
@patch("logic.src.utils.plotting.landscape.plt")
def test_plot_loss_landscape(mock_plt, mock_random_plane, mock_two_opt, mock_get_batch, mock_model, tmp_path):
    """Test the full landscape plotting pipeline with mocks."""
    opts = OmegaConf.create({"device": "cpu", "model": {"temporal_horizon": 0}})
    output_dir = str(tmp_path / "plots")

    # Setup mocks
    mock_get_batch.return_value = {"dist": torch.zeros((1, 5, 5))}
    mock_two_opt.return_value = torch.zeros((16, 51), dtype=torch.long)
    mock_random_plane.return_value = [[1.0, 2.0], [3.0, 4.0]]  # Dummy grid of losses

    plot_loss_landscape(mock_model, opts, output_dir, epoch=1, resolution=2)

    # Verify directory creation
    assert os.path.exists(output_dir)

    # Verify calls
    assert mock_get_batch.called
    assert mock_random_plane.called
    assert mock_plt.savefig.called

    # Verify files? Since we mock plt, it won't actually save unless we are careful.
    # But checking if savefig was called is enough for unit test.


def test_imitation_loss_fn_with_wrapped_model(mock_model):
    """Test imitation loss when model is wrapped."""
    wrapper = MagicMock()
    wrapper.modules = [mock_model]

    x_batch = {"data": torch.zeros(1)}
    pi_target = torch.tensor([[1, 2]])

    loss = imitation_loss_fn(wrapper, x_batch, pi_target)
    assert loss == 2.0


def test_rl_loss_fn_exception_handling(mock_model, tmp_path, mocker):
    """Test that plot_loss_landscape handles exceptions during computation."""
    opts = OmegaConf.create({"device": "cpu", "model": {"temporal_horizon": 0}})
    output_dir = str(tmp_path / "plots_fail")

    # Mock random_plane to raise exception
    mocker.patch("loss_landscapes.random_plane", side_effect=Exception("Test Error"))
    mocker.patch("logic.src.utils.plotting.landscape.get_batch", return_value={"dist": torch.zeros((1, 5, 5))})
    mocker.patch(
        "logic.src.policies.vector.local_search.vectorized_two_opt",
        return_value=torch.zeros((16, 51), dtype=torch.long),
    )

    # Should not raise exception
    plot_loss_landscape(mock_model, opts, output_dir)
    assert os.path.exists(output_dir)
