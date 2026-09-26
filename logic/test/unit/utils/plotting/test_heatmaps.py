"""Unit tests for heatmaps.py."""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch
from logic.src.utils.plotting.heatmaps import plot_attention_heatmaps, plot_logit_lens

pytestmark = [pytest.mark.unit, pytest.mark.fast]


@pytest.fixture
def mock_model():
    model = MagicMock()
    # Mock parameters for device
    param = MagicMock()
    param.device = torch.device("cpu")
    model.parameters.return_value = iter([param])

    # Mock Embedder
    layer = MagicMock()
    # att is SkipConnection, att.module is MultiHeadAttention
    mha = MagicMock()
    mha.W_query.weight.data = torch.randn(2, 4, 4)  # 3D weight
    mha.W_key.weight.data = torch.randn(4, 4)  # 2D weight
    mha.W_val.weight.data = torch.randn(4, 4)  # 2D weight
    layer.att.module = mha
    layer.return_value = torch.randn(1, 5, 4)  # (Batch, Nodes, Dim)

    model.embedder.layers = [layer]
    model.embedder.dropout.side_effect = lambda x: x

    # Mock Decoder
    model.decoder._precompute.return_value = "fixed"
    model.decoder._get_log_p.return_value = (torch.zeros(1, 1, 5), None)

    # Mock Problem
    model.problem.make_state.return_value = "state"

    model._get_initial_embeddings.return_value = torch.randn(1, 5, 4)

    return model


@patch("logic.src.utils.plotting.heatmaps.plt")
@patch("logic.src.utils.plotting.heatmaps.sns")
def test_plot_attention_heatmaps(mock_sns, mock_plt, mock_model, tmp_path):
    """Test attention weight heatmap plotting."""
    output_dir = str(tmp_path / "heatmaps")
    plot_attention_heatmaps(mock_model, output_dir, epoch=1)

    assert os.path.exists(output_dir)
    assert mock_sns.heatmap.called
    assert mock_plt.savefig.called


@patch("logic.src.utils.plotting.heatmaps.plt")
@patch("logic.src.utils.plotting.heatmaps.sns")
def test_plot_logit_lens(mock_sns, mock_plt, mock_model, tmp_path):
    """Test logit lens probability distribution plotting."""
    output_file = str(tmp_path / "logit_lens.png")
    x_batch = {"data": torch.zeros(1), "dist": torch.zeros(1, 5, 5)}

    plot_logit_lens(mock_model, x_batch, output_file, epoch=1)

    assert os.path.exists(os.path.dirname(output_file))
    assert mock_sns.heatmap.called
    assert mock_plt.savefig.called
    # Verify model calls
    assert mock_model.decoder._get_log_p.called
    assert mock_model.problem.make_state.called
