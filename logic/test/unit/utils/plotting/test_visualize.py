"""
Tests for visualization utilities and plotting functions.
Merges functionality from:
- test_visualize.py
- test_visualize_coverage.py
- test_visualize_epoch_coverage.py
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from logic.src.utils.plotting import (
    get_batch,
    imitation_loss_fn,
    log_weight_distributions,
    main,
    plot_attention_heatmaps,
    plot_linechart,
    plot_tsp,
    plot_vehicle_routes,
    plot_weight_trajectories,
    project_node_embeddings,
    rl_loss_fn,
    visualize_epoch,
)
from omegaconf import OmegaConf

pytestmark = [pytest.mark.unit, pytest.mark.fast]


# ============================================================================
# Core Visualization Utils Tests
# ============================================================================


class TestVisualizeUtils(unittest.TestCase):
    """Test suite for core visualization utilities."""

    def test_get_batch(self):
        """Test generation of sample data batches for visualization."""

        device = torch.device("cpu")
        batch = get_batch(device, size=10, batch_size=2)
        self.assertTrue("depot" in batch)
        self.assertEqual(batch["depot"].shape, (2, 2))

    @patch("logic.src.utils.plotting.embeddings.plt")
    @patch("logic.src.utils.plotting.embeddings.os.listdir")
    @patch("logic.src.utils.plotting.embeddings.torch.load")
    def test_plot_weight_trajectories(self, mock_load, mock_listdir, mock_plt):
        """Test plotting of model weight changes over epochs."""

        mock_listdir.return_value = ["epoch-1.pt", "epoch-2.pt"]
        # Mock checkpoint
        mock_load.side_effect = [
            {"model": {"l1.weight": torch.randn(10, 10)}},
            {"model": {"l1.weight": torch.randn(10, 10) + 1.0}},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_out = os.path.join(tmp_dir, "out.png")
            plot_weight_trajectories(tmp_dir, dummy_out)

        mock_plt.savefig.assert_called()

    @patch("logic.src.utils.plotting.embeddings.SummaryWriter")
    def test_log_weight_distributions(self, mock_writer_cls):
        """Test logging of model weight histograms to TensorBoard."""

        model = MagicMock()
        model.named_parameters.return_value = [("p1", torch.randn(5))]
        mock_writer = mock_writer_cls.return_value

        log_weight_distributions(model, 1, "logs")
        mock_writer.add_histogram.assert_called()
        mock_writer.close.assert_called()

    @patch("logic.src.utils.plotting.embeddings.SummaryWriter")
    def test_project_node_embeddings(self, mock_writer_cls):
        """Test projection and logging of node embeddings to TensorBoard."""

        model = MagicMock()
        model._get_initial_embeddings.return_value = torch.randn(2, 5, 10)  # B, N, D
        model.embedder.return_value = torch.randn(2, 5, 10)
        batch = {"edges": None, "dist": None}
        mock_writer = mock_writer_cls.return_value

        project_node_embeddings(model, batch, "logs")
        mock_writer.add_embedding.assert_called()

    @patch("logic.src.utils.plotting.heatmaps.plt")
    @patch("logic.src.utils.plotting.heatmaps.sns")
    def test_plot_attention_heatmaps(self, mock_sns, mock_plt):
        """Test generation of attention heatmaps for model layers."""

        model = MagicMock()
        # Create a mock layer
        layer = MagicMock()
        # Create a mock for the module (mha)
        mha = MagicMock()
        # Mock W_query
        w_query = MagicMock()
        # Set up the chain so .numpy() returns a real array
        w_query.weight.data.cpu.return_value.numpy.return_value = np.zeros((8, 10, 10))

        # Attach W_query to mha
        mha.W_query = w_query
        # Mock others to avoid auto-creation issues or just point to same w_query
        mha.W_key = w_query
        mha.W_val = w_query

        # Attach mha to layer
        layer.att.module = mha

        model.embedder.layers = [layer]

        with tempfile.TemporaryDirectory() as tmp_dir:
            plot_attention_heatmaps(model, tmp_dir)

        mock_sns.heatmap.assert_called()
        mock_plt.savefig.assert_called()


# ============================================================================
# Plot Utilities Tests
# ============================================================================


@patch("logic.src.utils.plotting.charts.plt")
class TestPlotUtils(unittest.TestCase):
    """Test suite for general plotting utilities."""

    def test_plot_linechart(self, mock_plt):
        """Test line chart generation for evaluation metrics."""

        msg = np.zeros((1, 2, 6))  # 1 policy, 2 points, 6 metrics
        plot_linechart("out.png", msg, mock_plt.plot, ["pol1"])
        mock_plt.savefig.assert_called()

    def test_plot_tsp(self, mock_plt):
        """Test plotting of TSP tours."""

        xy = np.random.rand(5, 2)
        tour = np.array([0, 1, 2, 3, 4, 0])
        ax = MagicMock()
        plot_tsp(xy, tour, ax)
        ax.plot.assert_called()
        ax.scatter.assert_called()

    def test_plot_vehicle_routes(self, mock_plt):
        """Test plotting of multi-vehicle routing solutions."""

        data = {
            "depot": torch.tensor([0.5, 0.5]),
            "loc": torch.rand(5, 2),
            "waste": torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1]),
        }
        route = torch.tensor([0, 1, 2, 0, 3, 4, 5, 0])
        ax = MagicMock()
        plot_vehicle_routes(data, route, ax)
        ax.plot.assert_called()
        ax.add_collection.assert_not_called()


# ============================================================================
# Loss Function Tests (from coverage)
# ============================================================================


class TestVisualizeLossFunctions:
    """Class for visualization auxiliary functions (loss)."""

    def test_imitation_loss_fn(self):
        """Test imitation loss function for landscape plotting."""
        model = MagicMock()
        del model.modules
        del model.model

        # model returns (cost, log_likelihood, ...)
        # we need log_likelihood to have mean -1.5 -> loss = 1.5
        ll = torch.tensor([-1.5, -1.5])
        model.return_value = (None, ll)

        x_batch = {"coords": torch.rand(2, 5, 2)}
        pi_target = torch.randint(0, 5, (2, 5))

        # Ensure parameter device logic works
        p = MagicMock()
        p.device = torch.device("cpu")
        model.parameters.return_value = iter([p])

        loss = imitation_loss_fn(model, x_batch, pi_target)
        assert loss == 1.5

    def test_rl_loss_fn(self):
        """Test RL loss function for landscape plotting."""
        model = MagicMock()
        # Prevent auto-creation of attributes causing unwrapping logic to dive in
        del model.modules
        del model.model

        # Mock parameter to have device
        param = MagicMock()
        param.device = torch.device("cpu")
        model.parameters.return_value = iter([param])

        # Returns cost, _, _, _, _ (5 values)
        cost = torch.tensor([10.0, 20.0])
        model.return_value = (cost, None, None, None, None)

        x_batch = {"coords": torch.rand(2, 5, 2)}

        loss = rl_loss_fn(model, x_batch)
        assert loss == 15.0  # Mean of 10 and 20


# ============================================================================
# Visualize Epoch Tests (from coverage)
# ============================================================================


class TestVisualizeEpochCoverage:
    """Class for visualize_epoch tests."""

    @patch("logic.src.utils.plotting.plot_weight_trajectories")
    @patch("logic.src.utils.plotting.log_weight_distributions")
    @patch("logic.src.utils.plotting.plot_attention_heatmaps")
    @patch("logic.src.utils.plotting.plot_loss_landscape")
    @patch("logic.src.utils.plotting.project_node_embeddings")
    @patch("logic.src.utils.plotting.get_batch")
    def test_visualize_epoch_all_modes(self, mock_batch, mock_embed, mock_loss, mock_att, mock_dist, mock_traj):
        """Test visualize_epoch with all modes enabled."""
        model = MagicMock()
        # Ensure parameters() returns a fresh iterator each time
        p_mock = MagicMock()
        p_mock.device = torch.device("cpu")
        model.parameters.side_effect = lambda: iter([p_mock])

        problem = MagicMock()
        opts = OmegaConf.create(
            {
                "rl": {"viz_modes": ["trajectory", "distributions", "embeddings", "heatmaps", "logit_lens", "loss"]},
                "log_dir": "logs",
                "run_name": "test_run",
                "graph_size": 20,
                "save_dir": "checkpoints",
                "model": {"name": "am", "temporal_horizon": 0},
                "train": {"graph": {"num_loc": 20}},
            }
        )

        # mock plot_logit_lens as it is also called
        with patch("logic.src.utils.plotting.plot_logit_lens") as mock_lens:
            # Setup model mock for logit lens
            model.decoder._get_log_p.return_value = (torch.randn(2, 1, 20), None)

            visualize_epoch(model, problem, opts, epoch=1, tb_logger=MagicMock())

            assert mock_dist.called
            assert mock_embed.called
            assert mock_att.called
            assert mock_lens.called
            assert mock_loss.called
            assert mock_traj.called

    def test_visualize_epoch_no_modes(self):
        """Test visualize_epoch with no modes."""
        model = MagicMock()
        opts = OmegaConf.create(
            {"rl": {"viz_modes": []}, "model": {"name": "am", "temporal_horizon": 0}, "train": {"graph": {}}}
        )
        visualize_epoch(model, None, opts, 1)  # Should return immediately

    @patch(
        "sys.argv",
        [
            "prog",
            "--mode",
            "distributions",
            "--model_path",
            "model.pt",
            "--log_dir",
            "logs",
        ],
    )
    @patch("logic.src.utils.plotting.load_model_instance")
    @patch("logic.src.utils.plotting.log_weight_distributions")
    def test_main_distributions(self, mock_log, mock_load):
        """Test main function with distributions mode."""
        mock_load.return_value = MagicMock()
        main()
        assert mock_log.called

    @patch("sys.argv", ["prog", "--mode", "trajectory", "--checkpoint_dir", "ckpt_dir"])
    @patch("logic.src.utils.plotting.plot_weight_trajectories")
    def test_main_trajectory(self, mock_plot):
        """Test main function with trajectory mode."""
        main()
        assert mock_plot.called
