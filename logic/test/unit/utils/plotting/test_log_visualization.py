from unittest.mock import patch

import pandas as pd
import pytest
from logic.src.tracking.logging.structured_logging import log_training
from logic.src.utils.plotting.log_visualization import plot_training_logs
from omegaconf import OmegaConf

pytestmark = [pytest.mark.unit, pytest.mark.fast]


class TestLogVisualization:
    """Class for log visualization tests."""

    @patch("logic.src.tracking.logging.structured_logging.wandb")
    @patch("logic.src.utils.plotting.log_visualization.wandb")
    @patch("logic.src.utils.plotting.log_visualization.plt")
    def test_plot_training_logs(self, mock_plt, mock_wandb, mock_struct_wandb):
        """Test plot_training_logs with mocked components."""
        opts = OmegaConf.create(
            {
                "train": {
                    "checkpoints_dir": "checkpoints",
                    "save_dir": "checkpoints/run1",
                    "log_dir": "logs",
                    "train_time": False,
                },
                "rl": {"wandb_mode": "online"},
            }
        )
        columns = pd.MultiIndex.from_tuples([("loss", "mean"), ("loss", "std"), ("loss", "max"), ("loss", "min")])
        table_df = pd.DataFrame([[1.0, 0.1, 1.2, 0.8]], columns=columns)
        loss_keys = ["loss"]
        with patch("os.makedirs"), patch("pandas.DataFrame.to_parquet"):
            loss_keys, xname, x_values, swapped_df, output_dir, wandb_mode = log_training(
                loss_keys, table_df, opts, plot_logs=False
            )
            plot_training_logs(loss_keys, xname, x_values, swapped_df, output_dir, wandb_mode)  # pyrefly: ignore [bad-argument-type]
        assert mock_wandb.log.called
