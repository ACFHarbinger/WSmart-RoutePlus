from typing import Any, cast
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch
from logic.src.tracking.logging.modules.analysis import output_stats, runs_per_policy
from logic.src.tracking.logging.modules.metrics import log_epoch, log_values
from logic.src.tracking.logging.modules.storage import log_to_json, log_to_json2
from logic.src.tracking.logging.structured_logging import log_training
from omegaconf import OmegaConf

pytestmark = [pytest.mark.unit, pytest.mark.fast]


class TestLogUtils:
    """Class for log_utils tests."""

    @patch("wandb.log")
    def test_log_epoch(self, mock_wandb):
        """Test log_epoch with mocked wandb."""
        opts = OmegaConf.create({"rl": {"wandb_mode": "online"}, "train_time": False})
        x_tup = ("epoch", 1)
        loss_keys = ["loss"]
        epoch_loss = {"loss": [torch.tensor([1.0]), torch.tensor([2.0])]}
        log_epoch(x_tup, loss_keys, epoch_loss, opts)
        assert mock_wandb.called

    @patch("wandb.log")
    @patch("logic.src.tracking.logging.structured_logging.plt")
    def test_log_training(self, mock_plt, mock_wandb):
        """Test log_training with mocked components."""
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
                loss_keys, table_df, opts, False
            )
        assert mock_wandb.called

    @patch("logic.src.tracking.logging.modules.storage.os.path.isfile")
    @patch("logic.src.tracking.logging.modules.storage.read_json")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("json.dump")
    def test_log_to_json(self, mock_dump, mock_open, mock_read, mock_isfile):
        """Test log_to_json with mock file."""
        mock_isfile.return_value = True
        mock_read.return_value = {"runs": []}
        mock_open.return_value.__enter__.return_value = MagicMock()
        with (
            patch("logic.src.tracking.logging.modules.storage._sort_log"),
        ):
            res = log_to_json("path.json", ["val"], {"key": [1]}, sort_log_flag=True)
            assert res is not None
            assert mock_dump.called

    @patch("logic.src.tracking.logging.modules.storage.os.path.isfile")
    @patch("logic.src.tracking.logging.modules.storage.read_json")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("json.dump")
    def test_log_to_json2(self, mock_dump, mock_open, mock_read, mock_isfile):
        """Test log_to_json2 (thread-safe version) with mock file."""
        mock_isfile.return_value = False
        mock_read.return_value = []
        mock_open.return_value.__enter__.return_value = MagicMock()
        with patch("json.load", return_value=[]):
            res = log_to_json2("path.json", ["val"], {"key": [2]})
            assert res is not None
            assert mock_dump.called

    @patch("wandb.log")
    def test_log_values(self, mock_wandb):
        """Test log_values function."""
        cost = torch.tensor([10.0])
        grad_norms = (
            [torch.tensor([1.0]), torch.tensor([1.5])],
            [torch.tensor([0.5]), torch.tensor([0.6])],
        )
        epoch = 1
        batch_id = 5
        step = 100
        l_dict = {
            "reinforce_loss": torch.tensor([1.0]),
            "nll": torch.tensor([0.5]),
            "imitation_loss": torch.tensor([0.1]),
            "baseline_loss": torch.tensor([0.2]),
        }
        tb_logger = MagicMock()
        opts = OmegaConf.create(
            {
                "rl": {"wandb_mode": "online", "no_tensorboard": False},
                "train": {"train_time": False},
                "baseline": "critic",
            }
        )
        log_values(cost, cast(Any, grad_norms), epoch, batch_id, step, l_dict, tb_logger, opts)
        assert mock_wandb.called
        assert tb_logger.add_scalar.called

    @patch("logic.src.tracking.logging.modules.analysis.os.path.isfile", return_value=True)
    @patch("logic.src.tracking.logging.modules.analysis.read_json")
    @patch("logic.src.tracking.logging.modules.analysis.update_policy_log_section")
    def test_output_stats(self, mock_update_section, mock_read, mock_isfile):
        """Test output_stats function."""
        mock_read.return_value = {
            "samples": {
                "0": {"cost": 10.0},
                "1": {"cost": 20.0},
            }
        }
        # Invoke via compose_dirpath: (home_dir, ndays, nbins, output_dir, area,
        #   waste_type, data_distribution, run_name) then wrapped-fn kwargs
        mean, std = output_stats(  # type: ignore
            "home",
            1,
            50,
            "out",
            "area",
            "paper",
            "gamma",
            "run1",
            nsamples=2,
            policies=["policy1"],
            keys=["cost"],
            print_output=False,
        )
        assert isinstance(mean, dict)
        assert isinstance(std, dict)
        assert mock_update_section.called

    @patch("logic.src.tracking.logging.modules.analysis.os.path.exists")
    @patch("logic.src.tracking.logging.modules.analysis.read_json")
    def test_runs_per_policy(self, mock_read, mock_exists):
        """Test runs_per_policy function."""
        mock_exists.return_value = True
        mock_read.return_value = {"samples": {"0": {"cost": 10.0}, "1": {"cost": 20.0}}}
        # Invoke via compose_dirpath: list nbins → list of dir_paths
        res = runs_per_policy(  # type: ignore
            "home",
            1,
            [50],
            "out",
            "area",
            "paper",
            "gamma",
            "run1",
            nsamples=[2],
            policies=["policy1"],
            print_output=True,
        )
        assert isinstance(res, list)
        assert len(res) == 1
        assert res[0]["policy1"] == [0, 1]
