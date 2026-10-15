"""
Tests for RL pipeline features (epoch utilities, metrics).
"""

from typing import Any

import pytest
import torch
from logic.src.pipeline.rl.common.epoch import compute_validation_metrics, prepare_epoch
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]





class TestFeatures:
    """Tests for RL feature modules."""

    def test_prepare_epoch_baselines(self, am_setup, mocker):
        """Test dataset preparation with baseline wrapping."""
        policy = am_setup
        env = mocker.MagicMock()
        baseline = mocker.MagicMock()
        baseline.wrap_dataset.side_effect = lambda d, policy=None, env=None: "wrapped_dataset"
        baseline.unwrap_dataset.side_effect = lambda d: d

        dataset: Any = "original_dataset"

        # Training phase with baseline wrapping
        wrapped = prepare_epoch(policy, env, baseline, dataset, 0, phase="train")
        assert wrapped == "wrapped_dataset"
        baseline.wrap_dataset.assert_called_once_with(dataset, policy=policy, env=env)

        # Validation phase (no wrapping)
        val_ds = prepare_epoch(policy, env, baseline, dataset, 0, phase="val")
        assert val_ds == dataset

    def test_compute_validation_metrics(self, mocker):
        """Test validation metric aggregation."""
        env = mocker.MagicMock()
        # Mock env.evaluate_tour
        env.evaluate_tour.return_value = {"waste": torch.tensor([10.0, 20.0]), "dist": torch.tensor([2.0, 4.0])}
        env.get_costs.return_value = {"waste": torch.tensor([10.0, 20.0]), "dist": torch.tensor([2.0, 4.0])}
        env.get_num_overflows.return_value = torch.tensor([1.0, 0.0])

        batch = TensorDict({"loc": torch.randn(2, 10, 2)}, batch_size=[2])
        out = {"reward": torch.tensor([1.0, 2.0]), "actions": torch.zeros(2, 5)}

        metrics = compute_validation_metrics(out, batch, env)

        assert metrics["val/reward"] == 1.5
        assert metrics["val/waste"] == 15.0
        assert metrics["val/dist"] == 3.0
        assert metrics["val/efficiency"] == 5.0  # 15 / 3
        assert metrics["val/overflows"] == 0.5
