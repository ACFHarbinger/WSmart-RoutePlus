"""Tests for baselines.py."""

from typing import cast
from unittest.mock import MagicMock

import pytest
import torch
from logic.src.pipeline.learning.common.baselines import (
    ExponentialBaseline,
    RolloutBaseline,
    WarmupBaseline,
    get_baseline,
)
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_exponential_baseline_updates():
    """Verify exponential moving average updates."""
    bl = ExponentialBaseline(beta=0.5)
    r1 = torch.tensor([10.0])
    r2 = torch.tensor([20.0])

    # First eval sets mean. eval does not read td; None is the historical unused state.
    unused = cast(TensorDict, None)
    bl.eval(unused, r1)
    assert bl.running_mean == 10.0

    # Second eval updates mean: 0.5*10 + 0.5*20 = 15
    val2 = bl.eval(unused, r2)
    assert bl.running_mean == 15.0
    assert val2.item() == 15.0


def test_rollout_baseline_setup():
    """Verify rollout baseline policy setup."""
    mock_policy = MagicMock(spec=torch.nn.Module)
    mock_policy.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]

    bl = RolloutBaseline()
    bl.setup(mock_policy)

    assert bl.baseline_policy is not None
    # Should be in eval mode
    assert not any(p.requires_grad for p in bl.baseline_policy.parameters())


def test_warmup_baseline_transition():
    """Verify warmup baseline transition from exponential to target."""
    target_bl = MagicMock()
    target_bl.eval.return_value = torch.tensor([100.0])

    bl = WarmupBaseline(baseline=target_bl, warmup_epochs=2)
    reward = torch.tensor([10.0])

    # Epoch 0 (alpha=0.5 if we assume epoch_callback called with epoch=0)
    bl.epoch_callback(MagicMock(), 0)
    assert bl.alpha == 0.5

    # Blend: 0.5 * 100 + 0.5 * 10 = 55. eval does not read td.
    unused = cast(TensorDict, None)
    val = bl.eval(unused, reward)
    assert val.item() == 55.0

    # Epoch 1 (alpha=1.0)
    bl.epoch_callback(MagicMock(), 1)
    assert bl.alpha == 1.0
    val = bl.eval(unused, reward)
    assert val.item() == 100.0


def test_get_baseline_factory():
    """Verify baseline factory creation."""
    bl = get_baseline("exponential", beta=0.9)
    assert isinstance(bl, ExponentialBaseline)
    assert bl.beta == 0.9
