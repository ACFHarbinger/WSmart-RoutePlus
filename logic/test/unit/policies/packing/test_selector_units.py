"""Selector unit and config regressions (B-cursor-01, B-cursor-04)."""

from dataclasses import asdict

import numpy as np
import pytest
from logic.src.configs.policies.other.packing import ServiceLevelSelectionConfig
from logic.src.interfaces.context.packing_context import PackingContext
from logic.src.policies.packing.last_minute import LastMinuteSelection

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_typed_service_level_config_carries_the_horizon():
    """The typed path used to drop horizon_days, so the action fell back to 3 days."""
    assert asdict(ServiceLevelSelectionConfig(confidence_factor=0.84))["horizon_days"] == 1
    assert asdict(ServiceLevelSelectionConfig(confidence_factor=0.84, horizon_days=2))["horizon_days"] == 2


def test_last_minute_threshold_is_inclusive_percent():
    """A threshold of 70 selects bins at 70 % or more (the paper's CF = 0.70)."""
    ctx = PackingContext(bin_ids=np.arange(3), current_fill=np.array([50.0, 70.0, 90.0]), threshold=70.0)
    selected, _ = LastMinuteSelection().pack(ctx)
    assert selected == [2, 3]
