"""Selector unit and config regressions (B-cursor-01, B-cursor-04)."""

from dataclasses import asdict

import numpy as np
import pytest
from logic.src.configs.policies.other.mandatory_selection import ServiceLevelSelectionConfig
from logic.src.interfaces.context.selection_context import SelectionContext
from logic.src.policies.mandatory_selection.selection_last_minute import LastMinuteSelection

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_typed_service_level_config_carries_the_horizon():
    """The typed path used to drop horizon_days, so the action fell back to 3 days."""
    assert asdict(ServiceLevelSelectionConfig(confidence_factor=0.84))["horizon_days"] == 1
    assert asdict(ServiceLevelSelectionConfig(confidence_factor=0.84, horizon_days=2))["horizon_days"] == 2


def test_last_minute_threshold_is_inclusive_percent():
    """A threshold of 70 selects bins at 70 % or more (the paper's CF = 0.70)."""
    ctx = SelectionContext(bin_ids=np.arange(3), current_fill=np.array([50.0, 70.0, 90.0]), threshold=70.0)
    selected, _ = LastMinuteSelection().select_bins(ctx)
    assert selected == [2, 3]
