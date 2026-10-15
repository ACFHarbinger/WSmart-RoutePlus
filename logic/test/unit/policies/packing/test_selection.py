"""Tests for Packing algorithms."""

import numpy as np
import pytest
from logic.src.interfaces.context.packing_context import PackingContext
from logic.src.policies.packing.last_minute import LastMinuteSelection
from logic.src.policies.packing.lookahead import LookaheadSelection
from logic.src.policies.packing.regular import RegularSelection

pytestmark = [pytest.mark.unit, pytest.mark.fast]





@pytest.fixture
def base_context():
    """Base context for selection tests."""
    n_bins = 5
    # IDs 0..4 in context used as indices, but strategies return (idx + 1)
    return PackingContext(
        bin_ids=np.arange(n_bins),
        current_fill=np.array([10.0, 95.0, 30.0, 85.0, 50.0]),
        current_day=3,
        threshold=90.0
    )

class TestRegularSelection:
    def test_regular_selection_logic(self):
        strategy = RegularSelection()
        # Case 1: ((current_day - 1) % threshold) == 0 -> select all
        # ((4 - 1) % 3) == 0.
        ctx = PackingContext(bin_ids=np.array([0, 1]), current_fill=np.zeros(2), current_day=4, threshold=3.0)
        # Should return [1, 2] (indices 0, 1 mapped to IDs 1, 2)
        mandatory, _ = strategy.pack(ctx)
        assert list(mandatory) == [1, 2]

        # Case 2: skip
        ctx = PackingContext(bin_ids=np.array([0, 1]), current_fill=np.zeros(2), current_day=5, threshold=3.0)
        mandatory, _ = strategy.pack(ctx)
        assert list(mandatory) == []

class TestLastMinuteSelection:
    def test_last_minute_basic(self, base_context):
        strategy = LastMinuteSelection()
        mandatory, _ = strategy.pack(base_context)
        # current_fill = [10.0, 95.0, 30.0, 85.0, 50.0], threshold=90.0
        # Index 1 is > 90. Mapped to ID 2.
        assert list(mandatory) == [2]

class TestLookaheadSelection:
    def test_lookahead_no_prediction(self, base_context):
        strategy = LookaheadSelection()
        mandatory, _ = strategy.pack(base_context)
        assert list(mandatory) == []
