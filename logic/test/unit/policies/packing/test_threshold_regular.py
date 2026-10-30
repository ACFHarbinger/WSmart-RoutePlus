"""Threshold Regular: scheduled days select only the bins at or above the minimum fill."""

import numpy as np
import pytest
from logic.src.interfaces.context import PackingContext
from logic.src.policies.packing.regular import RegularSelection

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _context(day: int, min_fill: float) -> PackingContext:
    return PackingContext(
        bin_ids=np.arange(4, dtype="int32"),
        current_fill=np.array([10.0, 50.0, 49.9, 95.0]),
        current_day=day,
        threshold=3,
        min_fill=min_fill,
    )


@pytest.mark.parametrize("day", [1, 4, 7])
def test_scheduled_days_select_bins_at_or_above_the_minimum_fill(day):
    selected, _ = RegularSelection().pack(_context(day, 50.0))
    assert selected == [2, 4]


@pytest.mark.parametrize("day", [2, 3, 5, 6])
def test_other_days_select_nothing(day):
    selected, _ = RegularSelection().pack(_context(day, 50.0))
    assert selected == []


def test_zero_minimum_fill_is_plain_regular():
    selected, _ = RegularSelection().pack(_context(1, 0.0))
    assert selected == [1, 2, 3, 4]
