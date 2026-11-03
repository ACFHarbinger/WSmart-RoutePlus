"""Opening stock: each bin opens with the deposits since a random last collection, capped at capacity."""

import numpy as np
import pytest
from logic.src.constants import MAX_CAPACITY_PERCENT
from logic.src.pipeline.simulations.bins.filling import BinsFillingMixin

pytestmark = [pytest.mark.unit, pytest.mark.fast]


class _Dataset:
    distribution = "gamma3"


class _Bins(BinsFillingMixin):
    def __init__(self, n: int):
        self.n = n
        self.data_dir = "."
        self.waste_dataset = _Dataset()
        self.real_c = np.zeros(n)
        self.c = np.zeros(n)
        self.level_history = [np.zeros(n)]


def test_zero_window_keeps_the_bins_empty():
    bins = _Bins(20)
    assert not bins.set_initial_fill(0, seed=1).any()
    assert not bins.real_c.any()


def test_opening_stock_is_reproducible_and_within_capacity():
    a, b = _Bins(200), _Bins(200)
    opening = a.set_initial_fill(7, seed=11)
    np.testing.assert_array_equal(opening, b.set_initial_fill(7, seed=11))
    assert (opening >= 0).all() and (opening <= MAX_CAPACITY_PERCENT).all()
    assert (opening == 0).any() and (opening > 0).mean() > 0.5  # tau = 0 empties some bins only
    np.testing.assert_array_equal(a.real_c, opening)
    np.testing.assert_array_equal(a.c, opening)
    np.testing.assert_array_equal(a.level_history[-1], opening)


def test_different_seeds_give_different_stock():
    assert not np.array_equal(_Bins(50).set_initial_fill(5, seed=1), _Bins(50).set_initial_fill(5, seed=2))
