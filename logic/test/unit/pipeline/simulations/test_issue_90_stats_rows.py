"""Regression tests for issue #90: external stats-file sample length."""

import numpy as np
import pytest
from logic.src.pipeline.simulations.bins import Bins

pytestmark = [pytest.mark.unit, pytest.mark.fast]


class _ExternalWaste:
    """File-backed stand-in. Not a GenerativeDataset, so the row check applies."""

    def __init__(self, waste: np.ndarray, noisy: np.ndarray) -> None:
        self._waste = waste
        self._noisy = noisy

    def __getitem__(self, index: int):
        return {"waste": self._waste, "noisy_waste": self._noisy}


def test_external_stats_sample_rejects_a_short_file(tmp_path, mock_bins_params_loader):
    """A file-backed stats sample with only n_days rows fails before day 1."""
    horizon = 4
    bins = Bins(
        n=3,
        data_dir=str(tmp_path),
        sample_dist="gamma",
        area="riomaior",
        waste_type="paper",
        n_days=horizon,
        n_samples=1,
        seed=1,
    )
    short = np.ones((horizon, 3))
    bins.waste_dataset = _ExternalWaste(short, short.copy())
    bins.start_with_fill = True
    with pytest.raises(ValueError, match="opening level"):
        bins.set_sample_waste(0, horizon=horizon)
    assert np.all(bins.real_c == 0)


def test_external_stats_sample_accepts_an_opening_row_plus_each_day(tmp_path, mock_bins_params_loader):
    """n_days + 1 rows cover the opening level and the final day's deposit."""
    horizon = 4
    bins = Bins(
        n=3,
        data_dir=str(tmp_path),
        sample_dist="gamma",
        area="riomaior",
        waste_type="paper",
        n_days=horizon,
        n_samples=1,
        seed=1,
    )
    waste = np.arange((horizon + 1) * 3, dtype=float).reshape(horizon + 1, 3)
    bins.waste_dataset = _ExternalWaste(waste, waste.copy())
    bins.start_with_fill = True
    bins.set_sample_waste(0, horizon=horizon)
    assert np.array_equal(bins.real_c, waste[0])
    bins.load_filling(horizon)


def test_generated_stats_sample_is_not_rejected_for_length(tmp_path, mock_bins_params_loader):
    """Generated samples are sized by the caller and skip the file-length check."""
    bins = Bins(
        n=3,
        data_dir=str(tmp_path),
        sample_dist="gamma",
        area="riomaior",
        waste_type="paper",
        n_days=4,
        n_samples=1,
        seed=1,
    )
    bins.start_with_fill = True
    bins.set_sample_waste(0, horizon=4)
    assert len(bins.waste_fills) == 4
