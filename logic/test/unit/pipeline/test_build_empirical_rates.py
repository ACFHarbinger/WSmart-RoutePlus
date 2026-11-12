"""Tests for logic/gen/build_empirical_rates.py and the Empirical rate source of the loader."""

import importlib.util
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[4]


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_empirical_rates", ROOT / "logic/gen/build_empirical_rates.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_empirical_rates"] = module
    spec.loader.exec_module(module)
    return module


def _records(rows):
    return pd.DataFrame(rows, columns=["idcontentor", "idrecolha", "Data de ínicio", "Enchimento"])


def test_rate_is_fill_at_collection_over_the_interval():
    """The bin is emptied at each collection, so the waste added is the next reading, spread over the interval."""
    b = _load_builder()
    rates = b.collection_rates(
        _records(
            [
                (7, 1, "2024-01-01 08:00", 40),  # first visit: no previous collection, no rate
                (7, 2, "2024-01-05 08:00", 60),  # 60% in 4 days -> 15%/day on Jan 2..5
                (7, 3, "2024-01-07 08:00", 20),  # 20% in 2 days -> 10%/day on Jan 6..7
            ]
        )
    )
    s = rates[7]
    assert np.isnan(s["2024-01-01"])
    assert (s["2024-01-02":"2024-01-05"] == 15.0).all()
    assert (s["2024-01-06":"2024-01-07"] == 10.0).all()


def test_intervals_and_readings_outside_the_limits_are_left_out():
    b = _load_builder()
    rates = b.collection_rates(
        _records(
            [
                (1, 1, "2024-01-01 08:00", 50),
                (1, 2, "2024-01-01 10:00", 50),  # same visit recorded twice (2 h): dropped
                (1, 3, "2024-04-01 08:00", 80),  # 91 days: probably a missed record, dropped
                (1, 4, "2024-04-03 08:00", 150),  # reading above 100%: dropped
                (1, 4, "2024-04-03 08:00", 150),  # duplicate row of the same tour
            ]
        )
    )
    assert rates[1].isna().all()


def test_loader_requires_the_records_series_for_figueira_da_foz(tmp_path):
    from logic.src.utils.data.loader import load_grid_base

    (tmp_path / "bins_waste").mkdir()
    (tmp_path / "coordinates").mkdir()
    with pytest.raises(FileNotFoundError, match="build_empirical_rates.py"):
        load_grid_base(None, "figueiradafoz", str(tmp_path), ids=[1, 2])


def test_loader_asks_for_mean_preserving_draws(tmp_path):
    from logic.src.utils.data import loader

    (tmp_path / "bins_waste").mkdir()
    pd.DataFrame({"Date": ["2024-01-01"], "1": [3.0]}).to_csv(
        tmp_path / "bins_waste" / "old_out_crude_rate[riomaior].csv", index=False
    )
    with mock.patch("logic.submodules.wsmart_bin_analysis.GridBase") as grid:
        loader.load_grid_base(None, "riomaior", str(tmp_path), ids=[1])
    assert grid.call_args.kwargs["mean_preserving"] is True
    assert grid.call_args.kwargs["names"][0] == "old_out_crude_rate[riomaior].csv"


def test_bins_without_a_kept_interval_take_the_daily_median():
    b = _load_builder()
    days = pd.date_range("2024-01-01", periods=3, name="Date")
    rates = pd.DataFrame({1: [2.0, 4.0, 6.0], 2: [4.0, 8.0, np.nan], 3: [np.nan] * 3}, index=days)
    out, filled = b.fill_empty_bins(rates, [1, 2, 3, 4])
    assert filled == [3, 4]
    assert list(out[3]) == [3.0, 6.0, 6.0]  # median of bins 1 and 2 each day (bin 2 missing on day 3)
    assert list(out[4]) == list(out[3])
    assert list(out[1]) == [2.0, 4.0, 6.0]


def test_weights_share_the_tour_weight_in_proportion_to_the_readings():
    """Tour 2 weighs 30 kg over two bins read at 50% and 25%: 20 kg and 10 kg, or 40% and 20% of a 50 kg bin."""
    b = _load_builder()
    rows = pd.DataFrame(
        [
            (1, 1, "2024-01-01 08:00", 50, 30.0),
            (2, 1, "2024-01-01 08:00", 50, 30.0),
            (1, 2, "2024-01-05 08:00", 50, 30.0),  # 20 kg in 4 days -> 40% / 4 = 10%/day
            (2, 2, "2024-01-05 08:00", 25, 30.0),  # 10 kg in 4 days -> 20% / 4 = 5%/day
            (1, 3, "2024-01-07 08:00", 100, 500.0),  # 500 kg on one bin: capped at 100% -> 50%/day
        ],
        columns=["idcontentor", "idrecolha", "Data de ínicio", "Enchimento", "Peso total"],
    )
    w = b.collection_rates(rows, source="weights", capacity_kg=50.0)
    r = b.collection_rates(rows, source="readings")
    assert (w.loc["2024-01-02":"2024-01-05", 1] == 10.0).all()
    assert (w.loc["2024-01-02":"2024-01-05", 2] == 5.0).all()
    assert (w.loc["2024-01-06":"2024-01-07", 1] == 50.0).all()
    assert (r.loc["2024-01-02":"2024-01-05", 1] == 12.5).all()  # the reading itself: 50% / 4 days
    with pytest.raises(ValueError):
        b.collection_rates(rows, source="scale")


@pytest.mark.parametrize(
    "rates, name",
    [
        (None, "out_rate_records[figdafoz].csv"),
        ("readings", "out_rate_records[figdafoz].csv"),
        ("weights", "out_rate_weights[figdafoz].csv"),
    ],
)
def test_loader_reads_the_chosen_figueira_series(tmp_path, rates, name):
    from logic.src.utils.data import loader

    (tmp_path / "bins_waste").mkdir()
    pd.DataFrame({"Date": ["2024-01-01"], "1": [3.0]}).to_csv(tmp_path / "bins_waste" / name, index=False)
    with mock.patch("logic.submodules.wsmart_bin_analysis.GridBase") as grid:
        loader.load_grid_base(None, "figueiradafoz", str(tmp_path), ids=[1], rates=rates)
    assert grid.call_args.kwargs["names"][0] == name


def test_loader_rejects_an_unknown_rate_series(tmp_path):
    from logic.src.utils.data.loader import load_grid_base

    with pytest.raises(ValueError, match="rates must be one of"):
        load_grid_base(None, "figueiradafoz", str(tmp_path), ids=[1], rates="scale")
