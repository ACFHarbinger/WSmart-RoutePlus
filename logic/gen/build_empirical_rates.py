"""Build the daily accumulation series of Figueira da Foz from the operator's collection records.

The Empirical demand process samples each bin's daily fill increment from a per-bin rate series. Rio Maior has
continuous fill sensors, and its crude series (``old_out_crude_rate[riomaior].csv``) is the day-to-day change of
the sensor level with collection days removed. Figueira da Foz has no sensor feed: its only readings are the fill
recorded when a bin is collected, after which the bin is emptied. Its crude series (``out_rate_crude[figdafoz].csv``)
spreads the difference between two consecutive readings over the days between them, but the bin was empty after
the first reading, so that difference is not the waste added. The series averages to about zero whatever the true
rate is, and must not be used.

This script writes the series that the readings do support. For every bin and every interval between two
consecutive collections of that bin, the waste added is the fill read at the second collection (the bin was emptied
at the first), so the daily rate over the interval is

    rate = fill at the second collection (% of capacity) / length of the interval (days),

assigned to every calendar day of the interval. Intervals shorter than half a day (the same visit recorded twice)
or longer than 60 days (probably a missed record) are left out, as are readings outside [0, 100]. Days outside any
kept interval are empty (NaN). Every bin of the area's bin information file gets a column, and a bin without any
kept interval (never collected, or only in intervals that were left out) takes the daily median rate of the bins
that have one, so the simulator never draws an empty deposit for it. The file has the layout of the crude series (a ``Date`` column and one column per
bin ID), and ``logic/src/utils/data/loader.py`` reads it for Figueira da Foz.

Readings are capped at 100%, so where bins overflow the rate is a lower bound. The readings are drivers' estimates
in four steps (25, 50, 75 and 100%, nothing lower), and every tour is also weighed. The script therefore writes two
series, and the simulator chooses one with ``sim.empirical_rates`` (``load_grid_base(..., rates=...)``):

* ``readings`` (``out_rate_records[figdafoz].csv``): the fill read at each collection, as above.
* ``weights`` (``out_rate_weights[figdafoz].csv``): the fill of each collection taken from the weighed tour mass. The
  tour weight is shared among the tour's bins in proportion to their readings and converted to percent of the bin
  capacity of the simulator (2.5 m3 at the packaging density of get_area_params, 50 kg), capped at 100%. The series
  then carries the weighed mass and keeps each bin's relative fill. At the set density the readings imply about a
  third more mass than the weights, so this series is about a quarter lower.

Usage (from the repository root; data/simulator is not under version control, so run it once per machine):
    uv run python logic/gen/build_empirical_rates.py
    uv run python logic/gen/build_empirical_rates.py --data-dir data/simulator --min-gap 0.5 --max-gap 60
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

RECORDS = "Enchimentos_com_Recolhas[FigFoz].xlsx"
INFO = "out_info[figdafoz].csv"
OUTPUTS = {"readings": "out_rate_records[figdafoz].csv", "weights": "out_rate_weights[figdafoz].csv"}


def collection_rates(
    records: pd.DataFrame,
    min_gap: float = 0.5,
    max_gap: float = 60.0,
    source: str = "readings",
    capacity_kg: float = 50.0,
) -> pd.DataFrame:
    """Daily accumulation rate (% of capacity per day) of every bin, one row per calendar day.

    Args:
        records: Collection records with the columns ``idcontentor``, ``idrecolha``, ``Data de ínicio`` and
            ``Enchimento`` (one row per bin emptied on a tour), and ``Peso total`` (the tour's weight, kg) for
            ``source="weights"``.
        min_gap: Shortest interval between two collections that is kept, in days.
        max_gap: Longest interval kept, in days.
        source: ``"readings"`` (the fill read at the collection) or ``"weights"`` (the tour weight shared among its
            bins in proportion to their readings, in percent of ``capacity_kg``, capped at 100).
        capacity_kg: Mass of a full bin, for ``source="weights"``.

    Returns:
        A frame indexed by date (daily) with one column per bin ID.
    """
    if source not in OUTPUTS:
        raise ValueError(f"source must be one of {sorted(OUTPUTS)}, got {source!r}")
    r = records[records.idrecolha.notna() & records.idcontentor.notna()].copy()
    r["start"] = pd.to_datetime(r["Data de ínicio"], errors="coerce")
    r["fill"] = pd.to_numeric(r["Enchimento"], errors="coerce")
    r = r.dropna(subset=["start", "fill"])
    r["idcontentor"] = r["idcontentor"].astype(int)
    r = r.drop_duplicates(["idcontentor", "idrecolha"]).sort_values(["idcontentor", "start"])
    if source == "weights":
        weight = pd.to_numeric(r["Peso total"], errors="coerce")
        share = r.fill / r.groupby("idrecolha").fill.transform("sum")
        r["fill"] = (weight * share / capacity_kg * 100.0).clip(upper=100.0)
        r = r[weight > 0]
    r["prev"] = r.groupby("idcontentor").start.shift()
    r["gap"] = (r.start - r.prev).dt.total_seconds() / 86400
    ok = r[(r.gap >= min_gap) & (r.gap <= max_gap) & r.fill.between(0, 100)]
    days = pd.date_range(r.start.min().normalize(), r.start.max().normalize(), freq="D")
    out = pd.DataFrame(np.nan, index=days, columns=sorted(r.idcontentor.unique()))
    out.index.name = "Date"
    for row in ok.itertuples(index=False):
        first, last = row.prev.normalize() + pd.Timedelta(days=1), row.start.normalize()
        if last >= first:
            out.loc[first:last, row.idcontentor] = row.fill / row.gap
    return out


def fill_empty_bins(rates: pd.DataFrame, bin_ids) -> tuple:
    """Add every bin of ``bin_ids`` and give a bin without any rate the daily median rate of the others."""
    out = rates.reindex(columns=sorted(set(rates.columns) | {int(b) for b in bin_ids}))
    empty = out.columns[out.notna().sum() == 0]
    if len(empty):
        median = out.drop(columns=empty).median(axis=1)
        for b in empty:
            out[b] = median
    return out, list(empty)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", default=os.path.join("data", "simulator"))
    parser.add_argument("--min-gap", type=float, default=0.5)
    parser.add_argument("--max-gap", type=float, default=60.0)
    args = parser.parse_args()
    src = os.path.join(args.data_dir, "bins_waste", RECORDS)
    if not os.path.exists(src):
        print(f"missing {src}: copy data/simulator from the project data first", file=sys.stderr)
        return 1
    records = pd.read_excel(src)
    info = pd.read_csv(os.path.join(args.data_dir, "coordinates", INFO))
    capacity_kg = bin_capacity_kg()
    for source, name in OUTPUTS.items():
        rates = collection_rates(records, args.min_gap, args.max_gap, source=source, capacity_kg=capacity_kg)
        rates, filled = fill_empty_bins(rates, info["ID"].dropna().astype(int))
        dst = os.path.join(args.data_dir, "bins_waste", name)
        rates.round(2).reset_index().assign(Date=lambda f: f.Date.dt.strftime("%Y-%m-%d")).to_csv(dst, index=False)
        per_bin = rates.mean()
        print(
            f"{dst} ({source}): {rates.shape[1]} bins, {rates.shape[0]} days, "
            f"{int(rates.notna().sum().sum())} bin-days with a rate; per-bin mean {per_bin.mean():.2f} %/day "
            f"(median {per_bin.median():.2f}); {len(filled)} bins without a kept interval take the daily median rate"
        )
    return 0


def bin_capacity_kg() -> float:
    """Mass of a full Figueira da Foz packaging bin in the simulator (bin volume times density)."""
    sys.path.insert(0, os.getcwd())
    from logic.src.pipeline.simulations.repository.base import SimulationRepository

    _, _, density, _, volume = SimulationRepository.get_area_params("figueiradafoz", "plastic")
    return float(volume * density)


if __name__ == "__main__":
    sys.exit(main())
