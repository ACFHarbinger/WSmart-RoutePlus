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
kept interval are empty (NaN). The file has the layout of the crude series (a ``Date`` column and one column per
bin ID), and ``logic/src/utils/data/loader.py`` reads it for Figueira da Foz.

Readings are capped at 100%, so where bins overflow the rate is a lower bound. The readings look like drivers'
estimates in coarse steps.

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
OUTPUT = "out_rate_records[figdafoz].csv"


def collection_rates(records: pd.DataFrame, min_gap: float = 0.5, max_gap: float = 60.0) -> pd.DataFrame:
    """Daily accumulation rate (% of capacity per day) of every bin, one row per calendar day.

    Args:
        records: Collection records with the columns ``idcontentor``, ``idrecolha``, ``Data de ínicio`` and
            ``Enchimento`` (one row per bin emptied on a tour).
        min_gap: Shortest interval between two collections that is kept, in days.
        max_gap: Longest interval kept, in days.

    Returns:
        A frame indexed by date (daily) with one column per bin ID.
    """
    r = records[records.idrecolha.notna() & records.idcontentor.notna()].copy()
    r["start"] = pd.to_datetime(r["Data de ínicio"], errors="coerce")
    r["fill"] = pd.to_numeric(r["Enchimento"], errors="coerce")
    r = r.dropna(subset=["start", "fill"])
    r["idcontentor"] = r["idcontentor"].astype(int)
    r = r.drop_duplicates(["idcontentor", "idrecolha"]).sort_values(["idcontentor", "start"])
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
    rates = collection_rates(pd.read_excel(src), args.min_gap, args.max_gap)
    dst = os.path.join(args.data_dir, "bins_waste", OUTPUT)
    rates.round(2).reset_index().assign(Date=lambda f: f.Date.dt.strftime("%Y-%m-%d")).to_csv(dst, index=False)
    per_bin = rates.mean()
    print(
        f"{dst}: {rates.shape[1]} bins, {rates.shape[0]} days, {int(rates.notna().sum().sum())} bin-days with a rate; "
        f"per-bin mean {per_bin.mean():.2f} %/day (median {per_bin.median():.2f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
