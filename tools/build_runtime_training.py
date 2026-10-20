"""Build the cloud runner's runtime training table (CR-11) from local rerun logs and summaries.

Usage::

    python tools/build_runtime_training.py \\
        --ctl ~/Repositories/Doc/wsr-package/cptp-rerun/ctl ~/Repositories/Doc/wsr-package/acohh-rerun/ctl \\
        --summary docs/private/global/simulation/rerun-2026-10-19 \\
        --out logic/controllers/cloud/prediction/runtime_training.csv

Two kinds of rows:

- ``rerun_wall``: the wall time of one cell run by a rerun driver, from its ``events.log``
  (``done <seconds>s <cell_id>``) joined with the driver's ``manifest.json``.
- ``policy_time``: every other run of the summary CSVs, whose wall time is estimated as
  its policy time plus the median wall-minus-policy overhead of the ``rerun_wall`` cells
  with the same horizon and network size (on those cells wall time is 1.00-1.03 x policy time).

Every row was run on one laptop with several cells in parallel (``machine_type=laptop-i9-12900hx``).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

COLUMNS = [
    "policy",
    "packing",
    "improver",
    "distribution",
    "area",
    "size",
    "horizon",
    "machine_type",
    "wall_seconds",
    "source",
    "cell_id",
]
MACHINE = "laptop-i9-12900hx"
POLICIES = {
    "ACO_HH": "aco_hh",
    "ALNS": "alns",
    "AM": "am",
    "BPC": "bpc",
    "CVRP": "cvrp",
    "HGS": "hgs",
    "PG-CLNS": "pg_clns",
    "PSOMA": "psoma",
    "SANS": "sans",
    "SWC-TCF": "swc_tcf",
}
PACKING = {"LA": "lookahead", "LMCF70": "cf70", "LMCF90": "cf90", "SLSL1": "sl1", "SLSL2": "sl2"}
AREAS = {"Figueira da Foz": "figueiradafoz", "Rio Maior": "riomaior"}
DISTRIBUTIONS = {"Empirical": "emp", "Gamma-3": "gamma3"}


def rerun_rows(ctl_dirs: List[Path]) -> List[Dict[str, object]]:
    rows = []
    for ctl in ctl_dirs:
        cells = {c["cell_id"]: c for c in json.loads((ctl / "manifest.json").read_text())["cells"]}
        for line in (ctl / "events.log").read_text().splitlines():
            match = re.search(r" done (\d+(?:\.\d+)?)s (\S+)", line)
            if not match or match.group(2) not in cells:
                continue
            c = cells[match.group(2)]
            rows.append(
                {
                    "policy": c["policy"],
                    "packing": c["selection_variant"],
                    "improver": c["improver"],
                    "distribution": c["distribution"],
                    "area": c["area"],
                    "size": c["size"],
                    "horizon": c["horizon"],
                    "machine_type": MACHINE,
                    "wall_seconds": float(match.group(1)),
                    "source": "rerun_wall",
                    "cell_id": c["cell_id"],
                }
            )
    return rows


def summary_rows(summary_dir: Path) -> List[Tuple[Dict[str, object], float]]:
    """``(row without wall time, policy seconds)`` for every summary run."""
    out = []
    for horizon, name in ((30, "simulation_summary.csv"), (90, "simulation_summary_90d.csv")):
        for r in csv.DictReader(open(summary_dir / name)):
            packing = PACKING.get(
                r["strategy"] + r["cf"] + r["sl_var"], (r["strategy"] + r["cf"] + r["sl_var"]).lower()
            )
            row = {
                "policy": POLICIES.get(r["constructor"], r["constructor"].lower()),
                "packing": packing,
                "improver": (r["improver"] or "none").lower(),
                "distribution": DISTRIBUTIONS[r["dist"]],
                "area": AREAS[r["city"]],
                "size": int(r["N"]),
                "horizon": horizon,
                "machine_type": MACHINE,
                "source": "policy_time",
            }
            row["cell_id"] = (
                f"{horizon}d/{row['area']}/{row['size']}/{row['distribution']}/{row['improver']}/{packing}/{row['policy']}"
            )
            out.append((row, float(r["time"])))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ctl", nargs="+", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    walls = rerun_rows(args.ctl)
    runs = summary_rows(args.summary)
    by_key = {
        (r["horizon"], r["area"], r["size"], r["distribution"], r["improver"], r["packing"], r["policy"]): t
        for r, t in runs
    }
    overhead: Dict[Tuple[int, int], List[float]] = {}
    for w in walls:
        key = (w["horizon"], w["area"], w["size"], w["distribution"], w["improver"], w["packing"], w["policy"])
        if key in by_key:
            overhead.setdefault((w["horizon"], w["size"]), []).append(w["wall_seconds"] - by_key[key])
    median_overhead = {k: statistics.median(v) for k, v in overhead.items()}
    timed = {
        (w["horizon"], w["area"], w["size"], w["distribution"], w["improver"], w["packing"], w["policy"]) for w in walls
    }
    rows = list(walls)
    for row, policy_seconds in runs:
        key = (
            row["horizon"],
            row["area"],
            row["size"],
            row["distribution"],
            row["improver"],
            row["packing"],
            row["policy"],
        )
        if key in timed:
            continue  # the direct wall time is already in the table
        row["wall_seconds"] = round(policy_seconds + median_overhead.get((row["horizon"], row["size"]), 0.0), 1)
        rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["source"], r["cell_id"])))
    print(f"{len(rows)} rows ({len(walls)} rerun_wall); overhead by (horizon, size): {median_overhead}")


if __name__ == "__main__":
    main()
