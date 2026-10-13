"""Rebuild the simulation summary CSVs (archive schema) from the per-run logs under assets/output.

Usage:
    python logic/gen/build_simulation_summary.py OUT_DIR [--root assets/output]

Writes ``simulation_summary.csv`` (30-day) and ``simulation_summary_90d.csv`` (90-day) into
OUT_DIR. Each row is one policy configuration in one scenario cell, taken from the ``mean``
block of its per-run log, which is what the simulator averages over samples.

The schema matches the archived CSVs under ``docs/private/global/simulation/``, which
``gen_paper_latex.py`` reads.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
from typing import Dict, List, Optional, Tuple

COLS = [
    "city",
    "N",
    "dist",
    "improver",
    "strategy",
    "cf",
    "sl_var",
    "acceptance",
    "constructor",
    "overflows",
    "kg",
    "ncol",
    "kg_lost",
    "km",
    "kgkm",
    "reward",
    "profit",
    "time",
    "days",
]
AREAS = {
    "figueiradafoz350": ("Figueira da Foz", 350),
    "riomaior100": ("Rio Maior", 100),
    "riomaior170": ("Rio Maior", 170),
}
DISTS = {"emp": "Empirical", "gamma3": "Gamma-3"}
# (policy-key stem without the improver suffix, constructor label, acceptance label)
CONSTRUCTORS = [
    ("swc_tcf_gurobi", "SWC-TCF", "gurobi"),
    ("bpc_custom", "BPC", "custom"),
    ("alns_custom_bmc", "ALNS", "custom_bmc"),
    ("hgs_custom_oi", "HGS", "custom_oi"),
    ("pg_clns_custom", "PG-CLNS", "custom"),
    ("psoma_bmc", "PSOMA", "bmc"),
    ("sans_new", "SANS", "new"),
    ("aco_hh_custom", "ACO_HH", "custom"),
]


def parse_log(log: pathlib.Path) -> Optional[Dict[str, object]]:
    """Return one summary row for a per-run log, or None if it is not a tested configuration."""
    area, dist_dir, ms_dir = log.parts[-4], log.parts[-3], log.parts[-2]
    area_key = area.replace("_plastic", "")
    if area_key not in AREAS or dist_dir not in DISTS:
        return None
    city, n = AREAS[area_key]
    improver = "CLS" if ms_dir.endswith("_cls") else "FTSP"

    stem = re.sub(r"_1N\.json$", "", log.name[len("log_") :])
    strategy, cf, sl_var = "", "", ""
    if stem.startswith("lookahead_"):
        strategy, stem = "LA", stem[len("lookahead_") :]
    elif stem.startswith("last_minute_cf"):
        strategy, cf = "LM", "CF" + stem[len("last_minute_cf") : len("last_minute_cf") + 2]
        stem = stem[len("last_minute_cf70_") :]
    elif stem.startswith("service_level"):
        strategy, sl_var = "SL", "SL" + stem[len("service_level")]
        stem = stem[len("service_level1_") :]
    stem = re.sub(r"_(cls|ftsp)$", "", stem)

    matches = [c for c in CONSTRUCTORS if stem == c[0]]
    if not matches:
        return None  # regular_* and other non-tested configurations
    _, constructor, acceptance = matches[0]

    mean = json.loads(log.read_text())["mean"]
    return {
        "city": city,
        "N": n,
        "dist": DISTS[dist_dir],
        "improver": improver,
        "strategy": strategy,
        "cf": cf,
        "sl_var": sl_var,
        "acceptance": acceptance,
        "constructor": constructor,
        "overflows": mean["overflows"],
        "kg": mean["kg"],
        "ncol": mean["ncol"],
        "kg_lost": mean["kg_lost"],
        "km": mean["km"],
        "kgkm": mean["kg/km"],
        "reward": mean["reward"],
        "profit": mean["profit"],
        "time": mean["time"],
        "days": mean["days"],
    }


def collect(root: pathlib.Path) -> Dict[str, List[Dict[str, object]]]:
    """Collect the rows of each horizon, keyed by ``30`` or ``90``."""
    rows: Dict[str, List[Dict[str, object]]] = {"30": [], "90": []}
    for folder, key in (("30days", "30"), ("90days", "90")):
        for log in sorted((root / folder).glob("*/*/*/log_*_1N.json")):
            row = parse_log(log)
            if row is not None:
                rows[key].append(row)
    return rows


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=pathlib.Path, help="Folder for the two summary CSVs.")
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("assets/output"))
    args = parser.parse_args(argv)

    rows = collect(args.root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Tuple[pathlib.Path, int]] = []
    for key, name in (("30", "simulation_summary.csv"), ("90", "simulation_summary_90d.csv")):
        path = args.out_dir / name
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLS)
            writer.writeheader()
            writer.writerows(rows[key])
        outputs.append((path, len(rows[key])))
    for path, count in outputs:
        print(f"{path}: {count} rows")


if __name__ == "__main__":
    main()
