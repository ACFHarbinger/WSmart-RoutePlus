"""
Generate a pairwise distance matrix CSV for a given region and waste type.

Saves to data/simulator/distance_matrix/<dm-filepath>.

The output format matches the existing project matrices:
  header row : -1,<depot_id>,<bin1_id>,...
  data rows  : <node_id>,<dist0>,<dist1>,...

Supported methods
-----------------
  gmaps   Google Maps Distance Matrix API (requires GOOGLE_API_KEY in env/vars.env)
  hsd     Haversine great-circle distance
  ogd     Euclidean (OGD) approximation
  gdsc    Geodesic (WGS-84 ellipsoid)
  osm     OpenStreetMap routing (requires osmnx)

Usage
-----
    uv run python logic/gen/gen_dist_matrix.py \\
        --area figueiradafoz \\
        --waste-type plastic \\
        --method gmaps \\
        --dm-filepath "gmaps_distmat_plastic[figueiradafoz].csv"
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

# Ensure project root is on sys.path when running directly
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from logic.src.constants import ROOT_DIR  # noqa: E402
from logic.src.data.network import (  # noqa: E402
    EuclideanStrategy,
    GeodesicStrategy,
    GoogleMapsStrategy,
    HaversineStrategy,
    OSMStrategy,
)
from logic.src.pipeline.simulations.repository import (  # noqa: E402
    load_depot,
    load_simulator_data,
    set_repository_from_path,
)

STRATEGIES = {
    "gmaps": GoogleMapsStrategy,
    "hsd": HaversineStrategy,
    "ogd": EuclideanStrategy,
    "gdsc": GeodesicStrategy,
    "osm": OSMStrategy,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--area", required=True, help="Region name (e.g. figueiradafoz, riomaior)")
    p.add_argument("--waste-type", required=True, help="Waste type (e.g. plastic, paper, glass)")
    p.add_argument(
        "--method",
        required=True,
        choices=list(STRATEGIES),
        help="Distance computation method",
    )
    p.add_argument(
        "--dm-filepath",
        required=True,
        help="Output filename, saved to data/simulator/distance_matrix/",
    )
    p.add_argument(
        "--num-bins",
        type=int,
        default=10_000,
        help="Maximum number of bins to load (default: 10000 = load all available)",
    )
    p.add_argument(
        "--env-file",
        default="vars.env",
        help="Env file in env/ directory for API keys (default: vars.env)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = os.path.join(ROOT_DIR, "data", "simulator")
    set_repository_from_path(data_dir)

    # ── Load coordinates ──────────────────────────────────────────────────────
    depot_df = load_depot(data_dir, args.area)
    depot_row = depot_df[["ID", "Lat", "Lng"]].copy()

    _, bins_coords = load_simulator_data(data_dir, args.num_bins, args.area, args.waste_type)
    bins = bins_coords[["ID", "Lat", "Lng"]].copy()

    # Depot at position 0, then all bins
    coords = pd.concat([depot_row, bins], ignore_index=True)
    n = len(coords)
    ids = coords["ID"].to_numpy()
    print(f"Nodes: {n}  (1 depot + {len(bins)} {args.waste_type} bins in {args.area})")

    # ── Compute distance matrix ───────────────────────────────────────────────
    print(f"Computing {n}×{n} distance matrix with method='{args.method}' ...")
    strategy = STRATEGIES[args.method]()
    dm = strategy.calculate(coords, env_filename=args.env_file)

    # ── Save in project-standard format ──────────────────────────────────────
    out_path = os.path.join(ROOT_DIR, "data", "simulator", "distance_matrix", args.dm_filepath)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"Saving → {out_path}")
    with open(out_path, "w") as f:
        f.write(",".join(["-1"] + [str(int(i)) for i in ids]) + "\n")
        for i, row in enumerate(dm):
            f.write(",".join([str(int(ids[i]))] + [f"{v:.4f}" for v in row]) + "\n")

    print(f"Done. {n}×{n} matrix saved.")


if __name__ == "__main__":
    main()
