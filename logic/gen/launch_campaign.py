"""Launch the remote-depot and central-depot benchmark campaign.

For each scenario of ``benchmark_campaign`` (5 deposit realizations per depot placement) this
runs every routing algorithm with each route improver, plus the fixed-schedule baseline, as one
simulator invocation per cell. Each invocation runs the campaign's packing variants through the
overlay policy files (60 s routing limit, no retries, profit-aware operators from the configs).
Finished cells leave a ``.done`` marker and are skipped on a rerun.

With ``--delayed-start`` only the calendar rules are run (threshold-regular TR-k with every routing
algorithm and improver, and the fixed schedule Regular-k), with the first collection on day k instead
of day 1; the outputs go to ``outputs/campaign_delayed`` unless ``--out`` is given. With
``--initial-fill-days W`` every bin opens with the stock of a random 0..W days since its last
collection (``sim.initial_fill_days``) instead of empty, on the same horizon deposits; the outputs go
to ``outputs/campaign_fill<W>`` (``outputs/campaign_fill<W>_delayed`` with ``--delayed-start``).

Usage:
    uv run python logic/gen/launch_campaign.py --network rm100 --dist emp --workers 4 --cores 8
    uv run python logic/gen/launch_campaign.py --network rm100 --dist emp --delayed-start
    uv run python logic/gen/launch_campaign.py --network rm100 --dist emp --initial-fill-days 7
    uv run python logic/gen/launch_campaign.py --network ff350 --dry-run
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from logic.src.pipeline.simulations.campaign import NETWORKS, CampaignScenario, benchmark_campaign  # noqa: E402
from logic.src.pipeline.simulations.campaign_policies import (  # noqa: E402
    DELAYED_PACKING,
    DELAYED_STATIC_PACKING,
    IMPROVERS,
    ROUTING_POLICIES,
    STATIC_POLICY,
    write_overlay,
)


@dataclass(frozen=True)
class Cell:
    """One simulator invocation.

    Attributes:
        name: Unique cell name.
        argv: Command line.
        env: Extra environment variables.
        out_dir: Output directory of the invocation.
    """

    name: str
    argv: List[str]
    env: Dict[str, str]
    out_dir: Path


def build_cells(args: argparse.Namespace, overlay_root: Path) -> List[Cell]:
    """Expand the campaign into cells.

    Args:
        args: Parsed command line.
        overlay_root: Directory that receives one overlay per improver.

    Returns:
        List[Cell]: One cell per scenario, routing algorithm and improver, plus the static baseline.
    """
    network = NETWORKS[args.network]
    scenarios = [s for s in benchmark_campaign(args.per_depot) if args.depots in ("both", s.depot_mode)]
    delayed = getattr(args, "delayed_start", False)
    packing = DELAYED_PACKING if delayed else None
    static_overlay = None
    if delayed:  # the fixed schedule reads its delayed calendar from its own overlay
        static_overlay = overlay_root / "static"
        write_overlay(STATIC_POLICY, "none", static_overlay, packing=DELAYED_STATIC_PACKING)
    cells: List[Cell] = []
    for scenario in scenarios:
        out_dir = Path(args.out) / f"{args.network}_{args.dist}" / f"{scenario.depot_mode}_seed{scenario.seed}"
        cells.append(_cell(args, network.overrides(), scenario, STATIC_POLICY, "static", static_overlay, out_dir))
        for improver in IMPROVERS:
            overlay = overlay_root / improver
            for policy in ROUTING_POLICIES:
                write_overlay(policy, improver, overlay, packing=packing)
                cells.append(_cell(args, network.overrides(), scenario, policy, improver, overlay, out_dir))
    return cells


def _cell(args, net_overrides, scenario: CampaignScenario, policy, improver, overlay, out_dir: Path) -> Cell:
    overrides = [
        "test_sim",
        f"sim.policies=[{policy}]",
        f"sim.days={args.days}",
        f"sim.data_distribution={args.dist}",
        f"sim.cpu_cores={args.cores}",
        f"sim.output_dir={out_dir}",
        "sim.no_cuda=true",
        *net_overrides,
        *scenario.overrides(),
    ]
    if getattr(args, "initial_fill_days", 0):
        overrides.append(f"sim.initial_fill_days={args.initial_fill_days}")
    env = {"WSR_POLICY_CONFIG_DIR": str(overlay)} if overlay else {}
    return Cell(
        f"{scenario.depot_mode}_{scenario.seed}_{policy}_{improver}",
        [sys.executable, "main.py", *overrides],
        env,
        out_dir,
    )


def run_cell(cell: Cell, log_dir: Path) -> int:
    """Run one cell unless it has finished before.

    Args:
        cell: The cell to run.
        log_dir: Directory of the cell's console log.

    Returns:
        int: The process return code (0 for a skipped cell).
    """
    marker = log_dir / f"{cell.name}.done"
    if marker.exists():
        return 0
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"{cell.name}.log", "w") as log:
        code = subprocess.call(
            cell.argv, cwd=ROOT, env={**os.environ, **cell.env}, stdout=log, stderr=subprocess.STDOUT
        )
    if code == 0:
        marker.touch()
    print(f"[{'ok' if code == 0 else 'FAILED'}] {cell.name}", flush=True)
    return code


def main() -> int:
    """Parse the command line, then run or list the cells.

    Returns:
        int: Number of failed cells.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--network", choices=sorted(NETWORKS), default="rm100")
    parser.add_argument("--dist", choices=("emp", "gamma3"), default="emp")
    parser.add_argument("--depots", choices=("both", "remote", "centroid"), default="both")
    parser.add_argument("--per-depot", type=int, default=5, help="independent scenarios per depot placement")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4, help="cells run at once")
    parser.add_argument("--cores", type=int, default=8, help="processes per cell (sim.cpu_cores)")
    parser.add_argument("--out", default=None, help="output root (default outputs/campaign[_delayed])")
    parser.add_argument(
        "--delayed-start",
        action="store_true",
        help="rerun only TR-k and Regular-k, with the first collection on day k",
    )
    parser.add_argument(
        "--initial-fill-days",
        type=int,
        default=0,
        help="open each bin with the stock of a random 0..W days since its last collection (0: empty)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.out is None:
        args.out = "outputs/campaign"
        if args.initial_fill_days > 0:
            args.out += f"_fill{args.initial_fill_days}"
        if args.delayed_start:
            args.out += "_delayed"
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
        args.out = str(out)
    cells = build_cells(args, out / "overlays")
    if args.dry_run:
        for cell in cells:
            env = " ".join(f"{k}={v}" for k, v in cell.env.items())
            print(f"{env} {shlex.join(cell.argv)}")
        print(f"{len(cells)} cells")
        return 0
    log_dir = out / "logs" / f"{args.network}_{args.dist}"
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(lambda c: run_cell(c, log_dir), cells))
    return sum(code != 0 for code in codes)


if __name__ == "__main__":
    raise SystemExit(main())
