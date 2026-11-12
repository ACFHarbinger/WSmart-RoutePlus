"""Launch the remote-depot and central-depot benchmark campaign.

For each scenario of ``benchmark_campaign`` (5 deposit realizations per depot placement) this
runs every routing algorithm with each route improver, plus the fixed-schedule baseline, as one
simulator invocation per cell. Each invocation runs the campaign's packing variants through the
overlay policy files (60 s routing limit, no retries, profit-aware operators from the configs).
Finished cells leave a ``.done`` marker and are skipped on a rerun.

With ``--configs`` only the listed (policy, improver, packing variants) triples run, each a ``policy/improver/variant+variant``
entry (``cvrp/none/regular_lvl4`` is the fixed schedule Regular-4), separated by commas; this reruns a chosen policy
set, such as the union of the Pareto fronts on a longer horizon (``--days 90``).

With ``--profit-aware on|off`` every routing algorithm runs with its ``profit_aware_operators`` switch forced
(the profit-operator ablation); ``--packing`` and ``--improvers`` restrict the packing variants and improvers.

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
    uv run python logic/gen/launch_campaign.py --network rm100 --dist emp --days 90 --out outputs/campaign_90 \
        --configs alns/cls/last_minute_cf90+service_level1,cvrp/none/regular_lvl4
    uv run python logic/gen/launch_campaign.py --network rm100 --dist emp --policies alns,aco_hh,hgs,pg_clns \
        --profit-aware off --packing last_minute_cf90,regular_tr_lvl4 --improvers cls --out outputs/campaign_ablation_off
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
    campaign_packing_subset,
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
    names = [p for p in (getattr(args, "packing", None) or "").split(",") if p]
    if names:
        packing = campaign_packing_subset(names)
    improvers = [i for i in (getattr(args, "improvers", None) or "").split(",") if i] or list(IMPROVERS)
    unknown = sorted(set(improvers) - set(IMPROVERS))
    if unknown:
        raise ValueError(f"unknown improvers {unknown}; choose from {sorted(IMPROVERS)}")
    arm = getattr(args, "profit_aware", "configs")
    profit_aware = None if arm == "configs" else arm == "on"
    static_overlay = None
    if delayed:  # the fixed schedule reads its delayed calendar from its own overlay
        static_overlay = overlay_root / "static"
        write_overlay(STATIC_POLICY, "none", static_overlay, packing=DELAYED_STATIC_PACKING)
    configs = [c for c in (getattr(args, "configs", None) or "").split(",") if c]
    if configs:
        return _config_cells(args, overlay_root, scenarios, configs)
    cells: List[Cell] = []
    only = [p for p in (getattr(args, "policies", None) or "").split(",") if p]
    model_path = getattr(args, "model_path", None)
    for scenario in scenarios:
        out_dir = Path(args.out) / f"{args.network}_{args.dist}" / f"{scenario.depot_mode}_seed{scenario.seed}"
        if not only:  # a policy subset leaves the fixed schedule to the full campaign
            cells.append(_cell(args, network.overrides(), scenario, STATIC_POLICY, "static", static_overlay, out_dir))
        for improver in improvers:
            overlay = overlay_root / improver
            for policy in only or ROUTING_POLICIES:
                write_overlay(
                    policy, improver, overlay, packing=packing, model_path=model_path, profit_aware=profit_aware
                )
                cells.append(_cell(args, network.overrides(), scenario, policy, improver, overlay, out_dir))
    return cells


def _config_cells(args: argparse.Namespace, overlay_root: Path, scenarios, configs: List[str]) -> List[Cell]:
    """Cells of explicit ``policy/improver/variant+variant`` triples, on every scenario.

    Args:
        args: Parsed command line.
        overlay_root: Directory that receives one overlay folder per policy and improver.
        scenarios: The campaign scenarios to run.
        configs: The triples.

    Returns:
        List[Cell]: One cell per scenario and triple.

    Raises:
        ValueError: If a triple is malformed or names an unknown policy, improver or variant.
    """
    network = NETWORKS[args.network]
    model_path = getattr(args, "model_path", None)
    arm = getattr(args, "profit_aware", "configs")
    profit_aware = None if arm == "configs" else arm == "on"
    jobs = []
    for spec in configs:
        parts = spec.split("/")
        if len(parts) != 3 or not parts[2]:
            raise ValueError(f"--configs entry {spec!r} is not policy/improver/variant+variant")
        policy, improver, variants = parts[0], parts[1], parts[2].split("+")
        if improver not in IMPROVERS:
            raise ValueError(f"unknown improver {improver!r} in {spec!r}; choose from {sorted(IMPROVERS)}")
        overlay = overlay_root / f"{policy}_{improver}"
        if policy == STATIC_POLICY:
            if improver != "none":
                raise ValueError("the fixed schedule runs without route improvement")
            write_overlay(STATIC_POLICY, "none", overlay, packing=[{"other/packing/pk_regular.yaml": variants}])
        elif policy in ROUTING_POLICIES:
            write_overlay(
                policy,
                improver,
                overlay,
                packing=campaign_packing_subset(variants),
                model_path=model_path,
                profit_aware=profit_aware,
            )
        else:
            raise ValueError(f"unknown policy {policy!r} in {spec!r}")
        jobs.append((policy, improver, overlay))
    cells: List[Cell] = []
    for scenario in scenarios:
        out_dir = Path(args.out) / f"{args.network}_{args.dist}" / f"{scenario.depot_mode}_seed{scenario.seed}"
        for policy, improver, overlay in jobs:
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
    parser.add_argument("--policies", default="", help="comma-separated policy names to run alone (e.g. na)")
    parser.add_argument("--model-path", default=None, help="trained model directory for the na policy")
    parser.add_argument(
        "--profit-aware",
        choices=("configs", "on", "off"),
        default="configs",
        help="force every profit_aware_operators switch (profit-operator ablation); configs keeps the policy files",
    )
    parser.add_argument(
        "--configs",
        default="",
        help="comma-separated policy/improver/variant+variant triples to run alone (cvrp/none/regular_lvl4: Regular-4)",
    )
    parser.add_argument("--packing", default="", help="comma-separated campaign packing variants to run (default all)")
    parser.add_argument("--improvers", default="", help="comma-separated improvers to run (default all)")
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
