"""

Table builders live in ``paper_latex_tables``; this module keeps the CLI and re-exports those names.
Generate the LaTeX tables and figures for the MPVRPP paper.

Sibling of gen_simulation_analysis.py (markdown reports) and gen_presentation.py
(PPTX deck): same summary-CSV schema, same theme/style machinery, same Jinja
conventions -- but emitting ``.tex`` fragments that ``paper.tex`` \\input's, plus
the figures those fragments reference, written straight into the paper's own
``Images/Results/`` tree at the sizes the document uses.

The point is that no number in the paper is typed by hand. Every table cell and
every figure here is derived from
``docs/private/global/simulation/simulation_summary.csv`` (30-day horizon) and
``simulation_summary_90d.csv`` (90-day), so the paper cannot silently drift from
the data the way a hand-maintained table does.

Two properties of that data are enforced centrally, in this module, so that the
paper, the markdown reports and the website cannot disagree about them:

**Degenerate runs are excluded from aggregates.** The simulator gives every
routing algorithm the same arrival sequence within a scenario cell. Collectable mass
is not fixed: waste above capacity is lost, so timing changes the kilograms
that remain. Runs that collect drastically less are anomalous outcomes. The
30-day logs in question still contain every day; they stop collecting
mid-horizon rather than ending early.
``find_degenerate_runs`` flags rows whose collected tonnage falls more than
``SHORTFALL_THRESHOLD`` below their cell median. The threshold is not arbitrary:
the shortfall distribution has median 0.0 and rises to only 0.071 outside the
runs it catches, which sit at 0.30, 0.47, 0.47 and 0.86. The gap is the signal.
Excluded rows are reported in their own table rather than dropped silently --
they are a real and citable scale limit of the two-commodity-flow MIP.

``drop_affected_cells`` then removes the *entire* scenario cell each degenerate
run belonged to, for every routing algorithm. Dropping only the offending rows would
reintroduce the same selection bias this module refuses to accept in the 90-day
data: all four degenerate runs are SWC-TCF's, so removing just them would
average SWC-TCF over the scenarios where it did not fail while its rivals are
averaged over those too. Losing the cell for everyone costs rows but keeps every
routing algorithm's mean over an identical set of scenarios, which is the only thing
that makes the marginal means comparable.

``balance_marginal`` handles the same problem for every *other* stage or
scenario factor. Because
``constructor`` is absent from ``CELL_KEYS``, dropping a cell removes all
routing algorithms together and the routing algorithm marginal stays balanced for free. Any
other stage -- selection variant, improver -- is *inside* ``CELL_KEYS``, so
dropping a cell removes one level of that stage and leaves its siblings intact.
That is how the selection-strategy table came to average Look-Ahead over a
scenario set with the study's hardest cell deleted while Last-Minute and
Service-Level (SL1) kept theirs, which made Look-Ahead look better on service
than it is. Marginal tables therefore restrict to the slices every level of the
compared stage actually shares.

Note that the ``days`` column counts *collection* days, not elapsed days, so a
low value there is not evidence of truncation on its own: several of the best
policies collect on 15 of 30 days precisely because they bundle well. Tonnage is
the honest signal.

**The 90-day selection rule is pending recovery.** The archive is a
non-factorial subset and does not reproduce Pareto-only carry-forward.
The horizon table pairs each observed configuration against itself;
it does not rank routing algorithms across differently composed subsets or
estimate a population-level horizon effect.

Non-Python content lives in sibling directories, as with the other generators:
  jinja/paper_*.tex.j2       LaTeX fragment templates
  json/paper_latex_config.json  labels, captions, defaults
  json/simulation_metadata.json colour palettes (shared)
  style/{dark,light}.mplstyle   matplotlib style sheets

Idempotent: will not overwrite existing output unless --force is passed.

Usage
-----
    uv run python logic/gen/gen_paper_latex.py --force
    uv run python logic/gen/gen_paper_latex.py \\
        --horizon 30=docs/private/global/simulation/simulation_summary.csv \\
        --horizon 90=docs/private/global/simulation/simulation_summary_90d.csv \\
        --tables-dir assets/papers/<paper>/Tables \\
        --figures-dir assets/papers/<paper>/Images/Results/Generated \\
        --theme light --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if not __package__:
    # Script-style invocation keeps working: python logic/gen/gen_paper_latex.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from logic.gen.paper_data import (
    DEFAULT_SUMMARY_DIR,
    LEARNED_KEY,
    REPO_ROOT,
    STATIC_KEY,
    drop_affected_cells,
    learned_runs,
    load_horizons,
    paired_horizon_frame,
    split_degenerate,
)
from logic.gen.paper_diagrams import fig_policy_space, fig_simulation_loop
from logic.gen.paper_figures import (
    fig_fill_trajectory,
    fig_improver_paired,
    fig_pareto,
    fig_scale,
    fig_strategy_tradeoff,
)
from logic.gen.paper_networks import fig_networks
from logic.gen.report_utils import apply_theme, load_json, load_theme

from .paper_latex_tables import (
    table_excluded as table_excluded,
)
from .paper_latex_tables import (
    table_horizon as table_horizon,
)
from .paper_latex_tables import (
    table_improvers as table_improvers,
)
from .paper_latex_tables import (
    table_routing as table_routing,
)
from .paper_latex_tables import (
    table_scenarios as table_scenarios,
)
from .paper_latex_tables import (
    table_strategies as table_strategies,
)

# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Data integrity
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# LaTeX helpers
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Network geometry
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_horizons(values: list[str] | None) -> dict[int, Path]:
    if not values:
        return {
            30: DEFAULT_SUMMARY_DIR / "simulation_summary.csv",
            90: DEFAULT_SUMMARY_DIR / "simulation_summary_90d.csv",
        }
    out = {}
    for item in values:
        days, _, csv = item.partition("=")
        if not csv:
            raise SystemExit(f"--horizon expects DAYS=CSV, got {item!r}")
        out[int(days)] = Path(csv)
    return out


def write(path: Path, text: str, force: bool) -> None:
    if not text:
        return
    if path.exists() and not force:
        print(f"  Skipped (exists, use --force): {path.name}")
        return
    path.write_text(text, encoding="utf-8")
    print(f"  Wrote: {path.name}")


def main() -> None:
    cfg = load_json("paper_latex_config.json")
    default_paper = REPO_ROOT / cfg["paths"]["paper_dir"]

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--horizon",
        action="append",
        metavar="DAYS=CSV",
        help="summary CSV for one horizon; repeatable (default: 30d and 90d)",
    )
    ap.add_argument("--tables-dir", type=Path, default=default_paper / cfg["paths"]["tables_subdir"])
    ap.add_argument("--figures-dir", type=Path, default=default_paper / cfg["paths"]["figures_subdir"])
    ap.add_argument(
        "--theme",
        choices=["dark", "light"],
        default="light",
        help="light matches the paper's white background (default)",
    )
    ap.add_argument(
        "--primary-horizon", type=int, default=30, help="horizon whose grid is complete and therefore leads the results"
    )
    ap.add_argument("--force", action="store_true", help="overwrite existing output")
    ap.add_argument("--tables-only", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()

    theme = load_theme(args.theme)
    apply_theme(theme)
    colors = dict(load_json("simulation_metadata.json").get("constructor_colors", {}))

    print("Loading simulation summaries:")
    raw = load_horizons(parse_horizons(args.horizon))

    print("Checking data integrity:")
    learned_raw = raw[raw.constructor == LEARNED_KEY]
    static = raw[raw.constructor == STATIC_KEY]
    raw = raw[~raw.constructor.isin([LEARNED_KEY, STATIC_KEY])]
    clean, degenerate = split_degenerate(raw)
    clean = drop_affected_cells(clean, degenerate)
    paired = paired_horizon_frame(clean)
    print(f"  {len(clean)} runs retained; {len(paired)} configurations present at both horizons")
    learned = learned_runs(learned_raw, raw, degenerate)
    learned_paired = paired_horizon_frame(learned) if not learned.empty else learned

    horizon = args.primary_horizon

    if not args.figures_only:
        args.tables_dir.mkdir(parents=True, exist_ok=True)
        print(f"Writing LaTeX tables to {args.tables_dir}:")
        write(
            args.tables_dir / "results_routing.tex",
            table_routing(clean, horizon, cfg, learned, static, not degenerate.empty),
            args.force,
        )
        write(args.tables_dir / "results_strategies.tex", table_strategies(clean, horizon, cfg, learned), args.force)
        write(args.tables_dir / "results_improvers.tex", table_improvers(clean, horizon, cfg, learned), args.force)
        write(
            args.tables_dir / "results_horizon.tex",
            table_horizon(paired, cfg, learned_paired, int((raw.horizon == 90).sum())),
            args.force,
        )
        write(args.tables_dir / "results_scenarios.tex", table_scenarios(clean, horizon, cfg, learned), args.force)
        excluded = table_excluded(degenerate, cfg)
        if excluded:
            write(args.tables_dir / "results_excluded.tex", excluded, args.force)
        elif (args.tables_dir / "results_excluded.tex").exists() and args.force:
            # No run is excluded: a table left from an earlier vintage would contradict the paper.
            (args.tables_dir / "results_excluded.tex").unlink()
            print("  Removed: results_excluded.tex (no degenerate runs)")

    if not args.tables_only:
        args.figures_dir.mkdir(parents=True, exist_ok=True)
        print(f"Writing figures to {args.figures_dir}:")
        fig_policy_space(args.figures_dir, cfg)
        fig_simulation_loop(args.figures_dir, cfg)
        fig_networks(args.figures_dir, cfg)
        fig_pareto(clean, horizon, args.figures_dir, colors, cfg)
        fig_strategy_tradeoff(clean, horizon, args.figures_dir, cfg)
        fig_improver_paired(clean, horizon, args.figures_dir, cfg)
        fig_scale(clean, horizon, args.figures_dir, colors, cfg)
        fig_fill_trajectory(args.figures_dir)

    print("Done.")


if __name__ == "__main__":
    main()
