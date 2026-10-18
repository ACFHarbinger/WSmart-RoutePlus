"""
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

import numpy as np
import pandas as pd

if not __package__:
    # Script-style invocation keeps working: python logic/gen/gen_paper_latex.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from logic.gen.paper_data import (
    DEFAULT_SUMMARY_DIR,
    LEARNED_KEY,
    METRIC_GROUP_HEADERS,
    METRIC_GROUP_SUBHEADERS,
    REPO_ROOT,
    ROUTING_SPEC,
    SCENARIO_METRICS,
    SHORTFALL_THRESHOLD,
    STATIC_KEY,
    STRATEGY_SPEC,
    aggregate,
    balance_marginal,
    build_rows,
    display_name,
    drop_affected_cells,
    fmt,
    improver_pairs,
    learned_runs,
    load_horizons,
    mc_list,
    paired_horizon_frame,
    pending_blocks,
    scenario_frame,
    split_degenerate,
    strategy_frame,
    tex_escape,
    text_for,
    unmarked,
)
from logic.gen.paper_figures import (
    fig_fill_trajectory,
    fig_improver_paired,
    fig_networks,
    fig_pareto,
    fig_policy_space,
    fig_scale,
    fig_simulation_loop,
    fig_strategy_tradeoff,
)
from logic.gen.report_utils import apply_theme, load_json, load_theme, render_template

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


def table_routing(
    clean: pd.DataFrame, horizon: int, cfg: dict, learned: pd.DataFrame, static: pd.DataFrame, excluded: bool
) -> str:
    """Per-constructor means and medians over the balanced grid at one horizon."""
    sub = balance_marginal(clean[clean.horizon == horizon], "constructor")
    agg = aggregate(sub, "constructor", ROUTING_SPEC)
    agg = agg.sort_values(("kgkm", "mean"), ascending=False)
    counts = sub.groupby("constructor").size()
    agg.index = [display_name(c, cfg) for c in agg.index]
    lsub = learned[learned.horizon == horizon] if not learned.empty else learned
    blocks = []
    if not lsub.empty:
        lagg = aggregate(lsub, "constructor", ROUTING_SPEC)
        lagg.index = [display_name(c, cfg) for c in lagg.index]
        blocks.append({"title": "", "rows": build_rows(lagg, unmarked(ROUTING_SPEC))})
    ssub = static[static.horizon == horizon] if not static.empty else static
    if not ssub.empty:
        sagg = aggregate(ssub, "strategy", ROUTING_SPEC)
        blocks.append({"title": cfg["block_titles"]["static"], "rows": build_rows(sagg, unmarked(ROUTING_SPEC))})
    else:
        blocks += pending_blocks(cfg, "constructors")
    fields = {
        "horizon": horizon,
        "runs": len(sub),
        "runs_per": int(counts.min()),
        "learned_runs": len(lsub),
        "after_exclusion": " after whole-cell exclusion" if excluded else "",
        "static_runs": int(ssub.groupby("strategy").size().min()) if not ssub.empty else "TODO",
    }
    return render_template(
        "paper_results_table.tex.j2",
        tabcolsep=2,
        label=f"tab:constructors{horizon}",
        caption=text_for(cfg, "captions", "constructors", lsub).format(**fields),
        first_header=cfg["headers"]["constructor"],
        headers=METRIC_GROUP_HEADERS,
        subheaders=METRIC_GROUP_SUBHEADERS,
        column_spec="l" + "r" * len(ROUTING_SPEC),
        ncols=1 + len(ROUTING_SPEC),
        size=r"\footnotesize",
        rows=build_rows(agg, ROUTING_SPEC),
        blocks=blocks,
        note=text_for(cfg, "notes", "constructors", lsub).format(**fields),
    )


def table_strategies(clean: pd.DataFrame, horizon: int, cfg: dict, learned: pd.DataFrame) -> str:
    """Per-strategy-variant means: the efficiency/service trade-off, in one table."""
    agg = strategy_frame(clean, horizon, cfg)
    blocks = []
    if not learned.empty:
        lagg = strategy_frame(learned, horizon, cfg)
        blocks.append({"title": cfg["block_titles"]["learned"], "rows": build_rows(lagg, unmarked(STRATEGY_SPEC))})
    return render_template(
        "paper_results_table.tex.j2",
        tabcolsep=2,
        label=f"tab:strategies{horizon}",
        caption=text_for(cfg, "captions", "strategies", learned).format(horizon=horizon),
        first_header=cfg["headers"]["strategy"],
        headers=METRIC_GROUP_HEADERS,
        subheaders=METRIC_GROUP_SUBHEADERS,
        column_spec="l" + "r" * len(STRATEGY_SPEC),
        ncols=1 + len(STRATEGY_SPEC),
        size=r"\footnotesize",
        rows=build_rows(agg, STRATEGY_SPEC),
        blocks=blocks,
        note=cfg["notes"]["strategies"],
    )


def improver_rows(pairs: pd.DataFrame, cfg: dict) -> list[dict]:
    """One row per metric: CLS and Fast-TSP means, the mean delta, wins and ties."""
    rows = []
    for metric, decimals, better in cfg["improver_metrics"]:
        cls, ftsp = pairs[(metric, "CLS")], pairs[(metric, "FTSP")]
        delta = cls - ftsp
        # Ties are common on overflow counts, and a bare win count silently reads
        # them as losses. Report all three.
        wins = int((delta > 0).sum() if better == "max" else (delta < 0).sum())
        ties = int(np.isclose(delta, 0.0).sum())
        rows.append(
            {
                "label": tex_escape(cfg["metric_labels"][metric]),
                "cells": [
                    fmt(cls.mean(), decimals),
                    fmt(ftsp.mean(), decimals),
                    fmt(delta.mean(), decimals),
                    f"{wins} / {len(pairs) - ties}",
                    f"{ties}",
                ],
            }
        )
    return rows


def table_improvers(clean: pd.DataFrame, horizon: int, cfg: dict, learned: pd.DataFrame) -> str:
    pairs = improver_pairs(clean, horizon)
    blocks = []
    if not learned.empty:
        lpairs = improver_pairs(learned, horizon)
        title = cfg["block_titles"]["learned_pairs"].format(pairs=len(lpairs))
        blocks.append({"title": title, "rows": improver_rows(lpairs, cfg)})
    return render_template(
        "paper_results_table.tex.j2",
        tabcolsep=2,
        label=f"tab:improvers{horizon}",
        caption=text_for(cfg, "captions", "improvers", learned).format(horizon=horizon, pairs=len(pairs)),
        first_header=cfg["headers"]["metric"],
        headers=mc_list(cfg["headers"]["improver"]),
        column_spec="lrrrrr",
        ncols=6,
        rows=improver_rows(pairs, cfg),
        blocks=blocks,
        note=cfg["notes"]["improvers"],
    )


def horizon_rows(paired: pd.DataFrame, cfg: dict) -> list[dict]:
    """One row per routing algorithm: pair count and 30d/90d means of the paired metrics."""
    rows = []
    for constructor, grp in paired.groupby("constructor"):
        rows.append(
            {
                "label": tex_escape(display_name(constructor, cfg)),
                "cells": [
                    f"{len(grp)}",
                    fmt(grp[("kgkm", 30)].mean(), 2),
                    fmt(grp[("kgkm", 90)].mean(), 2),
                    fmt(grp[("overflows", 30)].mean(), 1),
                    fmt(grp[("overflows", 90)].mean(), 1),
                    fmt(grp[("kg_lost", 30)].mean(), 1),
                    fmt(grp[("kg_lost", 90)].mean(), 1),
                ],
            }
        )
    return rows


def table_horizon(paired: pd.DataFrame, cfg: dict, learned_paired: pd.DataFrame, n90: int) -> str:
    """Paired 30d-vs-90d comparison, per routing algorithm, on matched configurations."""
    if paired.empty:
        return ""
    blocks = []
    if not learned_paired.empty:
        blocks.append({"title": "", "rows": horizon_rows(learned_paired, cfg)})
    return render_template(
        "paper_results_table.tex.j2",
        tabcolsep=2,
        label="tab:horizon",
        caption=text_for(cfg, "captions", "horizon", learned_paired).format(pairs=len(paired)),
        first_header=cfg["headers"]["constructor"],
        headers=mc_list(cfg["headers"]["horizon"]),
        column_spec="lrrrrrrr",
        ncols=8,
        size=r"\footnotesize",
        rows=horizon_rows(paired, cfg),
        blocks=blocks,
        note=cfg["notes"]["horizon"].format(pairs=len(paired), n90=n90),
    )


def table_scenarios(clean: pd.DataFrame, horizon: int, cfg: dict, learned: pd.DataFrame) -> str:
    """Distribution and network marginals over like-for-like policy slices."""
    agg, dist_slices, network_slices = scenario_frame(clean, horizon)
    blocks = []
    if not learned.empty:
        lagg, _, _ = scenario_frame(learned, horizon)
        blocks.append({"title": cfg["block_titles"]["learned"], "rows": build_rows(lagg, SCENARIO_METRICS)})
    return render_template(
        "paper_results_table.tex.j2",
        tabcolsep=2,
        label="tab:scenarios30",
        caption=text_for(cfg, "captions", "scenarios", learned).format(horizon=horizon),
        first_header=cfg["headers"]["scenario_factor"],
        headers=mc_list(cfg["headers"]["scenario_metrics"]),
        column_spec="l" + "r" * len(SCENARIO_METRICS),
        ncols=1 + len(SCENARIO_METRICS),
        size=r"\footnotesize",
        rows=build_rows(agg, SCENARIO_METRICS),
        blocks=blocks,
        note=cfg["notes"]["scenarios"].format(dist_slices=dist_slices, network_slices=network_slices),
    )


def table_excluded(degenerate: pd.DataFrame, cfg: dict) -> str:
    """The degenerate runs, reported rather than quietly dropped."""
    if degenerate.empty:
        return ""
    rows = []
    for _, r in degenerate.sort_values("shortfall", ascending=False).iterrows():
        variant_key = f"{r.strategy}{r.cf}{r.sl_var}"
        variant = cfg["variant_labels_short"].get(variant_key, variant_key)
        rows.append(
            {
                "label": tex_escape(display_name(r.constructor, cfg)),
                "cells": [
                    f"{r.horizon:.0f}",
                    # Abbreviated: this is the widest table in the paper and the
                    # full region name plus a slash-joined policy triple overflows
                    # the LNCS text block.
                    tex_escape(f"{''.join(w[0] for w in r.city.split() if w[0].isupper())} {r.N}"),
                    tex_escape(f"{r.dist}/{variant}/{r.improver}"),
                    fmt(r.kg, 0),
                    f"{r.shortfall:.0%}".replace("%", r"\%"),
                    fmt(r.overflows, 0),
                    fmt(r.kg_lost, 1),
                ],
            }
        )
    return render_template(
        "paper_results_table.tex.j2",
        tabcolsep=2,
        label="tab:excluded",
        caption=cfg["captions"]["excluded"],
        first_header=cfg["headers"]["constructor_short"],
        headers=mc_list(cfg["headers"]["excluded"]),
        column_spec="lllrrrrr",
        size=r"\footnotesize",
        rows=rows,
        note=cfg["notes"]["excluded"].format(threshold=int(SHORTFALL_THRESHOLD * 100)),
    )


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
