"""LaTeX table builders for the MPVRPP paper.

Moved from ``logic.gen.gen_paper_latex``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if not __package__:
    # Script-style invocation keeps working: python logic/gen/gen_paper_latex.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from logic.gen.paper_data import (
    METRIC_GROUP_HEADERS,
    METRIC_GROUP_SUBHEADERS,
    ROUTING_SPEC,
    SCENARIO_METRICS,
    SHORTFALL_THRESHOLD,
    STRATEGY_SPEC,
    aggregate,
    balance_marginal,
    build_rows,
    display_name,
    fmt,
    improver_pairs,
    mc_list,
    pending_blocks,
    scenario_frame,
    strategy_frame,
    tex_escape,
    text_for,
    unmarked,
)
from logic.gen.report_utils import render_template


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


def matched_text(pairs: int, n90: int) -> str:
    """The sentence of the horizon note that says how many 90-day runs have a 30-day match."""
    if pairs == n90:
        return f"All {n90} classical 90-day runs have a matched 30-day counterpart."
    return f"{pairs} of the {n90} classical 90-day runs have a matched 30-day counterpart."


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
        note=cfg["notes"]["horizon"].format(matched=matched_text(len(paired), n90)),
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
