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
constructor the same arrival sequence within a scenario cell. Collectable mass
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
run belonged to, for every constructor. Dropping only the offending rows would
reintroduce the same selection bias this module refuses to accept in the 90-day
data: all four degenerate runs are SWC-TCF's, so removing just them would
average SWC-TCF over the scenarios where it did not fail while its rivals are
averaged over those too. Losing the cell for everyone costs rows but keeps every
constructor's mean over an identical set of scenarios, which is the only thing
that makes the marginal means comparable.

``balance_marginal`` handles the same problem for every *other* stage or
scenario factor. Because
``constructor`` is absent from ``CELL_KEYS``, dropping a cell removes all
constructors together and the constructor marginal stays balanced for free. Any
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
it does not rank constructors across differently composed subsets or
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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from report_utils import apply_theme, load_json, load_theme, render_template, savefig

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Raw simulation output tree, source of the road-distance matrices.
OUTPUT_DIR = REPO_ROOT / "assets" / "output" / "30days"

#: Retained coordinate-derived maps produced by gen_simulation_analysis.py from
#: the original (gitignored) bin-coordinate exports and OpenStreetMap roads.
NETWORK_MAP_DIR = REPO_ROOT / "docs" / "private" / "figures" / "simulation" / "30d"

#: Collected-tonnage shortfall, relative to the scenario-cell median, above which
#: a run is treated as degenerate rather than merely bad. See the module
#: docstring for why 0.20 sits in a genuine gap in the distribution.
SHORTFALL_THRESHOLD = 0.20

#: Minimum rows in a cell before its median means anything: two, so that there is
#: at least one comparator. It was 4, which silently exempted the sparse 90-day
#: cells from the check entirely and let through the single worst run in the
#: study -- SWC-TCF at 90 days on Figueira da Foz N=350 under Gamma-3, which
#: reached day 13 of 90 and recorded 23,886 overflow events at an 85.6% tonnage
#: shortfall. At 2 the rule flags exactly four runs across both horizons and the
#: next-highest shortfall in the data is 0.071, so the gap remains unambiguous.
MIN_CELL_SIZE = 2

#: The columns that identify a scenario cell: everything except the constructor,
#: so that the constructors within a cell are exactly the competitors that faced
#: the identical demand realisation.
CELL_KEYS = ["horizon", "city", "N", "dist", "strategy", "cf", "sl_var", "improver"]

#: The scenario coordinates, independent of any policy stage. A marginal
#: comparison across one stage must hold these fixed; see balance_marginal.
SCENARIO_KEYS = ["horizon", "city", "N", "dist"]

#: The policy stages that can be compared marginally, and the columns that
#: identify a level of each.
STAGE_KEYS = {
    "constructor": ["constructor"],
    "variant": ["strategy", "cf", "sl_var"],
    "improver": ["improver"],
    "dist": ["dist"],
    "network": ["city", "N"],
}

#: The columns that identify one policy configuration across horizons.
CONFIG_KEYS = ["city", "N", "dist", "improver", "strategy", "cf", "sl_var", "acceptance", "constructor"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


#: Columns that are populated only for the strategies that have variants (``cf``
#: for Last-Minute, ``sl_var`` for Service-Level) and are empty for Look-Ahead.
#: They are normalised to "" rather than left as NaN because pandas drops NaN
#: keys from groupby and pivot indices, which would silently delete every
#: Look-Ahead run from the aggregates.
OPTIONAL_KEYS = ["cf", "sl_var", "acceptance"]


def load_horizons(horizons: dict[int, Path]) -> pd.DataFrame:
    """Load one summary CSV per horizon into a single frame tagged with ``horizon``."""
    frames = []
    for days, csv in sorted(horizons.items()):
        if not csv.exists():
            raise SystemExit(f"Missing summary CSV for horizon {days}d: {csv}")
        df = pd.read_csv(csv)
        df["horizon"] = days
        frames.append(df)
        print(f"  Loaded {len(df):>4} runs at {days:>3}d from {csv}")
    combined = pd.concat(frames, ignore_index=True)
    for col in OPTIONAL_KEYS:
        combined[col] = combined[col].fillna("").astype(str)
    return combined


# --------------------------------------------------------------------------- #
# Data integrity
# --------------------------------------------------------------------------- #


def find_degenerate_runs(df: pd.DataFrame) -> pd.Series:
    """
    Boolean mask of runs that failed to complete rather than merely performing badly.

    Within a scenario cell every constructor faces the identical demand
    realisation, so collected tonnage is comparable by construction. A run far
    below its cell median did not collect the waste that was there to collect.
    """
    grouped = df.groupby(CELL_KEYS, dropna=False)["kg"]
    cell_median = grouped.transform("median")
    cell_size = grouped.transform("size")
    shortfall = 1.0 - df["kg"] / cell_median
    return (cell_size >= MIN_CELL_SIZE) & (shortfall > SHORTFALL_THRESHOLD)


def split_degenerate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(clean, degenerate)``, both with a ``shortfall`` column added."""
    grouped = df.groupby(CELL_KEYS, dropna=False)["kg"]
    df = df.assign(shortfall=1.0 - df["kg"] / grouped.transform("median"))
    mask = find_degenerate_runs(df)
    clean, degenerate = df[~mask].copy(), df[mask].copy()
    if len(degenerate):
        print(f"  Excluded {len(degenerate)} degenerate run(s) from aggregates:")
        for _, r in degenerate.iterrows():
            print(
                f"    {r.constructor:<8} {r.horizon:>3}d {r.city} N={r.N} {r.dist} "
                f"{r.strategy}{r.cf}{r.sl_var:<4} {r.improver}: "
                f"collected {r.kg:,.0f} kg ({r.shortfall:.0%} below cell median)"
            )
    return clean, degenerate


def drop_affected_cells(df: pd.DataFrame, degenerate: pd.DataFrame) -> pd.DataFrame:
    """
    Remove every scenario cell that contained a degenerate run, for all constructors.

    Dropping only the degenerate rows would reintroduce exactly the bias this
    module refuses to accept for the 90-day sample. All four degenerate runs
    belong to SWC-TCF, so removing them alone would leave SWC-TCF averaged over
    the scenarios where it did *not* fail while every rival is averaged over
    those scenarios too -- a mean quietly computed on a favourable subset.

    Removing the whole cell costs the other constructors those scenarios as well,
    but keeps every constructor's mean over an identical set of scenarios, which
    is the only reason the marginal means are comparable in the first place.
    """
    if degenerate.empty:
        return df
    affected = set(map(tuple, degenerate[CELL_KEYS].to_numpy()))
    keep = ~df[CELL_KEYS].apply(lambda row: tuple(row) in affected, axis=1)
    dropped = int((~keep).sum())
    print(
        f"  Dropped {dropped} further run(s) so that every constructor is averaged "
        f"over the same {len(affected)} fewer scenario cells"
    )
    return df[keep].copy()


def balance_marginal(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    """
    Restrict ``df`` to the slices in which *every* level of ``stage`` is present.

    ``drop_affected_cells`` balances the constructor marginal only, because
    ``constructor`` is absent from ``CELL_KEYS`` and so all constructors in a
    cell are removed together. Any other stage is inside ``CELL_KEYS``, so
    dropping a cell removes one level of that stage and leaves its siblings
    intact --- which is how the selection-strategy table came to average
    Look-Ahead over a scenario set with the study's hardest cell deleted while
    its rivals kept theirs.

    A marginal comparison across a stage is only meaningful when every level
    faced the same scenarios, so this keeps the slices --- scenario coordinates
    plus the *other* policy stages --- that survive for all levels of the stage
    being compared, and drops the rest for everyone.
    """
    stage_cols = STAGE_KEYS[stage]
    policy_cols = {column for columns in STAGE_KEYS.values() for column in columns}
    slice_cols = [
        c
        for c in df.columns
        if c in set(SCENARIO_KEYS) | policy_cols and c not in stage_cols
    ]
    levels = df[stage_cols].drop_duplicates()
    common: set | None = None
    for _, level in levels.iterrows():
        mask = (df[stage_cols] == level.values).all(axis=1)
        present = set(map(tuple, df.loc[mask, slice_cols].drop_duplicates().to_numpy()))
        common = present if common is None else common & present
    if not common:
        return df
    keep = df[slice_cols].apply(lambda row: tuple(row) in common, axis=1)
    dropped = int((~keep).sum())
    if dropped:
        print(f"  Balancing the {stage} marginal: dropped {dropped} run(s) from slices not shared by all levels")
    return df[keep].copy()


def paired_horizon_frame(clean: pd.DataFrame) -> pd.DataFrame:
    """
    Configurations observed at both horizons, as one row per configuration.

    This is the horizon contrast least distorted by the 90-day sampling, because
    each configuration is compared against *itself* rather than against a rival
    whose 90-day scenarios are a different, self-favouring subset.

    It does not identify a horizon effect for the full design. A configuration
    reached 90 days precisely because its 30-day result put it on the Pareto
    front, and there are no replicated seeds from which to estimate the
    resulting selection effect. The paired difference is therefore descriptive
    of the selected configurations only.
    """
    wide = clean.pivot_table(index=CONFIG_KEYS, columns="horizon", values=["kgkm", "overflows", "km", "time"])
    wide = wide.dropna()
    if wide.empty:
        return wide
    return wide.reset_index()


# --------------------------------------------------------------------------- #
# LaTeX helpers
# --------------------------------------------------------------------------- #


def display_name(raw: str, cfg: dict) -> str:
    """Map a CSV constructor key to the name the paper uses (ACO_HH -> ACO-HH)."""
    return cfg.get("constructor_labels", {}).get(raw, raw)


def tex_escape(s: object) -> str:
    """Escape the LaTeX specials that appear in this data (constructor names, mostly)."""
    out = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        out = out.replace(a, b)
    return out


def fmt(value: float, decimals: int = 2, *, best: bool = False) -> str:
    """Format one table cell, bolding it when it is the best in its column."""
    if pd.isna(value):
        return "--"
    text = f"{value:,.{decimals}f}" if decimals else f"{value:,.0f}"
    return rf"\textbf{{{text}}}" if best else text


def mc(text: str, span: int = 1) -> dict:
    """One header cell for ``paper_results_table.tex.j2``, rendered as a centered
    ``\\multicolumn`` regardless of the body column's own left/right alignment."""
    return {"text": text, "span": span}


def mc_list(texts: list[str]) -> list[dict]:
    """Wrap a flat list of plain header strings into ordinary (span=1) header cells."""
    return [mc(t) for t in texts]


def aggregate(sub: pd.DataFrame, by: str, spec: list[tuple[str, str, int, str]]) -> pd.DataFrame:
    """Aggregate ``sub`` by one key into the (column, aggfunc) pairs a spec asks for."""
    out = pd.DataFrame(index=sorted(sub[by].unique()))
    for col, how, _, _ in spec:
        if col == "n":
            out[(col, how)] = sub.groupby(by).size()
        else:
            out[(col, how)] = sub.groupby(by)[col].agg(how)
    return out


def build_rows(frame: pd.DataFrame, spec: list[tuple[str, str, int, str]]) -> list[dict]:
    """
    Turn an aggregated frame into template-ready rows, marking the best cell per column.

    ``spec`` is ``(column, aggfunc, decimals, direction)`` where direction is
    ``"max"`` or ``"min"``; ``"none"`` disables best-marking for columns where
    "best" is meaningless (a run count, or a descriptive quantity like the number
    of collections, which is neither good nor bad on its own).
    """
    best_of = {}
    for col, how, _, direction in spec:
        if direction in ("max", "min"):
            series = frame[(col, how)]
            best_of[(col, how)] = series.max() if direction == "max" else series.min()
    rows = []
    for label, row in frame.iterrows():
        cells = []
        for col, how, dec, _ in spec:
            key = (col, how)
            value = row[key]
            is_best = key in best_of and np.isclose(value, best_of[key])
            cells.append(fmt(value, dec, best=is_best))
        rows.append({"label": tex_escape(label), "cells": cells})
    return rows


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

#: Means are reported beside medians for the two headline metrics. A single run
#: out of ~96 moved Service-Level's mean overflow count from 9.3 to 1.4, so the
#: means here are genuinely fragile to individual scenarios and a median-based
#: reading is the honest companion, not a footnote.
CONSTRUCTOR_SPEC = [
    ("n", "size", 0, "none"),
    ("kgkm", "mean", 2, "max"),
    ("kgkm", "median", 2, "max"),
    ("overflows", "mean", 1, "min"),
    ("overflows", "median", 1, "min"),
    ("km", "mean", 0, "min"),
    ("time", "mean", 0, "min"),
]

STRATEGY_SPEC = CONSTRUCTOR_SPEC

#: Header cells matching CONSTRUCTOR_SPEC/STRATEGY_SPEC's seven columns: the two
#: headline metrics span a mean/median pair under one centered label, with the
#: mean/median distinction carried by a second header row rather than by a
#: literal "med." column label competing with the numbers for space.
METRIC_GROUP_HEADERS = [mc("Runs"), mc("kg/km", 2), mc("Overflows", 2), mc("km"), mc("Time (s)")]
METRIC_GROUP_SUBHEADERS = ["", "mean", "median", "mean", "median", "", ""]


def table_constructors(clean: pd.DataFrame, horizon: int, cfg: dict) -> str:
    """Per-constructor means and medians over the balanced grid at one horizon."""
    sub = balance_marginal(clean[clean.horizon == horizon], "constructor")
    agg = aggregate(sub, "constructor", CONSTRUCTOR_SPEC)
    agg = agg.sort_values(("kgkm", "mean"), ascending=False)
    counts = sub.groupby("constructor").size()
    agg.index = [display_name(c, cfg) for c in agg.index]
    return render_template(
        "paper_results_table.tex.j2",
        label=f"tab:constructors{horizon}",
        caption=cfg["captions"]["constructors"].format(horizon=horizon, runs=len(sub)),
        first_header=cfg["headers"]["constructor"],
        headers=METRIC_GROUP_HEADERS,
        subheaders=METRIC_GROUP_SUBHEADERS,
        column_spec="l" + "r" * len(CONSTRUCTOR_SPEC),
        rows=build_rows(agg, CONSTRUCTOR_SPEC),
        note=cfg["notes"]["constructors"].format(runs_per=int(counts.min())),
    )


def table_strategies(clean: pd.DataFrame, horizon: int, cfg: dict) -> str:
    """Per-strategy-variant means: the efficiency/service trade-off, in one table."""
    sub = balance_marginal(clean[clean.horizon == horizon], "variant").copy()
    sub["variant"] = sub.strategy + sub.cf + sub.sl_var
    order = [v for v in cfg["variant_order"] if v in set(sub.variant)]
    agg = aggregate(sub, "variant", STRATEGY_SPEC).reindex(order)
    agg.index = [cfg["variant_labels"].get(v, v) for v in agg.index]
    return render_template(
        "paper_results_table.tex.j2",
        label=f"tab:strategies{horizon}",
        caption=cfg["captions"]["strategies"].format(horizon=horizon),
        first_header=cfg["headers"]["strategy"],
        headers=METRIC_GROUP_HEADERS,
        subheaders=METRIC_GROUP_SUBHEADERS,
        column_spec="l" + "r" * len(STRATEGY_SPEC),
        size=r"\footnotesize",
        rows=build_rows(agg, STRATEGY_SPEC),
        note=cfg["notes"]["strategies"],
    )


def improver_pairs(clean: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    CLS against Fast-TSP on the configurations that differ *only* in improver.

    Pairing rather than averaging controls scenario mix: every pair has the same
    constructor, selection strategy, scenario and demand realisation. It does
    not isolate a causal improver effect, because stochastic upstream
    construction outcomes are not identical in every pair.
    """
    sub = clean[clean.horizon == horizon]
    keys = [k for k in CONFIG_KEYS if k != "improver"]
    wide = sub.pivot_table(index=keys, columns="improver", values=["kgkm", "km", "overflows", "time"])
    return wide.dropna()


def table_improvers(clean: pd.DataFrame, horizon: int, cfg: dict) -> str:
    pairs = improver_pairs(clean, horizon)
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
    return render_template(
        "paper_results_table.tex.j2",
        label=f"tab:improvers{horizon}",
        caption=cfg["captions"]["improvers"].format(horizon=horizon, pairs=len(pairs)),
        first_header=cfg["headers"]["metric"],
        headers=mc_list(cfg["headers"]["improver"]),
        column_spec="lrrrrr",
        rows=rows,
        note=cfg["notes"]["improvers"],
    )


def table_horizon(paired: pd.DataFrame, cfg: dict) -> str:
    """Paired 30d-vs-90d comparison, per constructor, on matched configurations."""
    if paired.empty:
        return ""
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
                ],
            }
        )
    return render_template(
        "paper_results_table.tex.j2",
        label="tab:horizon",
        caption=cfg["captions"]["horizon"].format(pairs=len(paired)),
        first_header=cfg["headers"]["constructor"],
        headers=mc_list(cfg["headers"]["horizon"]),
        column_spec="lrrrrr",
        rows=rows,
        note=cfg["notes"]["horizon"],
    )


def table_scenarios(clean: pd.DataFrame, horizon: int, cfg: dict) -> str:
    """Distribution and network marginals over like-for-like policy slices."""
    metrics = [
        ("n", "size", 0, "none"),
        ("kgkm", "mean", 2, "none"),
        ("overflows", "mean", 1, "none"),
        ("km", "mean", 0, "none"),
        ("time", "mean", 0, "none"),
    ]
    primary = clean[clean.horizon == horizon]
    dist = balance_marginal(primary, "dist").copy()
    dist["level"] = dist["dist"].map(lambda value: f"Demand: {value}")
    network = balance_marginal(primary, "network").copy()
    network["level"] = network.apply(
        lambda row: f"Network: {row.city} ($N={row.N}$)", axis=1
    )
    frame = pd.concat([dist, network], ignore_index=True)
    agg = aggregate(frame, "level", metrics)
    order = [
        "Demand: Empirical",
        "Demand: Gamma-3",
        "Network: Rio Maior ($N=100$)",
        "Network: Rio Maior ($N=170$)",
        "Network: Figueira da Foz ($N=350$)",
    ]
    agg = agg.reindex(order)
    return render_template(
        "paper_results_table.tex.j2",
        label="tab:scenarios30",
        caption=cfg["captions"]["scenarios"].format(horizon=horizon),
        first_header=cfg["headers"]["scenario_factor"],
        headers=mc_list(cfg["headers"]["scenario_metrics"]),
        column_spec="l" + "r" * len(metrics),
        size=r"\footnotesize",
        rows=build_rows(agg, metrics),
        note=cfg["notes"]["scenarios"],
    )


def table_excluded(degenerate: pd.DataFrame, cfg: dict) -> str:
    """The degenerate runs, reported rather than quietly dropped."""
    if degenerate.empty:
        return ""
    rows = []
    for _, r in degenerate.sort_values("shortfall", ascending=False).iterrows():
        variant = f"{r.strategy}{r.cf}{r.sl_var}"
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
                ],
            }
        )
    return render_template(
        "paper_results_table.tex.j2",
        label="tab:excluded",
        caption=cfg["captions"]["excluded"],
        first_header=cfg["headers"]["constructor_short"],
        headers=mc_list(cfg["headers"]["excluded"]),
        column_spec="lllrrrr",
        size=r"\footnotesize",
        rows=rows,
        note=cfg["notes"]["excluded"].format(threshold=int(SHORTFALL_THRESHOLD * 100)),
    )


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def pareto_front(points: pd.DataFrame) -> pd.DataFrame:
    """Non-dominated set: maximise kg/km, minimise overflows."""
    ordered = points.sort_values("kgkm", ascending=False)
    best, keep = np.inf, []
    for idx, row in ordered.iterrows():
        if row.overflows < best:
            keep.append(idx)
            best = row.overflows
    return points.loc[keep].sort_values("kgkm")

def fig_policy_space(out_dir: Path, cfg: dict) -> None:
    """
    The three-stage policy configuration space, as Fig. 1 of the paper.

    Drawn in figure coordinates on an invisible axes with explicit limits. The
    limits matter: a previous version placed the legend keys with single-point
    ``ax.plot`` calls, and a one-point dataset collapses the autoscaled axes to a
    sliver around that point. Everything else then sat outside the axes and
    ``bbox_inches="tight"`` grew the canvas to contain it, producing a 1215x35753
    image. Set the limits, and never let a stray artist drive them.
    """
    import matplotlib.patches as mpatches

    stages = cfg["policy_space"]["stages"]
    counts = cfg["policy_space"]["benchmark_counts"]

    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    centres = [0.175, 0.5, 0.825]
    box_w, box_top, box_bot = 0.29, 0.86, 0.40

    for centre, stage in zip(centres, stages, strict=True):
        colour = stage["color"]
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (centre - box_w / 2, box_bot), box_w, box_top - box_bot,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                facecolor=stage["fill"], edgecolor=colour, linewidth=1.4, zorder=2,
            )
        )
        ax.text(centre, box_top - 0.07, stage["title"], ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=colour, zorder=3)
        ax.text(centre, box_top - 0.135, stage["scope"], ha="center", va="center",
                fontsize=7.8, style="italic", color="#5a6673", zorder=3)
        ax.text(centre, (box_top + box_bot) / 2 - 0.055, "\n".join(stage["members"]),
                ha="center", va="center", fontsize=8.6, color="#26313d", zorder=3,
                linespacing=1.5)
        ax.text(centre, box_bot + 0.045, stage["footer"].format(**counts),
                ha="center", va="center", fontsize=7.6, color="#5a6673", zorder=3)

    for a, b in zip(centres, centres[1:], strict=False):
        ax.annotate("", xy=(b - box_w / 2 - 0.008, (box_top + box_bot) / 2),
                    xytext=(a + box_w / 2 + 0.008, (box_top + box_bot) / 2),
                    arrowprops=dict(arrowstyle="-|>", linewidth=1.6, color="#94a2b0"),
                    zorder=1)

    ax.text(0.5, 0.20, cfg["policy_space"]["benchmarked"].format(**counts),
            ha="center", va="center", fontsize=9.2, color="#26313d", zorder=3,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#eef2f5", edgecolor="#d5dde4"))
    ax.text(0.5, 0.045, cfg["policy_space"]["footnote"].format(**counts),
            ha="center", va="center", fontsize=7.8, color="#5a6673", zorder=3)

    savefig(fig, out_dir / "policy_configuration_space.png")


def fig_simulation_loop(out_dir: Path, cfg: dict) -> None:
    """
    The multi-period daily simulation loop architecture, as a figure.

    Visualises the daily cycle: stochastic waste accumulation, observation
    asymmetry (noisy telemetry vs. uncorrupted ground-truth evaluation),
    the three swappable policy stages, execution, and state carryover.
    """
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    c_env_border = "#1f4e78"
    c_env_fill = "#f0f4f8"
    c_noise_border = "#d97706"
    c_noise_fill = "#fffbeb"
    c_policy_border = "#059669"
    c_policy_fill = "#f0fdf4"
    c_eval_border = "#b91c1c"
    c_eval_fill = "#fef2f2"
    c_text_dark = "#1e293b"

    # Outer Day Container
    day_box = mpatches.FancyBboxPatch(
        (0.012, 0.03), 0.976, 0.94,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.5, zorder=0
    )
    ax.add_patch(day_box)
    ax.text(0.035, 0.93, "SIMULATED DAY CYCLE  (Day $t \\in \\{1, \\dots, \\tau\\}$)",
            fontsize=9.5, fontweight="bold", color="#334155", zorder=1)

    # 1. Environment & Accumulation Box
    box_env = mpatches.FancyBboxPatch(
        (0.055, 0.54), 0.25, 0.35,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=c_env_fill, edgecolor=c_env_border, linewidth=1.3, zorder=1
    )
    ax.add_patch(box_env)
    ax.text(0.180, 0.845, "1. Arrival and service flag", ha="center", fontsize=8.6, fontweight="bold", color=c_env_border)
    ax.text(0.180, 0.74, "Add today's arrival, then cap at $E_i$.\n$o_i^t = 1$ if the level equals $E_i$.\n$\\ell_i^t$ is only the mass above $E_i$.\nBoth are scored before routing.",
            ha="center", va="center", fontsize=7.2, color=c_text_dark, linespacing=1.25)
    ax.text(0.180, 0.585, "Decision state, pre-collection", ha="center", fontsize=7.0, fontweight="bold", color="#64748b")

    # 2. Noisy Sensing Box (Asymmetry Highlight)
    box_noise = mpatches.FancyBboxPatch(
        (0.055, 0.13), 0.25, 0.34,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=1.3, zorder=1
    )
    ax.add_patch(box_noise)
    ax.text(0.180, 0.43, "2. Sensor (unused)", ha="center", fontsize=9.2, fontweight="bold", color="#64748b")
    # The sensor model is a framework capability; the reported runs do not
    # exercise it. Earlier versions of this figure asserted the opposite in a
    # highlighted badge, contradicting the simulation protocol -- so the noise
    # term is now shown greyed, with the operative sigma = 0 identity in black.
    ax.text(0.180, 0.365, "Optional sensor noise:\n$\\epsilon_{i,t} \\sim \\mathcal{N}(0, \\sigma^2)$",
            ha="center", va="center", fontsize=7.6, color="#94a3b8", linespacing=1.3)
    ax.text(0.180, 0.285, "This study: $\\sigma = 0$ (true observations)",
            ha="center", va="center", fontsize=7.8, fontweight="bold", color=c_text_dark)

    # Scope Callout Badge 1 -- capability vs. exercised path
    ax.text(0.180, 0.19, "Noise supported, not exercised:\nevery reported run sets $\\sigma = 0$,\nso policies observe true levels",
            ha="center", va="center", fontsize=6.8, fontweight="bold", color="#475569",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f1f5f9", edgecolor="#94a3b8", linewidth=0.8))

    # Arrow 1 -> 2
    ax.annotate("", xy=(0.180, 0.48), xytext=(0.180, 0.535),
                arrowprops=dict(arrowstyle="-|>", linewidth=1.4, color="#94a3b8"))

    # 3. Policy Pipeline Big Container (Middle)
    box_policy_bg = mpatches.FancyBboxPatch(
        (0.34, 0.12), 0.35, 0.76,
        boxstyle="round,pad=0.015,rounding_size=0.015",
        facecolor=c_policy_fill, edgecolor=c_policy_border, linewidth=1.4, zorder=1
    )
    ax.add_patch(box_policy_bg)
    ax.text(0.515, 0.84, "3. Modular Policy Pipeline", ha="center", fontsize=9.6, fontweight="bold", color=c_policy_border)
    ax.text(0.515, 0.80, "Three Independently Swappable Plugin Registries", ha="center", fontsize=7.0, style="italic", color="#047857")

    # 3a. Stage 1: Mandatory Selection
    box_p1 = mpatches.FancyBboxPatch(
        (0.36, 0.58), 0.31, 0.18,
        boxstyle="round,pad=0.01,rounding_size=0.012",
        facecolor="#ffffff", edgecolor="#059669", linewidth=1.1, zorder=2
    )
    ax.add_patch(box_p1)
    ax.text(0.515, 0.72, "Stage 1: Mandatory Selection", ha="center", fontsize=8.4, fontweight="bold", color="#065f46")
    ax.text(0.515, 0.64, "Input: capped post-arrival level\nOutput: mandatory set $\\mathcal{M}^t$\n(LM70/LM90, LA, SL1/SL2)",
            ha="center", va="center", fontsize=7.2, color=c_text_dark, linespacing=1.2)

    # 3b. Stage 2: Route Construction
    box_p2 = mpatches.FancyBboxPatch(
        (0.36, 0.36), 0.31, 0.18,
        boxstyle="round,pad=0.01,rounding_size=0.012",
        facecolor="#ffffff", edgecolor="#059669", linewidth=1.1, zorder=2
    )
    ax.add_patch(box_p2)
    ax.text(0.515, 0.50, "Stage 2: Route Construction", ha="center", fontsize=8.4, fontweight="bold", color="#065f46")
    ax.text(0.515, 0.42, "Input: $\\mathcal{M}^t$, directed $d_{ij}$\nOutput: a tour (load not re-checked)\n(eight classical constructors)",
            ha="center", va="center", fontsize=7.2, color=c_text_dark, linespacing=1.2)

    # 3c. Stage 3: Route Improvement
    box_p3 = mpatches.FancyBboxPatch(
        (0.36, 0.14), 0.31, 0.18,
        boxstyle="round,pad=0.01,rounding_size=0.012",
        facecolor="#ffffff", edgecolor="#059669", linewidth=1.1, zorder=2
    )
    ax.add_patch(box_p3)
    ax.text(0.515, 0.28, "Stage 3: Route Improvement", ha="center", fontsize=8.4, fontweight="bold", color="#065f46")
    ax.text(0.515, 0.20, "Input: constructed tour\nOutput: refined tour\n(Classical Local Search, Fast-TSP)",
            ha="center", va="center", fontsize=7.2, color=c_text_dark, linespacing=1.2)

    # Internal pipeline arrows
    ax.annotate("", xy=(0.515, 0.55), xytext=(0.515, 0.58),
                arrowprops=dict(arrowstyle="-|>", linewidth=1.3, color="#059669"))
    ax.annotate("", xy=(0.515, 0.33), xytext=(0.515, 0.36),
                arrowprops=dict(arrowstyle="-|>", linewidth=1.3, color="#059669"))

    # Arrow Sensing -> Stage 1
    ax.annotate("", xy=(0.355, 0.67), xytext=(0.315, 0.30),
                arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=-0.25", linewidth=1.5, color="#94a3b8"))

    # 4. Route Execution Box (Top Right)
    box_exec = mpatches.FancyBboxPatch(
        (0.73, 0.54), 0.23, 0.34,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=c_env_fill, edgecolor=c_env_border, linewidth=1.3, zorder=1
    )
    ax.add_patch(box_exec)
    ax.text(0.845, 0.84, "4. Route Execution", ha="center", fontsize=9.2, fontweight="bold", color=c_env_border)
    ax.text(0.845, 0.71, "Empties visited bins.\nKilometres use directed $d_{ij}$.\nThese runs do not reject\na load above $Q$.",
            ha="center", va="center", fontsize=7.3, color=c_text_dark, linespacing=1.25)
    ax.text(0.845, 0.58, "Physical Collection", ha="center", fontsize=7.0, fontweight="bold", color="#64748b")

    # Arrow Policy -> Execution
    ax.annotate("", xy=(0.725, 0.71), xytext=(0.675, 0.23),
                arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=-0.3", linewidth=1.5, color="#059669"))

    # 5. Exact Accounting Box (Bottom Right)
    box_eval = mpatches.FancyBboxPatch(
        (0.73, 0.12), 0.23, 0.34,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=c_eval_fill, edgecolor=c_eval_border, linewidth=1.3, zorder=1
    )
    ax.add_patch(box_eval)
    ax.text(0.845, 0.42, "5. Collection log", ha="center", fontsize=9.2, fontweight="bold", color=c_eval_border)
    ax.text(0.845, 0.32, "Profit $= R\\cdot\\mathrm{kg} - C\\cdot\\mathrm{km}$.\nEfficiency $=$ total kg $/$ total km.\nRevised time: all three policy stages.",
            ha="center", va="center", fontsize=7.1, color=c_text_dark, linespacing=1.25)
    
    # Callout Badge 2 -- accounting reads the true state, not the sensed signal.
    # (With sigma = 0 the two coincide, so this is a statement about the
    # accounting contract, not an observed asymmetry in these runs.)
    ax.text(0.845, 0.18, "Overflow and lost mass\nare recorded in step 1,\nnot after the route",
            ha="center", va="center", fontsize=6.8, fontweight="bold", color="#991b1b",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fee2e2", edgecolor="#ef4444", linewidth=0.8))

    # Arrow Execution -> Accounting
    ax.annotate("", xy=(0.845, 0.47), xytext=(0.845, 0.53),
                arrowprops=dict(arrowstyle="-|>", linewidth=1.4, color=c_eval_border))

    # Loop back arrow: 5 -> Next Day State -> 1.
    # Routed as an orthogonal polyline through the clear lane below the boxes
    # and up the left margin.
    lane_y, lane_x = 0.055, 0.024
    ax.plot([0.845, 0.845, lane_x, lane_x], [0.108, lane_y, lane_y, 0.715],
            color="#475569", linewidth=1.6, linestyle="--", zorder=1,
            solid_capstyle="butt")
    ax.annotate("", xy=(0.042, 0.715), xytext=(lane_x, 0.715),
                arrowprops=dict(arrowstyle="-|>", linewidth=1.6, color="#475569"))
    # The label sits on the lane; its opaque bbox breaks the dashed line, which
    # reads as one routed path rather than as a line colliding with text.
    ax.text(0.48, lane_y, "Next-Day State Transition: uncollected residual fill carries forward to Day $t+1$",
            fontsize=7.4, style="italic", color="#475569", ha="center", va="center", zorder=3,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#f8fafc", edgecolor="#cbd5e1"))

    savefig(fig, out_dir / "simulation_loop.png")


def fig_pareto(clean: pd.DataFrame, horizon: int, out: Path, colors: dict, cfg: dict) -> None:
    """kg/km against overflows, per constructor, with the non-dominated front drawn."""
    sub = clean[clean.horizon == horizon]
    agg = sub.groupby("constructor").agg(kgkm=("kgkm", "mean"), overflows=("overflows", "mean"))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, row in agg.iterrows():
        ax.scatter(row.overflows, row.kgkm, s=120, zorder=3,
                   color=colors.get(name, "#4e88d9"), edgecolor="none")
        # Points are annotated in place rather than via a legend: eight labels in
        # a legend box would cover the front on a figure this small.
        ax.annotate(display_name(name, cfg), (row.overflows, row.kgkm),
                    textcoords="offset points", xytext=(7, 4), fontsize=8)
    front = pareto_front(agg)
    ax.plot(front.overflows, front.kgkm, drawstyle="steps-post", linewidth=1.4,
            linestyle="--", color="#888888", zorder=2)
    ax.set_xlabel(cfg["axis"]["overflows"])
    ax.set_ylabel(cfg["axis"]["kgkm"])
    ax.set_title(cfg["figure_titles"]["pareto"].format(horizon=horizon))
    ax.grid(True, alpha=0.25)
    savefig(fig, out / f"pareto_{horizon}d.png")


def fig_strategy_tradeoff(clean: pd.DataFrame, horizon: int, out: Path, cfg: dict) -> None:
    """The efficiency/service trade-off across strategy variants, as paired bars."""
    sub = balance_marginal(clean[clean.horizon == horizon], "variant").copy()
    sub["variant"] = sub.strategy + sub.cf + sub.sl_var
    order = [v for v in cfg["variant_order"] if v in set(sub.variant)]
    agg = sub.groupby("variant")[["kgkm", "overflows"]].mean().reindex(order)
    # Acronyms, not full names: five spelled-out variant names overprint each
    # other on a 6.4in axis, which is how the published Fig. 6 lost the leading
    # "S" of both Service-Level labels. These are the same short forms the
    # appendix figure captions already use (LA, LM70, LM90, SL1, SL2).
    short = cfg.get("variant_labels_short", {})
    labels = [short.get(v, cfg["variant_labels"].get(v, v)) for v in agg.index]

    fig, ax_left = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(agg))
    ax_left.bar(x - 0.2, agg.kgkm, width=0.4, color="#4e88d9", label=cfg["axis"]["kgkm"])
    ax_left.set_ylabel(cfg["axis"]["kgkm"], color="#4e88d9")
    ax_left.tick_params(axis="y", labelcolor="#4e88d9")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels)

    ax_right = ax_left.twinx()
    ax_right.bar(x + 0.2, agg.overflows, width=0.4, color="#e07830", label=cfg["axis"]["overflows"])
    ax_right.set_ylabel(cfg["axis"]["overflows"], color="#e07830")
    ax_right.tick_params(axis="y", labelcolor="#e07830")
    ax_right.grid(False)

    ax_left.set_title(cfg["figure_titles"]["strategy"].format(horizon=horizon))
    savefig(fig, out / f"strategy_tradeoff_{horizon}d.png")


def fig_improver_paired(clean: pd.DataFrame, horizon: int, out: Path, cfg: dict) -> None:
    """Per-pair CLS-minus-Fast-TSP efficiency delta, sorted -- the shape is the argument."""
    pairs = improver_pairs(clean, horizon)
    delta = (pairs[("kgkm", "CLS")] - pairs[("kgkm", "FTSP")]).sort_values().to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(np.arange(len(delta)), delta, width=1.0,
           color=np.where(delta > 0, "#5cb85c", "#c04070"))
    ax.axhline(0, color="#555555", linewidth=0.9)
    ax.set_xlabel(cfg["axis"]["pair_index"].format(pairs=len(delta)))
    ax.set_ylabel(cfg["axis"]["kgkm_delta"])
    ax.set_title(cfg["figure_titles"]["improver"].format(
        horizon=horizon, wins=int((delta > 0).sum()), pairs=len(delta)))
    ax.grid(True, axis="y", alpha=0.25)
    savefig(fig, out / f"improver_delta_{horizon}d.png")


# --------------------------------------------------------------------------- #
# Network geometry
# --------------------------------------------------------------------------- #

#: Where real per-bin coordinates live when the gitignored data tree is present.
#: Absent from a fresh clone, which is why read_network_layout falls back to an
#: embedding of the road-distance matrix and says so.
COORD_DIR = REPO_ROOT / "data" / "simulator" / "graphs"


def find_distance_matrix(scenario: dict) -> Path:
    """
    Pick the trustworthy copy of a network's road-distance matrix.

    Several copies of each matrix are stored across the run directories, and some
    are longer than they should be (issue #48). Reading the first ``n`` rows of an
    oversized copy is *not* safe: the oversized files do not begin with the
    canonical block. For Figueira da Foz the three correctly sized copies agree
    with each other and every oversized copy disagrees with all of them, and two
    of the Rio Maior N=170 oversized copies disagree as well.

    So the rule is to prefer a copy whose data-row count is exactly ``n``, and to
    take the majority content when several such copies exist.
    """
    from collections import Counter

    candidates = sorted((OUTPUT_DIR / scenario["key"]).glob(f"*/*/{scenario['dm']}"))
    if not candidates:
        raise SystemExit(f"No {scenario['dm']} under {scenario['key']}")

    exact = []
    for path in candidates:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) - 1 == len(lines[0].split(",")):
            exact.append(path)
    if not exact:
        raise SystemExit(
            f"Every copy of {scenario['dm']} for {scenario['key']} is mis-sized; "
            f"cannot pick a canonical matrix (issue #48)"
        )

    digests = Counter(path.read_bytes() for path in exact)
    winner, votes = digests.most_common(1)[0]
    chosen = next(path for path in exact if path.read_bytes() == winner)
    if len(digests) > 1:
        print(f"  Note: {scenario['dm']} for {scenario['key']} has {len(digests)} distinct "
              f"correctly sized variants; taking the {votes}-way majority (issue #48)")
    return chosen


def read_distance_matrix(path: Path) -> tuple[list[int], np.ndarray]:
    """
    Return ``(node_ids, symmetric distance matrix)`` from a project distmat CSV.

    The on-disk format is a header row of node ids (depot first) followed by one
    plain row of distances per node, with no leading row-label column. Road
    distances are mildly asymmetric because of one-way streets, so the result is
    symmetrised. Use ``find_distance_matrix`` to choose the file.
    """
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    node_ids = [int(float(x)) for x in lines[0].split(",")]
    n = len(node_ids)
    if len(lines) - 1 != n:
        raise SystemExit(f"{path} has {len(lines) - 1} data rows for {n} nodes (issue #48)")
    dist = np.array([[float(v) for v in line.split(",")][:n] for line in lines[1:]])
    return node_ids, (dist + dist.T) / 2.0


def classical_mds(dist: np.ndarray, ndim: int = 2) -> np.ndarray:
    """Classical multidimensional scaling of a distance matrix."""
    d2 = dist.astype(float) ** 2
    n = d2.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * centering @ d2 @ centering
    eigvals, eigvecs = np.linalg.eigh(b)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    return eigvecs[:, :ndim] * np.sqrt(np.maximum(eigvals[:ndim], 0.0))


def embedding_stress(coords: np.ndarray, dist: np.ndarray) -> float:
    """
    Kruskal-style stress of a 2-D embedding against the true distances.

    Reported in the figure caption rather than hidden, because road networks are
    not Euclidean -- rivers, one-way systems and coastlines all force detours
    that no planar layout can honour. A stress this large is the reason the
    fallback layout is labelled a layout and not a map.
    """
    diff = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1) - dist
    off = ~np.eye(len(dist), dtype=bool)
    return float(np.sqrt((diff[off] ** 2).sum() / (dist[off] ** 2).sum()))


def read_network_layout(scenario: dict) -> dict:
    """
    Positions for one network: true coordinates when available, else an embedding.

    Returns ``{"coords", "depot", "source", "stress"}``. ``source`` is
    ``"coordinates"`` or ``"embedding"`` and drives what the caption is allowed
    to claim.
    """
    node_ids, dist = read_distance_matrix(find_distance_matrix(scenario))

    coord_file = COORD_DIR / f"{scenario['key']}.csv"
    if coord_file.exists():
        frame = pd.read_csv(coord_file).set_index("id")
        coords = frame.loc[node_ids, ["x", "y"]].to_numpy(dtype=float)
        return {"coords": coords, "depot": 0, "source": "coordinates", "stress": None}

    coords = classical_mds(dist)
    return {
        "coords": coords,
        "depot": 0,
        "source": "embedding",
        "stress": embedding_stress(coords, dist),
    }


def fig_networks(out_dir: Path, cfg: dict) -> None:
    """Compose the retained coordinate maps for the two benchmark networks.

    The raw coordinate exports are intentionally gitignored, but the analysis
    pipeline's exact scenario maps are tracked.  Those artifacts were rendered
    from the selected bins' latitude/longitude over OpenStreetMap drive-network
    geometry.  Reusing them is both reproducible and geographically honest;
    reconstructing positions from a road-distance matrix is neither.
    """
    scenarios = cfg["networks"]["panels"]

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.25), facecolor="#f6f2e9")
    for ax, scenario in zip(axes, scenarios, strict=True):
        source = NETWORK_MAP_DIR / scenario["map_artifact"]
        if not source.exists():
            raise SystemExit(
                f"Missing coordinate-map artifact {source}; regenerate it with "
                "logic/gen/gen_simulation_analysis.py while the coordinate data are mounted"
            )

        image = plt.imread(source)[70:, :, :3].copy()  # discard the source title
        content = np.any(image < 0.94, axis=2)
        ys, xs = np.nonzero(content)
        pad = 12
        image = image[max(0, ys.min() - pad):ys.max() + pad,
                      max(0, xs.min() - pad):xs.max() + pad]

        # Retain the real geometry while giving the paper figure a legible,
        # cartographic palette instead of a large blank white canvas.
        background = np.all(image > 0.965, axis=2)
        roads = ((image[..., 0] > 0.62) & (image[..., 0] < 0.9)
                 & (np.max(image, axis=2) - np.min(image, axis=2) < 0.12))
        bins = ((image[..., 0] > 0.55) & (image[..., 1] < 0.52)
                & (image[..., 2] < 0.52))
        image[background] = np.array([0.965, 0.945, 0.89])
        image[roads] = np.array([0.48, 0.58, 0.65])
        rgb = np.array([int(scenario["color"][i:i + 2], 16) for i in (1, 3, 5)]) / 255
        image[bins] = rgb

        ax.set_facecolor("#f6f2e9")
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#91a0aa")
            spine.set_linewidth(0.8)
        ax.set_title(f"{scenario['city']}  ($N={scenario['N']}$)", fontsize=10,
                     fontweight="bold", pad=7)
        ax.text(0.03, 0.035, "selected plastic bins  |  OSM roads",
                transform=ax.transAxes, fontsize=6.7, color="#35434c",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "#fffdf7",
                      "edgecolor": "#aab5bb", "alpha": 0.92})
        ax.annotate("", xy=(0.94, 0.94), xytext=(0.94, 0.84),
                    xycoords="axes fraction",
                    arrowprops={"arrowstyle": "-|>", "color": "#35434c", "lw": 0.9})
        ax.text(0.94, 0.955, "N", ha="center", va="bottom",
                transform=ax.transAxes, fontsize=7, fontweight="bold", color="#35434c")

    fig.text(0.995, 0.006, "© OpenStreetMap contributors", ha="right", va="bottom",
             fontsize=5.5, color="#5d6970")
    savefig(fig, out_dir / "networks.png")
    print("    coordinate-derived selected-bin maps over OpenStreetMap road geometry")


def fig_scale(clean: pd.DataFrame, horizon: int, out: Path, colors: dict, cfg: dict) -> None:
    """Runtime against instance size, per constructor -- the cost side of the trade-off."""
    sub = clean[clean.horizon == horizon]
    agg = sub.pivot_table(index="N", columns="constructor", values="time", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name in agg.columns:
        ax.plot(agg.index, agg[name], marker="o", linewidth=1.6,
                color=colors.get(name, "#4e88d9"), label=display_name(name, cfg))
    ax.set_xlabel(cfg["axis"]["n_bins"])
    ax.set_ylabel(cfg["axis"]["time"])
    ax.set_yscale("log")
    ax.set_xticks(sorted(agg.index))
    ax.set_title(cfg["figure_titles"]["scale"].format(horizon=horizon))
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    savefig(fig, out / f"runtime_scaling_{horizon}d.png")


def _recover_increments(fills: np.ndarray) -> np.ndarray:
    """Recover the policy-independent daily waste increment per bin.

    The fill-history spreadsheet records the *realised* level after each day's
    collection, but the waste arriving each day is fixed by the demand
    realisation (the simulator re-seeds waste per policy-and-day, so every
    policy faces the identical increments). A drop between consecutive days
    marks a collection (reset to zero), so the next day's level is exactly that
    day's increment; otherwise the increment is the level rise.
    """
    inc = np.zeros_like(fills)
    inc[:, 0] = fills[:, 0]
    inc[:, 1:] = np.where(
        fills[:, 1:] >= fills[:, :-1],
        fills[:, 1:] - fills[:, :-1],
        fills[:, 1:],
    )
    return inc


def _simulate_last_minute(increments: np.ndarray, tau: float, capacity: float) -> np.ndarray:
    """Re-simulate the Last-Minute rule and return per-bin, per-day fill levels."""
    n_bins, n_days = increments.shape
    level = np.zeros(n_bins)
    out = np.zeros_like(increments)
    for d in range(n_days):
        collect = level >= tau
        level = np.minimum(level + increments[:, d], capacity)
        out[:, d] = level.copy()
        level[collect] = 0.0
    return out


def fig_fill_trajectory(out_dir: Path) -> None:
    """
    The multi-period mechanism behind the paper's central claim, as a figure.

    The static aggregates show an ordered trade-off between efficiency and
    overflow, but not *why*. This figure shows the Last-Minute mechanism for three bins'
    fill levels over the 30-day horizon under the two Last-Minute thresholds,
    CF70 and CF90, re-simulated from the recovered daily increments. CF70
    collects earlier and more often, holding the bin below the threshold; CF90
    defers collection until the bin is nearly full, hauling more per visit but
    occasionally overflowing. That is the trade-off, at the level of a single
    bin, before any aggregation.
    """
    import openpyxl

    candidate = (
        REPO_ROOT / "assets/output/30days/riomaior100_plastic/gamma3/lm_ftsp/fill_history"
    )
    xlsx = sorted(candidate.glob("*.xlsx"))[0] if candidate.is_dir() else None
    if xlsx is None:
        print("  Skipped fill-trajectory figure (no fill_history found).")
        return

    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    fills = np.array([[float(c) for c in row] for row in ws.iter_rows(values_only=True)])
    increments = _recover_increments(fills)
    capacity = 100.0

    levels_70 = _simulate_last_minute(increments, 70.0, capacity)
    levels_90 = _simulate_last_minute(increments, 90.0, capacity)

    # Choose three bins that span the accumulation spectrum: a heavy one that
    # overflows under CF90 but not CF70, a typical one, and a light one.
    total = increments.sum(axis=1)
    heavy = int(np.argmax((levels_90.max(axis=1) >= capacity) & (levels_70.max(axis=1) < capacity)))
    median = int(np.argsort(total)[len(total) // 2])
    light = int(np.argsort(total)[max(0, len(total) // 8)])

    days = np.arange(1, levels_70.shape[1] + 1)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9), sharey=True)
    for ax, (idx, label) in zip(
        axes, [(heavy, "High accumulation"), (median, "Typical"), (light, "Light")], strict=True
    ):
        ax.step(days, levels_70[idx], where="post", linewidth=1.4, color="#4e88d9", label="CF70")
        ax.step(days, levels_90[idx], where="post", linewidth=1.4, color="#e07830", linestyle="--", label="CF90")
        ax.axhline(100, color="#c04070", linewidth=0.9, linestyle=":")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Day", fontsize=8)
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("Fill level (% of capacity)", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Last-Minute selection at two thresholds (Rio Maior, $N=100$, Gamma-3)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    savefig(fig, out_dir / "fill_trajectory_30d.png")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_horizons(values: list[str] | None) -> dict[int, Path]:
    if not values:
        return {
            30: REPO_ROOT / "docs/private/global/simulation/simulation_summary.csv",
            90: REPO_ROOT / "docs/private/global/simulation/simulation_summary_90d.csv",
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
    ap.add_argument("--horizon", action="append", metavar="DAYS=CSV",
                    help="summary CSV for one horizon; repeatable (default: 30d and 90d)")
    ap.add_argument("--tables-dir", type=Path, default=default_paper / cfg["paths"]["tables_subdir"])
    ap.add_argument("--figures-dir", type=Path, default=default_paper / cfg["paths"]["figures_subdir"])
    ap.add_argument("--theme", choices=["dark", "light"], default="light",
                    help="light matches the paper's white background (default)")
    ap.add_argument("--primary-horizon", type=int, default=30,
                    help="horizon whose grid is complete and therefore leads the results")
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
    clean, degenerate = split_degenerate(raw)
    clean = drop_affected_cells(clean, degenerate)
    paired = paired_horizon_frame(clean)
    print(f"  {len(clean)} runs retained; {len(paired)} configurations present at both horizons")

    horizon = args.primary_horizon

    if not args.figures_only:
        args.tables_dir.mkdir(parents=True, exist_ok=True)
        print(f"Writing LaTeX tables to {args.tables_dir}:")
        write(args.tables_dir / "results_constructors.tex", table_constructors(clean, horizon, cfg), args.force)
        write(args.tables_dir / "results_strategies.tex", table_strategies(clean, horizon, cfg), args.force)
        write(args.tables_dir / "results_improvers.tex", table_improvers(clean, horizon, cfg), args.force)
        write(args.tables_dir / "results_horizon.tex", table_horizon(paired, cfg), args.force)
        write(args.tables_dir / "results_scenarios.tex", table_scenarios(clean, horizon, cfg), args.force)
        write(args.tables_dir / "results_excluded.tex", table_excluded(degenerate, cfg), args.force)

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
