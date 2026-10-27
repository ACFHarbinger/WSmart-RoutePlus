"""Data layer of the paper generator: summary loading, degenerate-run exclusion, balanced marginals and LaTeX formatting helpers.

The rationale for the exclusion and balancing rules is in ``gen_paper_latex.py``'s module docstring.

Attributes:
    load_horizons: Read the per-horizon summary CSVs.
    split_degenerate: Separate degenerate runs from the clean frame.
    balance_marginal: Restrict a marginal to the slices every level shares.
    aggregate: Mean/std of the reported metrics per group.

Example:
    >>> frames = load_horizons({30: Path("simulation_summary.csv")})
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    pass
else:
    pass


REPO_ROOT = Path(__file__).resolve().parents[2]


#: Raw simulation output tree, source of the road-distance matrices.
OUTPUT_DIR = REPO_ROOT / "outputs" / "simulations" / "30days"


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


#: The columns that identify a scenario cell: everything except the routing algorithm,
#: so that the routing algorithms within a cell are exactly the competitors that faced
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


#: The learned routing algorithm's ``constructor`` key. Its runs are reported in their own
#: block under each classical table and never enter the classical aggregates or cell medians.
LEARNED_KEY = "AM"


#: The static reference's ``constructor`` key: fixed Regular-k schedules (``strategy``) with the
#: CVRP routing algorithm and no improver. Outside the factorial design, so outside every
#: classical aggregate and the exclusion rule; reported as its own block of the routing table.
STATIC_KEY = "CVRP"


#: The summary vintage the paper reads: rerun-2026-10-19 plus the directed SWC-TCF rows (rerun-2026-10-26).
DEFAULT_SUMMARY_DIR = REPO_ROOT / "docs/private/global/simulation/rerun-2026-10-26"


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


def find_degenerate_runs(df: pd.DataFrame) -> pd.Series:
    """
    Boolean mask of runs that failed to complete rather than merely performing badly.

    Within a scenario cell every routing algorithm faces the identical demand
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
    Remove every scenario cell that contained a degenerate run, for all routing algorithms.

    Dropping only the degenerate rows would reintroduce exactly the bias this
    module refuses to accept for the 90-day sample. All four degenerate runs
    belong to SWC-TCF, so removing them alone would leave SWC-TCF averaged over
    the scenarios where it did *not* fail while every rival is averaged over
    those scenarios too -- a mean quietly computed on a favourable subset.

    Removing the whole cell costs the other routing algorithms those scenarios as well,
    but keeps every routing algorithm's mean over an identical set of scenarios, which
    is the only reason the marginal means are comparable in the first place.
    """
    if degenerate.empty:
        return df
    affected = set(map(tuple, degenerate[CELL_KEYS].to_numpy()))
    keep = ~df[CELL_KEYS].apply(lambda row: tuple(row) in affected, axis=1)
    dropped = int((~keep).sum())
    print(
        f"  Dropped {dropped} further run(s) so that every routing algorithm is averaged "
        f"over the same {len(affected)} fewer scenario cells"
    )
    return df[keep].copy()


def balance_marginal(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    """
    Restrict ``df`` to the slices in which *every* level of ``stage`` is present.

    ``drop_affected_cells`` balances the routing algorithm marginal only, because
    ``constructor`` is absent from ``CELL_KEYS`` and so all routing algorithms in a
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
    slice_cols = [c for c in df.columns if c in set(SCENARIO_KEYS) | policy_cols and c not in stage_cols]
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


def learned_runs(learned: pd.DataFrame, classical: pd.DataFrame, degenerate: pd.DataFrame) -> pd.DataFrame:
    """
    The learned routing algorithm's runs, put through the classical exclusion rule.

    Its runs sit outside the classical cell medians, so its shortfall is measured
    against the median of the classical runs in the same cell. It loses every
    cell the classical data loses, so that its blocks cover the same scenario
    cells as the tables above them, and any run of its own that falls more than
    ``SHORTFALL_THRESHOLD`` short is dropped and reported.
    """
    if learned.empty:
        return learned
    median = classical.groupby(CELL_KEYS, dropna=False)["kg"].median().rename("cell_median")
    learned = learned.join(median, on=CELL_KEYS)
    learned["shortfall"] = 1.0 - learned["kg"] / learned["cell_median"]
    short = learned[learned["shortfall"] > SHORTFALL_THRESHOLD]
    for _, r in short.iterrows():
        print(
            f"  Excluded {r.constructor} {r.horizon:>3}d {r.city} N={r.N} {r.dist} "
            f"{r.strategy}{r.cf}{r.sl_var:<4} {r.improver}: {r.shortfall:.0%} below the classical cell median"
        )
    kept = drop_affected_cells(learned.drop(index=short.index), degenerate)
    print(f"  {len(kept)} of {len(learned)} {LEARNED_KEY} runs retained")
    return kept.drop(columns=["cell_median"])


def paired_horizon_frame(clean: pd.DataFrame) -> pd.DataFrame:
    """
    Configurations observed at both horizons, as one row per configuration.

    This is the horizon contrast least distorted by the 90-day sampling, because
    each configuration is compared against *itself* rather than against a rival
    whose 90-day scenarios are a different, self-favouring subset.

    It does not identify a horizon effect for the full design. A configuration
    belongs to a non-factorial 90-day subset whose selection rule remains
    pending recovery; no replicated seeds identify the selection effect. The paired difference is therefore descriptive
    of the selected configurations only.
    """
    wide = clean.pivot_table(
        index=CONFIG_KEYS, columns="horizon", values=["kgkm", "overflows", "km", "time", "kg_lost"]
    )
    wide = wide.dropna()
    if wide.empty:
        return wide
    return wide.reset_index()


def display_name(raw: str, cfg: dict) -> str:
    """Map a CSV routing algorithm key to the name the paper uses (ACO_HH -> ACO-HH)."""
    return cfg.get("routing_labels", {}).get(raw, raw)


def tex_escape(s: object) -> str:
    """Escape the LaTeX specials that appear in this data (routing algorithm names, mostly)."""
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


#: Means are reported beside medians for the two headline metrics. A single run
#: out of ~96 moved Service-Level's mean overflow count from 9.3 to 1.4, so the
#: means here are genuinely fragile to individual scenarios and a median-based
#: reading is the honest companion, not a footnote.
ROUTING_SPEC = [
    ("n", "size", 0, "none"),
    ("kgkm", "mean", 2, "max"),
    ("kgkm", "median", 2, "max"),
    ("overflows", "mean", 1, "min"),
    ("overflows", "median", 1, "min"),
    ("km", "mean", 0, "min"),
    ("kg_lost", "mean", 1, "min"),
    ("time", "mean", 0, "min"),
]


STRATEGY_SPEC = ROUTING_SPEC


#: Header cells matching ROUTING_SPEC/STRATEGY_SPEC's seven columns: the two
#: headline metrics span a mean/median pair under one centered label, with the
#: mean/median distinction carried by a second header row rather than by a
#: literal "med." column label competing with the numbers for space.
METRIC_GROUP_HEADERS = [mc("Runs"), mc("kg/km", 2), mc("Overflows", 2), mc("km"), mc("Lost (kg)"), mc("Time (s)")]


METRIC_GROUP_SUBHEADERS = ["", "mean", "median", "mean", "median", "", "mean", ""]


def unmarked(spec: list[tuple[str, str, int, str]]) -> list[tuple[str, str, int, str]]:
    """The same spec with best-marking off: the learned block is not ranked against the classical rows."""
    return [(col, how, dec, "none") for col, how, dec, _ in spec]


def text_for(cfg: dict, section: str, key: str, learned: pd.DataFrame) -> str:
    """The ``*_am`` variant of a caption or note when the learned routing algorithm has runs."""
    texts = cfg[section]
    return texts.get(f"{key}_am", texts[key]) if not learned.empty else texts[key]


def pending_blocks(cfg: dict, table: str) -> list[dict]:
    """Placeholder blocks for results not in the summary CSVs yet, one TODO per cell."""
    blocks = []
    for block in cfg.get("pending_blocks", {}).get(table, []):
        rows = [{"label": label, "cells": ["TODO"] * block["columns"]} for label in block["rows"]]
        blocks.append({"title": block["title"], "rows": rows})
    return blocks


def strategy_frame(frame: pd.DataFrame, horizon: int, cfg: dict) -> pd.DataFrame:
    """Per-variant aggregates over the slices every variant shares, in the configured order."""
    sub = balance_marginal(frame[frame.horizon == horizon], "variant").copy()
    sub["variant"] = sub.strategy + sub.cf + sub.sl_var
    order = [v for v in cfg["variant_order"] if v in set(sub.variant)]
    agg = aggregate(sub, "variant", STRATEGY_SPEC).reindex(order)
    agg.index = [cfg["variant_labels"].get(v, v) for v in agg.index]
    return agg


def improver_pairs(clean: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    CLS against Fast-TSP on the configurations that differ *only* in improver.

    Pairing rather than averaging controls scenario mix: every pair has the same
    routing algorithm, packing algorithm, scenario and demand realisation. It does
    not isolate a causal improver effect, because stochastic upstream
    construction outcomes are not identical in every pair.
    """
    sub = clean[clean.horizon == horizon]
    keys = [k for k in CONFIG_KEYS if k != "improver"]
    wide = sub.pivot_table(index=keys, columns="improver", values=["kgkm", "km", "overflows", "time", "kg_lost"])
    return wide.dropna()


SCENARIO_METRICS = [
    ("n", "size", 0, "none"),
    ("kgkm", "mean", 2, "none"),
    ("overflows", "mean", 1, "none"),
    ("km", "mean", 0, "none"),
    ("kg_lost", "mean", 1, "none"),
    ("time", "mean", 0, "none"),
]


SCENARIO_ORDER = [
    "Demand: Empirical",
    "Demand: Gamma-3",
    "Rio Maior ($N=100$)",
    "Rio Maior ($N=170$)",
    "Figueira da Foz ($N=350$)",
]


def scenario_frame(frame: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, int, int]:
    """Demand-process and network aggregates, each over the slices all its levels share.

    Also returns the number of shared policy slices behind each block (runs per level).
    """
    primary = frame[frame.horizon == horizon]
    dist = balance_marginal(primary, "dist").copy()
    dist["level"] = dist["dist"].map(lambda value: f"Demand: {value}")
    network = balance_marginal(primary, "network").copy()
    network["level"] = network.apply(lambda row: f"{row.city} ($N={row.N}$)", axis=1)
    both = pd.concat([dist, network], ignore_index=True)
    slices = (len(dist) // max(dist["dist"].nunique(), 1), len(network) // max(network["level"].nunique(), 1))
    return aggregate(both, "level", SCENARIO_METRICS).reindex(SCENARIO_ORDER), *slices
