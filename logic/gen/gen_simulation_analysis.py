"""
Generate the simulation analysis markdown report and all associated figures.

The script is agnostic to the mandatory node selection strategies, route
constructors (with optional acceptance criteria), route improvers, simulation
scenarios (region, graph size, data distribution) and time horizons present in
the data: everything is auto-detected and can be narrowed via a JSON
configuration file (see logic/gen/json/simulation_analysis_config.json)
and/or CLI flags.

It also subsumes the old gen_simulation_csv.py: raw simulation output trees
(assets/output/<horizon>days/...) can be parsed directly into the summary CSV
schema, either standalone (--parse-output) or on the fly when a horizon entry
in the config specifies "output_dir" instead of (or in addition to) "csv".

For every requested horizon the same report sections are generated (ordered
from the smallest to the largest horizon); a final cross-horizon comparison
section is appended only when more than one horizon is analysed.

Non-Python content lives in sibling directories:
  jinja/simulation_analysis.md.j2   markdown template
  json/simulation_metadata.json     parsing metadata + colour palettes
  json/simulation_analysis_config.json  default analysis configuration
  json/themes.json                  dark/light theme definitions
  style/{dark,light}.mplstyle       matplotlib style sheets
  js/pareto_buttons.js              plotly button styling snippet

Usage
-----
    # Full default analysis (config-driven; 30d + 90d if both CSVs exist)
    uv run python logic/gen/gen_simulation_analysis.py --force

    # White-background charts, Pareto plots restricted to the front points
    uv run python logic/gen/gen_simulation_analysis.py \\
        --theme light --pareto-points front --force

    # Only one horizon, explicit CSV
    uv run python logic/gen/gen_simulation_analysis.py \\
        --horizon 30=docs/private/global/simulation/simulation_summary.csv --force

    # Regenerate a summary CSV from a raw output tree (old gen_simulation_csv)
    uv run python logic/gen/gen_simulation_analysis.py --parse-output \\
        --output-dir assets/output/90days \\
        --out-csv docs/private/global/simulation/simulation_summary_90d.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from report_utils import (  # pyrefly: ignore [missing-import]
    PLACEHOLDER,
    finalize_markdown,
    load_js,
    load_json,
    load_theme,
    render_template,
    savefig,
)

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

META = load_json("simulation_metadata.json")

MARKER_CYCLE = ["o", "s", "D", "^", "v", "P", "X", "*"]
HATCH_CYCLE = ["", "//", "xx", ".."]
PLOTLY_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "triangle-down", "pentagon", "x", "star"]


def disp(label) -> str:
    """Display-formatting for a raw data label: ACO_HH -> ACO-HH (data values are untouched)."""
    return str(label).replace("ACO_HH", "ACO-HH")


def disp_all(labels) -> list[str]:
    return [disp(x) for x in labels]


KGKM_LABEL = "KG / KM"

# ── Chart fontsize scaling ───────────────────────────────────────────────────
# Chart fontsizes are controlled per element type so that axis labels and
# legends can be sized differently from in-chart annotation/point labels.
# All FS_*(n) functions scale the literal n from the reference (11pt) to the
# configured base for that element type; the global FS() still provides a
# backward-compatible uniform scale.
#
# Override via CLI: --fontsize (global), --fontsize-axis, --fontsize-labels,
# --fontsize-legend, --fontsize-title.  Or via config "base_fontsize" key.
_BASE_FONTSIZE_REF = 11.0
_FONT_SCALE = 1.0
_AXIS_SCALE = 1.0
_LABEL_SCALE = 0.80   # annotation/point labels smaller than axis by default
_LEGEND_SCALE = 1.0
_TITLE_SCALE = 1.0


def FS(n: float) -> float:
    """Scale a chart fontsize literal by the global base-fontsize setting."""
    return n * _FONT_SCALE


def FS_AXIS(n: float) -> float:
    """Scale an axis/panel-title fontsize literal."""
    return n * _AXIS_SCALE


def FS_LABEL(n: float) -> float:
    """Scale a point-annotation / in-chart label fontsize literal (smaller bucket)."""
    return n * _LABEL_SCALE


def FS_LEGEND(n: float) -> float:
    """Scale a legend fontsize literal."""
    return n * _LEGEND_SCALE


def FS_TITLE(n: float) -> float:
    """Scale a figure super-title fontsize literal."""
    return n * _TITLE_SCALE


def set_chart_fontsize(
    base_fontsize: float,
    axis_fontsize: float | None = None,
    label_fontsize: float | None = None,
    legend_fontsize: float | None = None,
    title_fontsize: float | None = None,
) -> None:
    """Configure per-element fontsize scales.

    `base_fontsize` sets the global FS() scale.  Each optional per-element
    argument, if given, overrides that element's independent scale (FS_AXIS,
    FS_LABEL, FS_LEGEND, FS_TITLE); otherwise the element inherits the global
    scale — except FS_LABEL which defaults to 0.80 × global.
    """
    global _FONT_SCALE, _AXIS_SCALE, _LABEL_SCALE, _LEGEND_SCALE, _TITLE_SCALE
    _FONT_SCALE = base_fontsize / _BASE_FONTSIZE_REF
    matplotlib.rcParams["font.size"] = base_fontsize
    _AXIS_SCALE = (axis_fontsize / _BASE_FONTSIZE_REF) if axis_fontsize is not None else _FONT_SCALE
    _LEGEND_SCALE = (legend_fontsize / _BASE_FONTSIZE_REF) if legend_fontsize is not None else _FONT_SCALE
    _TITLE_SCALE = (title_fontsize / _BASE_FONTSIZE_REF) if title_fontsize is not None else _FONT_SCALE
    _LABEL_SCALE = (label_fontsize / _BASE_FONTSIZE_REF) if label_fontsize is not None else _FONT_SCALE * 0.80


# ── Raw output tree parsing (merged from gen_simulation_csv.py) ────────────────


def _parse_area_dir(dirname: str) -> tuple[str, int] | None:
    """Map e.g. 'riomaior100_plastic' → ('Rio Maior', 100)."""
    m = re.match(r"([a-z]+?)(\d+)(?:_\w+)?$", dirname)
    if not m:
        return None
    city = META["dir_city"].get(m.group(1))
    if city is None:
        return None
    return city, int(m.group(2))


def _parse_filename(stem: str) -> dict | None:
    """
    Parse log filename stem into metadata.

    Expected form: log_{strategy_tokens}_{constructor}[_{acceptance}]_{improver}_{N}N
    """
    if not stem.startswith("log_"):
        return None
    rest = stem[4:]

    strategy = cf = sl_var = None
    for spec in META["strategy_prefixes"]:
        if rest.startswith(spec["prefix"]):
            strategy, cf, sl_var = spec["strategy"], spec["cf"], spec["sl_var"]
            rest = rest[len(spec["prefix"]) :]
            break
    if strategy is None:
        return None

    m = re.search(META["improver_pattern"], rest, re.IGNORECASE)
    if not m:
        return None
    improver = m.group(1).upper()
    middle = rest[: m.start()]

    constructor = acceptance = None
    for token, label in META["constructors"]:
        if middle.startswith(token):
            constructor = label
            acceptance = middle[len(token) :].strip("_") or None
            break
    if constructor is None:
        return None

    return {
        "strategy": strategy,
        "cf": cf,
        "sl_var": sl_var,
        "improver": improver,
        "constructor": constructor,
        "acceptance": acceptance,
    }


def parse_output_dir(output_dir: Path) -> pd.DataFrame:
    """Walk *output_dir* and return a DataFrame of simulation metrics."""
    rows: list[dict] = []
    for area_dir in sorted(output_dir.iterdir()):
        if not area_dir.is_dir():
            continue
        city_n = _parse_area_dir(area_dir.name)
        if city_n is None:
            continue
        city, N = city_n

        for dist_dir in sorted(area_dir.iterdir()):
            if not dist_dir.is_dir():
                continue
            dist = META["dist_map"].get(dist_dir.name)
            if dist is None:
                continue

            for log_file in sorted(dist_dir.rglob("log_*.json")):
                meta = _parse_filename(log_file.stem)
                if meta is None:
                    continue
                try:
                    data = json.loads(log_file.read_text())
                except Exception:
                    continue
                mean = data.get("mean", {})
                if not mean:
                    continue
                rows.append(
                    {
                        "city": city,
                        "N": N,
                        "dist": dist,
                        "improver": meta["improver"],
                        "strategy": meta["strategy"],
                        "cf": meta["cf"] or "",
                        "sl_var": meta["sl_var"] or "",
                        "acceptance": meta["acceptance"] or "",
                        "constructor": meta["constructor"],
                        "overflows": mean.get("overflows", 0),
                        "kg": mean.get("kg", 0),
                        "ncol": mean.get("ncol", 0),
                        "kg_lost": mean.get("kg_lost", 0),
                        "km": mean.get("km", 0),
                        "kgkm": mean.get("kg/km", 0),
                        "reward": mean.get("reward", 0),
                        "profit": mean.get("profit", 0),
                        "time": mean.get("time", 0),
                        "days": mean.get("days", 0),
                    }
                )
    return pd.DataFrame(rows)


# ── Labels & data preparation ──────────────────────────────────────────────────


def region_label(city: str, N: int) -> str:
    short = META["city_short"].get(city, city[:3].upper())
    return f"{short}-{N}"


def scenario_label(s: dict) -> str:
    return f"{region_label(s['city'], s['N'])} / {s['dist']}"


def variant_label(row) -> str:
    strat = row["strategy"]
    if strat == "LM" and str(row.get("cf", "")) not in ("", "nan", "None"):
        return f"LM ({row['cf']})"
    if strat == "SL" and str(row.get("sl_var", "")) not in ("", "nan", "None"):
        return f"SL ({row['sl_var']})"
    return strat


def variant_color(label: str, fallback_idx: int = 0) -> str:
    palette = META["variant_colors"]
    if label in palette:
        return palette[label]
    cycle = list(palette.values()) + META["scenario_colors"]
    return cycle[fallback_idx % len(cycle)]


def load_horizon_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("cf", "sl_var", "acceptance"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")
    df["variant"] = df.apply(variant_label, axis=1)
    return df


def detect_scenarios(df: pd.DataFrame) -> list[dict]:
    scen = df[["city", "N", "dist"]].drop_duplicates().to_dict("records") # pyrefly: ignore [no-matching-overload]
    return sorted(scen, key=lambda s: (s["N"], s["city"], s["dist"]))


def filter_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Restrict *df* to the configured scenarios and policy components."""
    scenarios = config.get("scenarios")
    if scenarios:
        mask = pd.Series(False, index=df.index)
        for s in scenarios:
            mask |= (df.city == s["city"]) & (int(s["N"]) == df.N) & (df.dist == s["dist"])
        df = df[mask] # pyrefly: ignore [bad-assignment]
    pol = config.get("policies") or {}
    for key, col in [
        ("strategies", "strategy"),
        ("constructors", "constructor"),
        ("improvers", "improver"),
        ("acceptance", "acceptance"),
    ]:
        allowed = pol.get(key)
        if allowed:
            df = df[df[col].isin(allowed)] # pyrefly: ignore [bad-assignment]
    return df.reset_index(drop=True)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Average metrics over CF/SL variants (per scenario × strategy × constructor × improver)."""
    return (
        df.groupby(["city", "N", "dist", "improver", "strategy", "constructor"])[
            ["overflows", "kgkm", "km", "profit", "kg", "reward"]
        ]
        .mean()
        .reset_index()
    )


def _scen_sub(df: pd.DataFrame, s: dict) -> pd.DataFrame:
    return df[(df.city == s["city"]) & (s["N"] == df.N) & (df.dist == s["dist"])] # pyrefly: ignore [bad-return]


def _panel_grid(n: int, panel_size: tuple[float, float] = (10.0, 7.0)) -> tuple[plt.Figure, list]:
    ncols = min(n, 2)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows), squeeze=False)
    flat = [axes[i // ncols][i % ncols] for i in range(nrows * ncols)]
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def _norm_scales(opt, default: str = "linear") -> list[str]:
    if opt is None:
        return [default]
    if isinstance(opt, str):
        return [opt]
    return list(opt)


def pareto_indices(xs, ys) -> set[int]:
    """Positional indices of non-dominated points (min x=overflows, max y=kgkm)."""
    order = sorted(range(len(xs)), key=lambda i: (xs[i], -ys[i]))
    front, best = set(), -np.inf
    for i in order:
        if ys[i] > best:
            front.add(i)
            best = ys[i]
    return front


# ── Static figure generators ───────────────────────────────────────────────────


# Per-region Pareto-front (color, linestyle) pairs — deliberately darker and
# distinct from the LA/LM/SL point palette (blue/red/orange/teal/green), and
# keyed by region only (not distribution) so the same region draws identically
# in both the Empirical and Gamma-3 panels.
_FRONT_STYLES: list[tuple[str, str]] = [
    ("#5B2C6F", "-"),   # 1st region (RM-100) — dark purple, solid
    ("#7B3F00", "--"),  # 2nd region (RM-170) — dark bronze, dashed
    ("#37474F", ":"),   # 3rd region (FFZ-350) — dark slate, dotted
]


def gen_pareto_scatter(df: pd.DataFrame, ctx: dict, out_dir: Path) -> None:
    """Per-distribution panels of the overflows/kg-km space with Pareto fronts."""
    opts = ctx["charts"].get("pareto_scatter", {})
    scenarios, dists, improvers = ctx["scenarios"], ctx["dists"], ctx["improvers"]
    regions = ctx["regions"]
    variants = sorted(df["variant"].unique())
    vcolors = {v: variant_color(v, i) for i, v in enumerate(variants)}
    markers = {tuple(r): MARKER_CYCLE[i % len(MARKER_CYCLE)] for i, r in enumerate(regions)}
    front_style = {r: _FRONT_STYLES[i % len(_FRONT_STYLES)] for i, r in enumerate(regions)}
    points_mode = ctx.get("pareto_points") or opts.get("points", "all")

    def _make(xscale: str) -> plt.Figure:
        fig, axes = _panel_grid(len(dists), (11, 8))
        for ax, dist in zip(axes, dists, strict=True):
            for s in [sc for sc in scenarios if sc["dist"] == dist]:
                sub = _scen_sub(df[df.dist == dist], s).reset_index(drop=True) # pyrefly: ignore [bad-argument-type]
                if sub.empty:
                    continue
                front = pareto_indices(sub["overflows"].values, sub["kgkm"].values)
                for i, row in sub.iterrows():
                    if points_mode == "front" and i not in front:
                        continue
                    filled = row["improver"] == improvers[0]
                    color = vcolors[row["variant"]]
                    ax.scatter(
                        row["overflows"],
                        row["kgkm"],
                        marker=markers[(s["city"], s["N"])],
                        s=90 if i in front else 45,
                        facecolors=color if filled else "none",
                        edgecolors=color,
                        linewidths=1.4,
                        alpha=0.85,
                        zorder=3,
                    )
                pts = sorted((sub["overflows"][i], sub["kgkm"][i]) for i in front)
                if pts:
                    sx, sy = [pts[0][0]], [pts[0][1]]
                    for j in range(1, len(pts)):
                        sx += [pts[j][0], pts[j][0]]
                        sy += [pts[j - 1][1], pts[j][1]]
                    front_color, front_ls = front_style[(s["city"], s["N"])]
                    ax.plot(
                        sx,
                        sy,
                        linestyle=front_ls,
                        color=front_color,
                        linewidth=2.3,
                        alpha=0.95,
                        zorder=2,
                    )
            if xscale != "linear":
                ax.set_xscale("symlog", linthresh=1)
            ax.set_xlabel(f"Overflows ({ctx['n_days']} days)", fontsize=FS(19), fontweight="bold")
            ax.set_ylabel(f"Efficiency ({KGKM_LABEL})", fontsize=FS(19), fontweight="bold")
            ax.set_title(f"{dist}", fontsize=FS(19), fontweight="bold")
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontweight("bold")
            ax.tick_params(axis="both", labelsize=11)
            ax.yaxis.grid(True, alpha=0.4)
            ax.xaxis.grid(True, alpha=0.4)
            ax.set_axisbelow(True)
        leg = [mpatches.Patch(color=vcolors[v], label=v) for v in variants]
        leg += [
            plt.Line2D(
                [],
                [],
                marker=markers[tuple(r)],
                linestyle="",
                color=ctx["theme"]["muted"],
                label=region_label(r[0], r[1]),
            )
            for r in regions
        ]
        if len(improvers) > 1:
            leg += [
                plt.Line2D(
                    [], [], marker="o", linestyle="", color=ctx["theme"]["fg"], label=f"{improvers[0]} (filled)"
                ),
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    markerfacecolor="none",
                    color=ctx["theme"]["fg"],
                    label=f"{improvers[-1]} (open)",
                ),
            ]
        leg += [
            plt.Line2D(
                [0],
                [0],
                color=front_style[r][0],
                linestyle=front_style[r][1],
                linewidth=2.3,
                label=f"Front — {region_label(r[0], r[1])} (both distributions)",
            )
            for r in regions
        ]
        leg_obj = fig.legend(
            handles=leg, loc="lower center", ncol=min(len(leg), 6), fontsize=FS(12), bbox_to_anchor=(0.5, -0.155)
        )
        for text in leg_obj.get_texts():
            text.set_fontweight("bold")
        # keep the panels themselves as large as possible; the legend hangs below
        fig.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.13, wspace=0.18)
        return fig

    scales = _norm_scales(opts.get("x_scale"), "linear")
    savefig(_make(scales[0]), out_dir / "pareto_scatter.png")
    if len(scales) > 1:
        savefig(_make(scales[1]), out_dir / "pareto_scatter_log.png")


def gen_kpi_bar(dfm: pd.DataFrame, metric: str, ylabel: str, fname: str, ctx: dict, out_dir: Path, opts: dict) -> None:
    """Scenario×strategy bars (mean ± min/max across constructors), improvers as paired bars."""
    scenarios, dists, strategies, improvers = ctx["scenarios"], ctx["dists"], ctx["strategies"], ctx["improvers"]

    def _make(yscale: str) -> plt.Figure:
        fig, axes = _panel_grid(len(dists), (12, 7))
        pass
        n_imp = len(improvers)
        width = 0.8 / n_imp
        for ax, dist in zip(axes, dists, strict=True):
            scen_d = [s for s in scenarios if s["dist"] == dist]
            groups = [(s, strat) for s in scen_d for strat in strategies]
            labels = [f"{region_label(s['city'], s['N'])}\n{strat}" for s, strat in groups]
            x = np.arange(len(groups), dtype=float)
            for ii, imp in enumerate(improvers):
                means, lo, hi = [], [], []
                for s, strat in groups:
                    sub = _scen_sub(dfm[(dfm.improver == imp) & (dfm.strategy == strat)], s)[metric] # pyrefly: ignore [bad-argument-type]
                    if len(sub):
                        m = sub.mean()
                        means.append(m)
                        lo.append(m - sub.min())
                        hi.append(sub.max() - m)
                    else:
                        means.append(np.nan)
                        lo.append(np.nan)
                        hi.append(np.nan)
                xs = x + (ii - (n_imp - 1) / 2) * width
                colors = [META["strategy_colors"].get(strat, "#a0a0a0") for _, strat in groups]
                valid = ~np.isnan(means)
                ax.bar(
                    xs[valid],
                    np.array(means)[valid],
                    width * 0.92,
                    color=np.array(colors)[valid],
                    alpha=0.9 if ii == 0 else 0.6,
                    hatch=HATCH_CYCLE[ii % len(HATCH_CYCLE)],
                    edgecolor=ctx["theme"]["fg"],
                    linewidth=0.4,
                )
                for xi, m_, l_, h_ in zip(xs, means, lo, hi, strict=True):
                    if not np.isnan(m_):
                        ax.errorbar(
                            xi,
                            m_,
                            yerr=[[max(l_, 0)], [max(h_, 0)]],
                            fmt="none",
                            color=ctx["theme"]["fg"],
                            capsize=2,
                            linewidth=0.9,
                        )
            if yscale != "linear":
                ax.set_yscale("symlog", linthresh=1)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=FS(10))
            ax.set_ylabel(f"{ylabel} ({ctx['n_days']} days)", fontsize=FS(13))
            ax.set_title(f"{dist}", fontsize=FS(13))
            ax.tick_params(axis="y", labelsize=11)
            ax.yaxis.grid(True, alpha=0.4)
            ax.set_axisbelow(True)
        patches = [mpatches.Patch(color=META["strategy_colors"].get(s, "#a0a0a0"), label=s) for s in strategies]
        patches += [
            mpatches.Patch(facecolor="#909090", hatch=HATCH_CYCLE[i % len(HATCH_CYCLE)], label=imp)
            for i, imp in enumerate(improvers)
        ]
        fig.legend(handles=patches, loc="lower center", ncol=len(patches), fontsize=FS(14), bbox_to_anchor=(0.5, -0.05))
        plt.tight_layout()
        return fig

    scales = _norm_scales(opts.get("y_scale"), "linear")
    savefig(_make(scales[0]), out_dir / f"{fname}.png")
    if len(scales) > 1:
        savefig(_make(scales[1]), out_dir / f"{fname}_log.png")


def _kpi_panel(ax: plt.Axes, dfm: pd.DataFrame, metric: str, ctx: dict, use_log: bool) -> None:
    """Draw one scenario x strategy bar panel (mean +/- range across constructors) for `metric`."""
    scenarios, strategies, improvers = ctx["scenarios"], ctx["strategies"], ctx["improvers"]
    n_imp = len(improvers)
    width = 0.8 / n_imp
    groups = [(s, strat) for s in scenarios for strat in strategies]
    labels = [f"{region_label(s['city'], s['N'])}\n{strat}" for s, strat in groups]
    x = np.arange(len(groups), dtype=float)
    for ii, imp in enumerate(improvers):
        means, lo, hi = [], [], []
        for s, strat in groups:
            sub = _scen_sub(dfm[(dfm.improver == imp) & (dfm.strategy == strat)], s)[metric] # pyrefly: ignore [bad-argument-type]
            if len(sub):
                m = sub.mean()
                means.append(m)
                lo.append(m - sub.min())
                hi.append(sub.max() - m)
            else:
                means.append(np.nan)
                lo.append(np.nan)
                hi.append(np.nan)
        xs = x + (ii - (n_imp - 1) / 2) * width
        colors = [META["strategy_colors"].get(strat, "#a0a0a0") for _, strat in groups]
        valid = ~np.isnan(means)
        ax.bar(
            xs[valid], np.array(means)[valid], width * 0.92, color=np.array(colors)[valid],
            alpha=0.9 if ii == 0 else 0.6, hatch=HATCH_CYCLE[ii % len(HATCH_CYCLE)],
            edgecolor=ctx["theme"]["fg"], linewidth=0.5,
        )
        for xi, m_, l_, h_ in zip(xs, means, lo, hi, strict=True):
            if not np.isnan(m_):
                ax.errorbar(
                    xi, m_, yerr=[[max(l_, 0)], [max(h_, 0)]], fmt="none",
                    color=ctx["theme"]["fg"], capsize=3, linewidth=1.2,
                )
    if use_log:
        ax.set_yscale("symlog", linthresh=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS(10.5), fontweight="bold")
    # Long region tokens (FFZ-350) overlap their neighbours at the shared
    # size — shrink just those labels slightly.
    for lbl in ax.get_xticklabels():
        if lbl.get_text().startswith("FFZ"):
            lbl.set_fontsize(FS(9.2))
    ax.tick_params(axis="y", labelsize=13)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)


def gen_kpi_combined(dfm: pd.DataFrame, ctx: dict, out_dir: Path) -> None:
    """Single 2x2 figure combining the overflow (log) and kg/km (linear) KPI bar panels, one column per distribution."""
    dists = ctx["dists"]
    metrics = [("overflows", "Overflow Count", True), ("kgkm", f"{KGKM_LABEL} Efficiency", False)]
    fig, axes = plt.subplots(2, len(dists), figsize=(13 * len(dists), 14), squeeze=False)
    pass
    for mi, (metric, ylabel, use_log) in enumerate(metrics):
        for di, dist in enumerate(dists):
            ax = axes[mi][di]
            dist_ctx = {**ctx, "scenarios": [s for s in ctx["scenarios"] if s["dist"] == dist]}
            _kpi_panel(ax, dfm, metric, dist_ctx, use_log)
            ax.set_ylabel(ylabel, fontsize=FS(22), fontweight="bold")
            ax.set_title(dist, fontsize=FS(22), fontweight="bold")
    strategies, improvers = ctx["strategies"], ctx["improvers"]
    patches = [mpatches.Patch(color=META["strategy_colors"].get(s, "#a0a0a0"), label=s) for s in strategies]
    patches += [
        mpatches.Patch(facecolor="#909090", hatch=HATCH_CYCLE[i % len(HATCH_CYCLE)], label=imp)
        for i, imp in enumerate(improvers)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches), fontsize=FS(16), bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.1, hspace=0.3, wspace=0.25)
    savefig(fig, out_dir / "kpi_combined.png")


def gen_km_violin(df: pd.DataFrame, ctx: dict, out_dir: Path) -> None:
    """Violin plots of total km per strategy × scenario (constructors + improvers pooled)."""
    scenarios, dists, strategies = ctx["scenarios"], ctx["dists"], ctx["strategies"]
    fig, axes = _panel_grid(len(dists), (12, 7))
    pass
    for ax, dist in zip(axes, dists, strict=True):
        scen_d = [s for s in scenarios if s["dist"] == dist]
        groups, labels, strat_of = [], [], []
        for strat in strategies:
            for s in scen_d:
                grp = _scen_sub(df[df.strategy == strat], s)["km"].values # pyrefly: ignore [bad-argument-type]
                if len(grp) == 0:
                    grp = np.array([0.0, 0.0])
                elif len(grp) == 1:
                    grp = np.array([grp[0], grp[0]])
                groups.append(grp)
                labels.append(f"{strat}\n{region_label(s['city'], s['N'])}")
                strat_of.append(strat)
        if not groups:
            continue
        parts = ax.violinplot(groups, positions=range(len(groups)), showmedians=True, showextrema=True)
        for body, strat in zip(parts["bodies"], strat_of, strict=True):
            body.set_facecolor(META["strategy_colors"].get(strat, "#a0a0a0"))
            body.set_alpha(0.7)
        for key in ["cmedians", "cmins", "cmaxes", "cbars"]:
            parts[key].set_color(ctx["theme"]["accent_line"] if key == "cmedians" else ctx["theme"]["guide_line"])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=FS(8))
        ax.set_ylabel(f"Total km ({ctx['n_days']} days)", fontsize=FS(10))
        ax.set_title(f"Distance Distribution — {dist}", fontsize=FS(11))
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
    patches = [mpatches.Patch(color=META["strategy_colors"].get(s, "#a0a0a0"), label=s) for s in strategies]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches), fontsize=FS(11), bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    savefig(fig, out_dir / "km_violin.png")


def _policy_rows(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    combos = df[["variant", "constructor", "improver"]].drop_duplicates().values.tolist()
    return sorted((tuple(c) for c in combos), key=lambda c: (c[0], c[1], c[2]))


def gen_policy_scenario_heatmap(df: pd.DataFrame, ctx: dict, out_dir: Path) -> None:
    """Heatmaps with policy configurations on the rows and scenarios on the columns."""
    scenarios = ctx["scenarios"]
    rows = _policy_rows(df)
    row_labels = [f"{v} · {disp(c)} · {i}" for v, c, i in rows]
    col_labels = [scenario_label(s) for s in scenarios]
    for metric, cmap, mlabel in [
        ("overflows", "RdYlGn_r", "Overflow Count"), ("kgkm", "RdYlGn", f"{KGKM_LABEL} Efficiency")
    ]:
        mat = np.full((len(rows), len(scenarios)), np.nan)
        for ri, (v, c, i) in enumerate(rows):
            sub_p = df[(df.variant == v) & (df.constructor == c) & (df.improver == i)]
            for ci, s in enumerate(scenarios):
                vals = _scen_sub(sub_p, s)[metric] # pyrefly: ignore [bad-argument-type]
                if len(vals):
                    mat[ri, ci] = vals.mean()
        fig, ax = plt.subplots(figsize=(max(12, 1.9 * len(scenarios)), max(9, 0.4 * len(rows)) + 1.4))
        ax.set_title(
            f"Policy × Scenario Heatmap — {mlabel} ({ctx['n_days']} days)", fontsize=FS(16), fontweight="bold", pad=16
        )
        norm = matplotlib.colors.SymLogNorm(linthresh=10, vmin=0) if metric == "overflows" else None
        im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(mlabel, fontsize=FS(22), fontweight="bold")
        cbar.ax.tick_params(labelsize=13)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=FS(13), fontweight="bold", rotation=30, ha="right")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=FS(12), fontweight="bold")
        plt.tight_layout()
        savefig(fig, out_dir / f"policy_scenario_heatmap_{metric}.png")


def gen_scenario_constructor_heatmap(dfm: pd.DataFrame, ctx: dict, out_dir: Path, shared_axis_labels: bool = True, out_fname: str | None = None) -> None:
    """One heatmap panel per scenario: constructors on rows, strategy × improver on columns.

    `shared_axis_labels`, when False, suppresses the per-panel row/column tick
    labels entirely (the deck then shows a single shared legend to the side —
    see gen_presentation.py's side_legend option) so the figure itself can use
    the freed-up space to grow.
    """
    scenarios, strategies, improvers, constructors = (
        ctx["scenarios"],
        ctx["strategies"],
        ctx["improvers"],
        ctx["constructors"],
    )
    combos = [(s, i) for s in strategies for i in improvers]
    combo_labels = [f"{s}\n{i}" for s, i in combos]
    n = len(scenarios)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 7.6), squeeze=False)
    pass
    for row_i, (metric, cmap, mlabel) in enumerate(
        [("overflows", "RdYlGn_r", "Overflows"), ("kgkm", "RdYlGn", KGKM_LABEL)]
    ):
        for col_i, s in enumerate(scenarios):
            ax = axes[row_i][col_i]
            mat = np.full((len(constructors), len(combos)), np.nan)
            sub_s = _scen_sub(dfm, s)
            for ci, con in enumerate(constructors):
                for bi, (strat, imp) in enumerate(combos):
                    vals = sub_s[(sub_s.constructor == con) & (sub_s.strategy == strat) & (sub_s.improver == imp)][
                        metric
                    ]
                    if len(vals):
                        mat[ci, bi] = vals.mean()
            norm = matplotlib.colors.SymLogNorm(linthresh=10, vmin=0) if metric == "overflows" else None
            im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)
            cbar = plt.colorbar(im, ax=ax, shrink=0.75)
            cbar.ax.tick_params(labelsize=10)
            if shared_axis_labels:
                ax.set_xticks(range(len(combo_labels)))
                ax.set_xticklabels(combo_labels, fontsize=FS(8.6), fontweight="bold", rotation=0)
                ax.set_yticks(range(len(constructors)))
                ax.set_yticklabels(disp_all(constructors) if col_i == 0 else [], fontsize=FS(9.6), fontweight="bold")
            else:
                ax.set_xticks([])
                ax.set_yticks([])
            if row_i == 0:
                ax.set_title(scenario_label(s), fontsize=FS(10.8), fontweight="bold")
            if col_i == n - 1 and shared_axis_labels:
                ax.text(
                    1.28, 0.5, mlabel.upper(), transform=ax.transAxes, rotation=270, va="center", ha="left",
                    fontsize=FS(14), fontweight="bold", color=ctx["theme"]["fg"],
                )
    plt.tight_layout()
    if out_fname:
        fname = out_fname
    else:
        fname = "scenario_constructor_heatmap.png" if shared_axis_labels else "scenario_constructor_heatmap_full.png"
    savefig(fig, out_dir / fname)


def _annotate_no_overlap(ax, placed: list, x: float, y: float, text: str, fontsize: float,
                          fontweight: str = "bold") -> None:
    """Place a point-label near (x, y), nudging it through a few candidate offsets to dodge
    previously-placed labels on the same axes (approximate, but enough for a handful of points)."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    log_x = ax.get_xscale() in ("log", "symlog")

    def _norm_x(v: float) -> float:
        if log_x:
            lo, hi = np.log10(max(abs(xlim[0]), 1e-6)), np.log10(max(abs(xlim[1]), 1e-6))
            return (np.log10(max(abs(v), 1e-6)) - lo) / (hi - lo + 1e-9)
        return (v - xlim[0]) / (xlim[1] - xlim[0] + 1e-9)

    def _norm_y(v: float) -> float:
        return (v - ylim[0]) / (ylim[1] - ylim[0] + 1e-9)

    nx, ny = _norm_x(x), _norm_y(y)
    candidates = [
        (8, 7), (8, -20), (-58, 7), (-58, -20), (8, 24), (-58, 24), (8, -36), (-58, -36),
        (30, 34), (30, -40), (-80, 7), (8, 40),
    ]
    chosen = candidates[0]
    for dx, dy in candidates:
        onx, ony = nx + dx / 420, ny + dy / 230
        if all((onx - px) ** 2 + (ony - py) ** 2 > 0.018 for px, py in placed):
            chosen = (dx, dy)
            placed.append((onx, ony))
            break
    else:
        dx, dy = chosen
        placed.append((nx + dx / 480, ny + dy / 260))
    ax.annotate(
        text, (x, y), textcoords="offset points", xytext=chosen, fontsize=fontsize, fontweight=fontweight,
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8), zorder=6,
    )


def gen_strategy_bubble(dfm: pd.DataFrame, ctx: dict, out_dir: Path) -> None:
    """Bubble chart per distribution: one bubble per (strategy, scenario), size ∝ N."""
    opts = ctx["charts"].get("strategy_bubble", {})
    scenarios, dists, strategies = ctx["scenarios"], ctx["dists"], ctx["strategies"]
    regions = ctx["regions"]
    markers = {tuple(r): MARKER_CYCLE[i % len(MARKER_CYCLE)] for i, r in enumerate(regions)}

    def _make(xscale: str) -> plt.Figure:
        fig, axes = _panel_grid(len(dists), (10, 8))
        pass
        for ax, dist in zip(axes, dists, strict=True):
            placed: list = []
            if xscale != "linear":
                ax.set_xscale("symlog", linthresh=1)
            for s in [sc for sc in scenarios if sc["dist"] == dist]:
                for strat in strategies:
                    sub = _scen_sub(dfm[dfm.strategy == strat], s) # pyrefly: ignore [bad-argument-type]
                    if sub.empty:
                        continue
                    ov, eff = sub["overflows"].mean(), sub["kgkm"].mean()
                    ax.scatter(
                        ov,
                        eff,
                        c=META["strategy_colors"].get(strat, "#a0a0a0"),
                        marker=markers[(s["city"], s["N"])],
                        s=120 + 2 * s["N"],
                        alpha=0.75,
                        edgecolors=ctx["theme"]["accent_line"],
                        linewidths=0.5,
                    )
                    _annotate_no_overlap(ax, placed, ov, eff, region_label(s["city"], s["N"]), fontsize=FS_LABEL(11))
            ax.set_xlabel("Overflows", fontsize=FS_AXIS(18), fontweight="bold")
            ax.set_ylabel(f"{KGKM_LABEL} Efficiency", fontsize=FS_AXIS(18), fontweight="bold")
            ax.set_title(f"{dist}", fontsize=FS_AXIS(18), fontweight="bold")
            ax.tick_params(axis="both", labelsize=12)
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontweight("bold")
            ax.yaxis.grid(True, alpha=0.4)
            ax.xaxis.grid(True, alpha=0.4)
            ax.set_axisbelow(True)
        patches = [mpatches.Patch(color=META["strategy_colors"].get(s, "#a0a0a0"), label=s) for s in strategies]
        leg = fig.legend(handles=patches, loc="lower center", ncol=len(patches), fontsize=FS(15), bbox_to_anchor=(0.5, -0.05))
        for text in leg.get_texts():
            text.set_fontweight("bold")
        plt.tight_layout()
        return fig

    scales = _norm_scales(opts.get("x_scale"), "linear")
    savefig(_make(scales[0]), out_dir / "strategy_bubble.png")
    if len(scales) > 1:
        savefig(_make(scales[1]), out_dir / "strategy_bubble_log.png")


def gen_improver_bubble(dfm: pd.DataFrame, ctx: dict, out_dir: Path) -> None:
    """Bubble chart per distribution comparing route improvers (strategies pooled)."""
    opts = ctx["charts"].get("improver_bubble", {})
    scenarios, dists, improvers = ctx["scenarios"], ctx["dists"], ctx["improvers"]
    regions = ctx["regions"]
    markers = {tuple(r): MARKER_CYCLE[i % len(MARKER_CYCLE)] for i, r in enumerate(regions)}

    def _make(xscale: str) -> plt.Figure:
        fig, axes = _panel_grid(len(dists), (10, 8))
        pass
        for ax, dist in zip(axes, dists, strict=True):
            if xscale != "linear":
                ax.set_xscale("symlog", linthresh=1)
            placed: list = []
            for s in [sc for sc in scenarios if sc["dist"] == dist]:
                pts = {}
                for imp in improvers:
                    sub = _scen_sub(dfm[dfm.improver == imp], s) # pyrefly: ignore [bad-argument-type]
                    if sub.empty:
                        continue
                    ov, eff = sub["overflows"].mean(), sub["kgkm"].mean()
                    pts[imp] = (ov, eff)
                    ax.scatter(
                        ov,
                        eff,
                        c=META["improver_colors"].get(imp, "#a0a0a0"),
                        marker=markers[(s["city"], s["N"])],
                        s=120 + 2 * s["N"],
                        alpha=0.8,
                        edgecolors=ctx["theme"]["accent_line"],
                        linewidths=0.5,
                    )
                if len(pts) == 2:
                    (x1, y1), (x2, y2) = pts.values()
                    ax.plot([x1, x2], [y1, y2], "-", color=ctx["theme"]["guide_line"], linewidth=0.8, alpha=0.6)
                for _imp, (ov, eff) in pts.items():
                    _annotate_no_overlap(ax, placed, ov, eff, region_label(s["city"], s["N"]), fontsize=FS_LABEL(10))
            ax.set_xlabel("Overflows", fontsize=FS_AXIS(18), fontweight="bold")
            ax.set_ylabel(f"{KGKM_LABEL} Efficiency", fontsize=FS_AXIS(18), fontweight="bold")
            ax.set_title(f"{dist}", fontsize=FS_AXIS(18), fontweight="bold")
            ax.tick_params(axis="both", labelsize=12)
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontweight("bold")
            ax.yaxis.grid(True, alpha=0.4)
            ax.xaxis.grid(True, alpha=0.4)
            ax.set_axisbelow(True)
        patches = [mpatches.Patch(color=META["improver_colors"].get(i, "#a0a0a0"), label=i) for i in improvers]
        leg = fig.legend(handles=patches, loc="lower center", ncol=len(patches), fontsize=FS(15), bbox_to_anchor=(0.5, -0.05))
        for text in leg.get_texts():
            text.set_fontweight("bold")
        plt.tight_layout()
        return fig

    scales = _norm_scales(opts.get("x_scale"), "linear")
    savefig(_make(scales[0]), out_dir / "improver_bubble.png")
    if len(scales) > 1:
        savefig(_make(scales[1]), out_dir / "improver_bubble_log.png")


def gen_constructor_ranking(dfm: pd.DataFrame, ctx: dict, out_dir: Path) -> None:
    """Average rank of each constructor across scenarios × strategies × improvers."""
    constructors = ctx["constructors"]
    metrics = ["overflows", "kgkm", "km", "profit"]
    metric_labels = ["Overflows", KGKM_LABEL, "km", "Profit"]
    rank_asc = [True, False, True, False]
    fig, ax = plt.subplots(figsize=(max(12, 1.8 * len(constructors)), 8))
    pass
    mean_ranks: dict = {c: {m: [] for m in metrics} for c in constructors}
    for metric, asc in zip(metrics, rank_asc, strict=True):
        for _, grp in dfm.groupby(["city", "N", "dist", "strategy", "improver"]):
            gi = grp.set_index("constructor")
            r = gi[metric].rank(ascending=asc, method="average")
            for c in constructors:
                if c in r.index:
                    mean_ranks[c][metric].append(r[c])
    x = np.arange(len(constructors))
    w = 0.8 / len(metrics)
    for mi, (metric, label, color) in enumerate(zip(metrics, metric_labels, META["metric_colors"], strict=True)):
        vals = [np.mean(mean_ranks[c][metric]) if mean_ranks[c][metric] else np.nan for c in constructors]
        ax.bar(x + (mi - (len(metrics) - 1) / 2) * w, vals, w * 0.9, label=label, color=color, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(disp_all(constructors), rotation=15, ha="right", fontsize=FS(10))
    ax.set_ylabel("Average rank (lower = better)", fontsize=FS(11))
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    plt.tight_layout()
    savefig(fig, out_dir / "constructor_ranking.png")


_RADAR_BG = "#1a1a2e"
_RADAR_AX_BG = "#16213e"
_RADAR_MUTED = "#a0a0b0"
_RADAR_FAINT = "#2d2d4e"


def _render_radar(
    dfm: pd.DataFrame,
    ctx: dict,
    key: list[str],
    title: str,
    figsize: tuple[float, float],
    legend_anchor: tuple[float, float],
) -> plt.Figure:
    """Shared normalised-radar renderer for a given set of constructors `key`."""
    metrics = ["overflows", "kgkm", "km", "profit"]
    axes_labels = ["Overflows\n(fewer ↓)", f"{KGKM_LABEL}\n(higher ↑)", "km\n(fewer ↓)", "Profit\n(higher ↑)"]
    invert = [True, False, True, False]

    scores = {}
    for c in key:
        sub = dfm[dfm.constructor == c]
        scores[c] = []
        for metric, inv in zip(metrics, invert, strict=True):
            all_vals = dfm[metric].values
            v = sub[metric].mean() if len(sub) else np.nanmean(all_vals)  # pyrefly: ignore [no-matching-overload]
            mn, mx = np.nanmin(all_vals), np.nanmax(all_vals)  # pyrefly: ignore [no-matching-overload]
            norm = (v - mn) / (mx - mn + 1e-9) if mx > mn else 0.5
            scores[c].append(1 - norm if inv else norm)

    n_axes = len(metrics)
    angles = [i / n_axes * 2 * np.pi for i in range(n_axes)]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    # Explicit dark background regardless of the active mplstyle
    fig.patch.set_facecolor(_RADAR_BG)
    ax.patch.set_facecolor(_RADAR_AX_BG)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, fontsize=FS(15), color="#ffffff")
    ax.tick_params(axis="x", pad=72, colors="#ffffff")
    ax.set_ylim(0, 1)
    ax.yaxis.set_tick_params(labelcolor="#ffffff", labelsize=10)
    ax.spines["polar"].set_color(_RADAR_MUTED)
    for r, lbl in zip([0.25, 0.5, 0.75, 1.0], ["25%", "50%", "75%", "100%"], strict=True):
        ax.plot(angles, [r] * (n_axes + 1), "--", color=_RADAR_FAINT, linewidth=0.8)
        ax.text(0, r + 0.02, lbl, ha="center", va="bottom", fontsize=FS(10), color="#ffffff")
    for c in key:
        vals = scores[c] + scores[c][:1]
        color = META["constructor_colors"].get(c, "#e0e0e0")
        ax.plot(angles, vals, "o-", color=color, linewidth=2.5, markersize=5, label=disp(c))
        ax.fill(angles, vals, color=color, alpha=0.08)
    ax.set_title(title, fontsize=FS(15), fontweight="bold", pad=36, color="#ffffff")
    legend = ax.legend(loc="upper right", bbox_to_anchor=legend_anchor, fontsize=FS(13))
    legend.get_frame().set_facecolor(_RADAR_BG)
    legend.get_frame().set_edgecolor(_RADAR_MUTED)
    for text in legend.get_texts():
        text.set_color("#ffffff")
    fig.subplots_adjust(top=0.82)
    return fig


def gen_radar(dfm: pd.DataFrame, ctx: dict, out_dir: Path) -> None:
    """Normalised radar chart for a curated set of key constructors."""
    opts = ctx["charts"].get("radar", {})
    key = [c for c in opts.get("constructors", ctx["constructors"][:4]) if c in ctx["constructors"]]
    if not key:
        key = ctx["constructors"][:4]
    fig = _render_radar(dfm, ctx, key, "Policy Performance Radar\n(normalised; outer = better)", (10, 10), (1.25, 1.1))
    out_path = out_dir / "policy_radar.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def gen_radar_combined(dfm: pd.DataFrame, ctx: dict, out_dir: Path) -> None:
    """Normalised radar chart overlaying every route constructor (not just the curated subset)."""
    key = list(ctx["constructors"])
    fig = _render_radar(
        dfm,
        ctx,
        key,
        "Policy Performance Radar — All Constructors\n(normalised; outer = better)",
        (11, 10),
        (1.45, 1.1),
    )
    out_path = out_dir / "policy_radar_combined.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def gen_improver_delta(dfm: pd.DataFrame, ctx: dict, out_dir: Path) -> None:
    """Delta heatmap (last improver − first improver) per constructor × configuration."""
    improvers = ctx["improvers"]
    if len(improvers) < 2:
        return
    imp_a, imp_b = improvers[0], improvers[-1]
    scenarios, strategies, constructors = ctx["scenarios"], ctx["strategies"], ctx["constructors"]
    configs = [(s, strat) for s in scenarios for strat in strategies]
    col_labels = [f"{region_label(s['city'], s['N'])}\n{s['dist'][:3]}\n{strat}" for s, strat in configs]
    fig, axes = plt.subplots(1, 2, figsize=(28, 10))
    pass
    for ax, (metric, cmap) in zip(axes, [("overflows", "RdYlGn_r"), ("kgkm", "RdYlGn")], strict=True):
        mat = np.full((len(constructors), len(configs)), np.nan)
        for ci, c in enumerate(constructors):
            for cfi, (s, strat) in enumerate(configs):
                sub = _scen_sub(dfm[(dfm.strategy == strat) & (dfm.constructor == c)], s) # pyrefly: ignore [bad-argument-type]
                a = sub[sub.improver == imp_a][metric]
                b = sub[sub.improver == imp_b][metric]
                if len(a) and len(b):
                    mat[ci, cfi] = b.values[0] - a.values[0]
        finite = mat[~np.isnan(mat)]
        vmax = np.nanpercentile(np.abs(finite), 95) if len(finite) else 1
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.8, label=f"Δ {metric}")
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=FS(6), rotation=45, ha="right")
        ax.set_yticks(range(len(constructors)))
        ax.set_yticklabels(disp_all(constructors), fontsize=FS(9))
        ax.set_title(f"Δ {metric}", fontsize=FS(11))
    plt.tight_layout()
    savefig(fig, out_dir / "improver_delta.png")


# ── Interactive (plotly) figures ───────────────────────────────────────────────


def gen_interactive_html(df: pd.DataFrame, dfm: pd.DataFrame, ctx: dict, out_dir: Path) -> dict[str, Path]:  # noqa: C901
    if not HAS_PLOTLY:
        print("  [WARN] Plotly not available — skipping interactive HTML generation")
        return {}
    theme = ctx["theme"]
    template = theme["plotly_template"]
    scenarios, improvers = ctx["scenarios"], ctx["improvers"]
    regions = ctx["regions"]
    region_syms = {tuple(r): PLOTLY_SYMBOLS[i % len(PLOTLY_SYMBOLS)] for i, r in enumerate(regions)}
    variants = sorted(df["variant"].unique())
    vcolors = {v: variant_color(v, i) for i, v in enumerate(variants)}
    paths: dict[str, Path] = {}

    # pareto_scatter_interactive ────────────────────────────────────────────────
    fig = go.Figure()
    legend_shown: set = set()
    for s in scenarios:
        sub_s = _scen_sub(df, s)
        if sub_s.empty:
            continue
        sym_base = region_syms[(s["city"], s["N"])]
        for variant, sg in sub_s.groupby("variant"):
            color = vcolors[variant]
            key = (variant, s["dist"])
            show = key not in legend_shown
            legend_shown.add(key)
            syms = [sym_base if r.improver == improvers[0] else f"{sym_base}-open" for r in sg.itertuples()]
            fig.add_trace(
                go.Scatter(
                    x=sg["overflows"],
                    y=sg["kgkm"],
                    mode="markers",
                    name=f"{variant} ({s['dist'][:3]})",
                    legendgroup=f"{variant}_{s['dist']}",
                    showlegend=show,
                    marker=dict(symbol=syms, color=color, size=8, opacity=0.85, line=dict(width=1.5, color=color)),
                    text=[
                        f"Constructor: {r.constructor}<br>Selection: {r.variant}<br>"
                        f"Scenario: {scenario_label(s)}<br>Improver: {r.improver}<br>"
                        f"Overflows: {r.overflows:.1f}<br>kg/km: {r.kgkm:.3f}<br>"
                        f"km: {r.km:.0f}<br>Profit: {r.profit:.0f}"
                        for r in sg.itertuples()
                    ],
                    hovertemplate="%{text}<extra></extra>",
                )
            )
    n_scatter_traces = len(fig.data)  # pyrefly: ignore [bad-argument-type]

    # Pareto front lines + front-only markers per scenario (one colour per scenario)
    front_colors = {
        (s["city"], s["N"], s["dist"]): META["scenario_colors"][i % len(META["scenario_colors"])]
        for i, s in enumerate(scenarios)
    }
    for s in scenarios:
        sub = _scen_sub(df, s).reset_index(drop=True)
        if sub.empty:
            continue
        front = pareto_indices(sub["overflows"].values, sub["kgkm"].values)
        pts = sorted((float(sub["overflows"][i]), float(sub["kgkm"][i])) for i in front)
        sx, sy = [pts[0][0]], [pts[0][1]]
        for j in range(1, len(pts)):
            sx += [pts[j][0], pts[j][0]]
            sy += [pts[j - 1][1], pts[j][1]]
        fig.add_trace(
            go.Scatter(
                x=sx,
                y=sy,
                mode="lines",
                name=f"Pareto — {scenario_label(s)}",
                line=dict(width=2, dash="dash", color=front_colors[(s["city"], s["N"], s["dist"])]),
                hoverinfo="skip",
            )
        )
    n_line_traces = len(fig.data) - n_scatter_traces  # pyrefly: ignore [bad-argument-type]
    for s in scenarios:
        sub = _scen_sub(df, s).reset_index(drop=True)
        if sub.empty:
            continue
        front_rows = sub.iloc[sorted(pareto_indices(sub["overflows"].values, sub["kgkm"].values))]
        sym_base = region_syms[(s["city"], s["N"])]
        fig.add_trace(
            go.Scatter(
                x=front_rows["overflows"],
                y=front_rows["kgkm"],
                mode="markers",
                name=f"★ Pareto — {scenario_label(s)}",
                marker=dict(
                    symbol=[
                        sym_base if r.improver == improvers[0] else f"{sym_base}-open" for r in front_rows.itertuples()
                    ],
                    color=[vcolors[r.variant] for r in front_rows.itertuples()],
                    size=13,
                    opacity=1.0,
                ),
                text=[
                    f"Constructor: {r.constructor}<br>Selection: {r.variant}<br>"
                    f"Scenario: {scenario_label(s)}<br>Improver: {r.improver}<br>"
                    f"Overflows: {r.overflows:.1f}<br>kg/km: {r.kgkm:.3f}<br><b>★ Pareto optimal</b>"
                    for r in front_rows.itertuples()
                ],
                hovertemplate="%{text}<extra></extra>",
                visible=False,
            )
        )
    n_front_traces = len(fig.data) - n_scatter_traces - n_line_traces  # pyrefly: ignore [bad-argument-type]
    all_vis = [True] * n_scatter_traces + [True] * n_line_traces + [False] * n_front_traces
    front_vis = [False] * n_scatter_traces + [True] * n_line_traces + [True] * n_front_traces
    fig.update_layout(
        title=dict(
            text=f"Overflow vs Efficiency — All Runs ({ctx['n_days']} days; hover for details)", x=0.5, xanchor="center"
        ),
        xaxis_title=f"Overflows ({ctx['n_days']} days)",
        yaxis_title="kg/km",
        template=template,
        height=750,
        hovermode="closest",
        margin=dict(t=80, b=90),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.5,
                y=-0.08,
                xanchor="center",
                yanchor="top",
                buttons=[
                    dict(label="All Points", method="update", args=[{"visible": all_vis}]),
                    dict(label="Pareto Front Only", method="update", args=[{"visible": front_vis}]),
                ],
                showactive=True,
                bgcolor=theme["button_bg"],
                bordercolor=theme["button_border"],
                font=dict(color=theme["fg"], size=13),
                pad=dict(l=10, r=10, t=5, b=5),
            )
        ],
    )
    btn_js = load_js(
        "pareto_buttons.js",
        div_id="pareto-scatter",
        btn_bg=theme["button_bg"],
        btn_border=theme["button_border"],
        btn_fg=theme["fg"],
    )
    p = out_dir / "pareto_scatter_interactive.html"
    fig.write_html(str(p), include_plotlyjs="cdn", div_id="pareto-scatter", post_script=btn_js)
    paths["pareto"] = p
    print(f"  Saved: {p.name}")

    # strategy_bubble_interactive ───────────────────────────────────────────────
    agg = (
        dfm.groupby(["city", "N", "dist", "strategy", "improver"])[["overflows", "kgkm", "km", "profit"]]
        .mean()
        .reset_index()
    )
    fig2 = go.Figure()
    leg_seen: set = set()
    for strat in ctx["strategies"]:
        for s in scenarios:
            sub2 = _scen_sub(agg[agg.strategy == strat], s) # pyrefly: ignore [bad-argument-type]
            if sub2.empty:
                continue
            sym_base = region_syms[(s["city"], s["N"])]
            key = (strat, s["dist"])
            show = key not in leg_seen
            leg_seen.add(key)
            fig2.add_trace(
                go.Scatter(
                    x=sub2["overflows"],
                    y=sub2["kgkm"],
                    mode="markers",
                    name=f"{strat} ({s['dist'][:3]})",
                    legendgroup=f"{strat}_{s['dist']}",
                    showlegend=show,
                    marker=dict(
                        color=META["strategy_colors"].get(strat, "gray"),
                        symbol=[
                            sym_base if r.improver == improvers[0] else f"{sym_base}-open" for r in sub2.itertuples()
                        ],
                        size=(sub2["N"] / 15 + 5).tolist(),
                        opacity=0.85,
                        line=dict(width=1.5, color=theme["fg"]),
                    ),
                    text=[
                        f"Strategy: {r.strategy}<br>Scenario: {scenario_label(s)}<br>"
                        f"Improver: {r.improver}<br>Mean overflows: {r.overflows:.2f}<br>"
                        f"Mean kg/km: {r.kgkm:.3f}"
                        for r in sub2.itertuples()
                    ],
                    hovertemplate="%{text}<extra></extra>",
                )
            )
    fig2.update_layout(
        title=dict(
            text="Strategy Trade-off Bubble Chart (bubble size ∝ N; open marker = second improver)",
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Mean overflows",
        yaxis_title="Mean kg/km",
        template=template,
        height=650,
        hovermode="closest",
        margin=dict(t=90),
    )
    p2 = out_dir / "strategy_bubble_interactive.html"
    fig2.write_html(str(p2), include_plotlyjs="cdn")
    paths["bubble"] = p2
    print(f"  Saved: {p2.name}")

    # policy_heatmap_interactive ────────────────────────────────────────────────
    rows = _policy_rows(df)
    row_labels = [f"{v} · {c} · {i}" for v, c, i in rows]
    col_labels = [scenario_label(s) for s in scenarios]
    mats = {}
    for metric in ["overflows", "kgkm"]:
        mat = np.full((len(rows), len(scenarios)), np.nan)
        for ri, (v, c, i) in enumerate(rows):
            sub_p = df[(df.variant == v) & (df.constructor == c) & (df.improver == i)]
            for ci, s in enumerate(scenarios):
                vals = _scen_sub(sub_p, s)[metric] # pyrefly: ignore [bad-argument-type]
                if len(vals):
                    mat[ri, ci] = vals.mean()
        mats[metric] = mat
    figH = go.Figure()
    figH.add_trace(
        go.Heatmap(
            z=mats["overflows"],
            x=col_labels,
            y=row_labels,
            colorscale="RdYlGn_r" if True else "RdYlGn",
            hovertemplate="Policy: %{y}<br>Scenario: %{x}<br>Value: %{z:.2f}<extra></extra>",
        )
    )
    figH.update_layout(
        title=dict(text="Policy × Scenario Heatmap", x=0.5, xanchor="center"),
        template=template,
        height=max(700, 22 * len(rows)),
        margin=dict(b=90, l=240),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.5,
                y=-0.05,
                xanchor="center",
                yanchor="top",
                buttons=[
                    dict(
                        label="Overflows",
                        method="restyle",
                        args=[{"z": [mats["overflows"].tolist()], "colorscale": "RdYlGn_r"}],
                    ),
                    dict(
                        label="kg/km",
                        method="restyle",
                        args=[{"z": [mats["kgkm"].tolist()], "colorscale": "RdYlGn"}],
                    ),
                ],
                showactive=True,
                bgcolor=theme["button_bg"],
                bordercolor=theme["button_border"],
                font=dict(color=theme["fg"], size=12),
            )
        ],
    )
    p3 = out_dir / "policy_heatmap_interactive.html"
    figH.write_html(str(p3), include_plotlyjs="cdn")
    paths["heatmap"] = p3
    print(f"  Saved: {p3.name}")
    return paths


# ── Bin-location maps ──────────────────────────────────────────────────────────

_COORD_DIR = Path("data/wsr_simulator/coordinates")


def _fix_stripped_decimal(val: float, lo: float, hi: float) -> float:
    """Recover a decimal point dropped during export (e.g. 401484222222222 -> 40.1484222222222)."""
    lo, hi = sorted([abs(lo), abs(hi)])
    s = abs(val)
    if lo <= s <= hi:
        return val
    for k in range(1, 18):
        cand = s / (10**k)
        if lo <= cand <= hi:
            return math.copysign(cand, val)
    return val


def _load_bin_coords(city: str) -> pd.DataFrame:
    """Every known bin's (lat, lon) for `city`, from the raw sensor/coordinate exports."""
    if city == "Rio Maior":
        df = pd.read_csv(_COORD_DIR / "out_info[riomaior].csv")
        df = df.rename(columns={"Latitude": "lat", "Longitude": "lon"})
    elif city == "Figueira da Foz":
        df = pd.read_csv(_COORD_DIR / "out_info[figdafoz].csv")
        df["lat"] = df["Latitude"].apply(lambda v: _fix_stripped_decimal(v, 36, 43))
        df["lon"] = df["Longitude"].apply(lambda v: _fix_stripped_decimal(v, -10, -6))
    else:
        raise ValueError(f"No coordinate source known for city {city!r}")
    return df.drop_duplicates(subset=["ID"])[["lat", "lon"]].dropna()


def gen_bin_location_map(city: str, out_path: Path, mode: str = "street") -> Path | None:
    """A map of every known bin location for `city` ('street' = real OSM road network basemap, else a
    plain lat/lon scatter)."""
    try:
        coords = _load_bin_coords(city)
    except FileNotFoundError as exc:
        print(f"  [WARN] Bin coordinates unavailable for {city}: {exc}")
        return None
    if coords.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 6.2))
    if mode == "street":
        try:
            import osmnx as ox

            graph = ox.graph_from_place(f"{city}, Portugal", network_type="drive")
            ox.plot_graph(
                graph, ax=ax, show=False, close=False, node_size=0, edge_color="#B9C2CC", edge_linewidth=0.6,
                bgcolor="white",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] Street basemap unavailable for {city} ({exc}); falling back to plain scatter")
            mode = "scatter"
    if mode != "street":
        ax.set_facecolor("white")
        ax.invert_yaxis()  # keep the same north-up orientation osmnx would use
        ax.set_aspect("auto")  # stretch to fill the wide figure rather than a strict geographic ratio
        ax.set_xlabel("Longitude", fontsize=FS(11))
        ax.set_ylabel("Latitude", fontsize=FS(11))

    ax.scatter(
        coords["lon"], coords["lat"], s=22, color="#C0392B", alpha=0.85, edgecolor="white", linewidth=0.4, zorder=5
    )
    ax.set_title(f"{city} — {len(coords)} Waste Bins", fontsize=FS(14), fontweight="bold")
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path.name} ({mode})")
    return out_path


def gen_bin_location_maps(out_dir: Path, mode: str = "street") -> None:
    gen_bin_location_map("Rio Maior", out_dir / "riomaior_map.png", mode)
    gen_bin_location_map("Figueira da Foz", out_dir / "figueiradafoz_map.png", mode)


def _load_selected_scenario_coords(city: str, n_bins: int) -> pd.DataFrame:
    """(lat, lon) of exactly the bins simulated in a scenario.

    Rio Maior: the exported selection files
    (`selected_coordinates/coordinates{N}_plastic[riomaior].xlsx`, which
    include the depot as ID 0 — dropped here). Figueira da Foz: the plastic
    ("Mistura de embalagens") bins of `out_info[figdafoz].csv` sorted by ID —
    the base frame the simulator builds — indexed by the positional selection
    in `data/wsr_simulator/bins_selection/graphs_{N}V_1N_plastic.json`.
    """
    if city == "Rio Maior":
        xlsx = _COORD_DIR / "selected_coordinates" / f"coordinates{n_bins}_plastic[riomaior].xlsx"
        df = pd.read_excel(xlsx).rename(columns={"Lat": "lat", "Lng": "lon"})
        df = df[df["ID"] != 0]  # ID 0 is the depot, not a bin
        return df[["lat", "lon"]].dropna()
    if city == "Figueira da Foz":
        info = pd.read_csv(_COORD_DIR / "out_info[figdafoz].csv")
        info = info[info["description"] == "Mistura de embalagens"]
        info = info.assign(
            lat=info["Latitude"].apply(lambda v: _fix_stripped_decimal(v, 36, 43)),
            lon=info["Longitude"].apply(lambda v: _fix_stripped_decimal(v, -10, -6)),
        )
        base = info.drop_duplicates(subset=["ID"]).sort_values("ID").reset_index(drop=True)
        sel_path = _COORD_DIR.parent / "bins_selection" / f"graphs_{n_bins}V_1N_plastic.json"
        idx = json.loads(sel_path.read_text(encoding="utf-8"))[0]
        picked = base.iloc[[i for i in idx if i < len(base)]]
        return picked[["lat", "lon"]].dropna()
    raise ValueError(f"No selected-bin coordinate source for city {city!r}")


def gen_selected_bin_maps(out_dir: Path, mode: str = "street") -> None:
    """Generate per-scenario selected-bin maps (RM-100, RM-170, FFZ-350).

    Loads the actual per-scenario bin selections (see
    `_load_selected_scenario_coords`); if that fails, falls back to copying the
    corresponding all-bin map so the slide still has an image placeholder.
    """
    _SCENARIO_SPECS = [
        ("Rio Maior",       100, "riomaior100_selected_map.png",      "riomaior_map.png"),
        ("Rio Maior",       170, "riomaior170_selected_map.png",      "riomaior_map.png"),
        ("Figueira da Foz", 350, "figueiradafoz350_selected_map.png", "figueiradafoz_map.png"),
    ]
    import shutil as _shutil

    for city, n_bins, out_name, fallback_name in _SCENARIO_SPECS:
        out_path = out_dir / out_name
        try:
            coords = _load_selected_scenario_coords(city, n_bins)
            if coords.empty:
                raise ValueError("selection resolved to zero bins")
            fig, ax = plt.subplots(figsize=(9, 6.2))
            if mode == "street":
                try:
                    import osmnx as ox
                    graph = ox.graph_from_place(f"{city}, Portugal", network_type="drive")
                    ox.plot_graph(graph, ax=ax, show=False, close=False, node_size=0,
                                  edge_color="#B9C2CC", edge_linewidth=0.6, bgcolor="white")
                except Exception as exc:  # noqa: BLE001
                    print(f"  [WARN] Street basemap unavailable for {city} ({exc}); scatter fallback")
                    ax.set_facecolor("white")
                    ax.invert_yaxis()
                    ax.set_xlabel("Longitude", fontsize=FS(11))
                    ax.set_ylabel("Latitude", fontsize=FS(11))
            else:
                ax.set_facecolor("white")
                ax.invert_yaxis()
                ax.set_xlabel("Longitude", fontsize=FS(11))
                ax.set_ylabel("Latitude", fontsize=FS(11))
            ax.scatter(coords["lon"], coords["lat"], s=22, color="#C0392B",
                       alpha=0.85, edgecolor="white", linewidth=0.4, zorder=5)
            ax.set_title(f"{city} — {n_bins} Plastic Bins", fontsize=FS(14), fontweight="bold")
            fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"  Saved: {out_name}")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] Could not render selected-bin map for {city} N={n_bins}: {exc}")
        # Fallback: copy the all-bin map
        fallback_path = out_dir / fallback_name
        if fallback_path.exists():
            _shutil.copy2(fallback_path, out_path)
            print(f"  [WARN] No selected-bin coords for {city} N={n_bins}; copied {fallback_name} → {out_name}")
        else:
            print(f"  [WARN] No selected-bin coords and no fallback map for {city} N={n_bins}; skipped")


# ── Horizon comparison figures ─────────────────────────────────────────────────


def gen_horizon_comparison(horizons: list[dict], ctx: dict, out_dir: Path) -> None:
    """Cross-horizon comparison charts (one bar colour per horizon)."""
    scenarios, strategies = ctx["scenarios"], ctx["strategies"]
    configs = [(s, strat) for s in scenarios for strat in strategies]
    col_labels = [f"{region_label(s['city'], s['N'])}\n{s['dist'][:3]}\n{strat}" for s, strat in configs]
    hcolors = META["horizon_colors"]

    def _means(dfm: pd.DataFrame, metric: str) -> np.ndarray:
        out = []
        for s, strat in configs:
            sub = _scen_sub(dfm[dfm.strategy == strat], s)[metric] # pyrefly: ignore [bad-argument-type]
            out.append(sub.mean() if len(sub) else np.nan)
        return np.array(out)

    for metric, fname, ylabel in [
        ("overflows", "horizon_overflow_comparison.png", "Mean overflows"),
        ("kgkm", "horizon_kgkm_comparison.png", "Mean kg/km"),
    ]:
        fig, ax = plt.subplots(figsize=(max(18, 0.9 * len(configs)), 8))
        pass
        x = np.arange(len(configs), dtype=float)
        n_h = len(horizons)
        w = 0.8 / n_h
        for hi, h in enumerate(horizons):
            vals = _means(h["dfm"], metric)
            ax.bar(
                x + (hi - (n_h - 1) / 2) * w,
                vals,
                w * 0.9,
                color=hcolors[hi % len(hcolors)],
                alpha=0.85,
                label=f"{h['days']} days",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(col_labels, fontsize=FS(6), rotation=45, ha="right")
        ax.set_ylabel(ylabel, fontsize=FS(10))
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(fontsize=FS(10))
        plt.tight_layout()
        savefig(fig, out_dir / fname)

    # relative delta between the shortest and longest horizons
    first, last = horizons[0], horizons[-1]
    m_a = _means(first["dfm"], "overflows")
    m_b = _means(last["dfm"], "overflows")
    delta = np.where(m_a > 0, (m_b - m_a) / m_a * 100, np.nan)
    fig, ax = plt.subplots(figsize=(max(18, 0.9 * len(configs)), 8))
    pass
    x = np.arange(len(configs))
    valid = ~np.isnan(delta)
    colors = ["#e05c5c" if v > 0 else "#5cb85c" for v in delta[valid]]
    ax.bar(x[valid], delta[valid], color=colors, alpha=0.85)
    ax.axhline(0, color=ctx["theme"]["fg"], linewidth=0.8, linestyle="--")
    ax.set_xticks(x[valid])
    ax.set_xticklabels(np.array(col_labels)[valid], fontsize=FS(6), rotation=45, ha="right")
    ax.set_ylabel("Δ overflows (%)", fontsize=FS(10))
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    savefig(fig, out_dir / "horizon_overflow_delta.png")

    # constructor rankings across horizons
    all_cons = sorted(set(itertools.chain.from_iterable(h["dfm"]["constructor"].unique() for h in horizons)))
    metrics_rank = ["overflows", "kgkm", "km", "profit"]
    rank_asc = [True, False, True, False]

    def _avg_ranks(dfm: pd.DataFrame) -> dict[str, float]:
        ranks: dict[str, list] = {c: [] for c in all_cons}
        for metric, asc in zip(metrics_rank, rank_asc, strict=True):
            for _, grp in dfm.groupby(["city", "N", "dist", "strategy", "improver"]):
                gi = grp.set_index("constructor")
                r = gi[metric].rank(ascending=asc, method="average")
                for c in all_cons:
                    if c in r.index:
                        ranks[c].append(r[c])
        return {c: float(np.mean(v)) if v else np.nan for c, v in ranks.items()}

    fig, ax = plt.subplots(figsize=(max(12, 1.8 * len(all_cons)), 8))
    pass
    x = np.arange(len(all_cons), dtype=float)
    n_h = len(horizons)
    w = 0.8 / n_h
    for hi, h in enumerate(horizons):
        r = _avg_ranks(h["dfm"])
        ax.bar(
            x + (hi - (n_h - 1) / 2) * w,
            [r.get(c, np.nan) for c in all_cons],
            w * 0.9,
            color=hcolors[hi % len(hcolors)],
            alpha=0.85,
            label=f"{h['days']} days",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(all_cons, rotation=15, ha="right", fontsize=FS(10))
    ax.set_ylabel("Average rank (lower = better)", fontsize=FS(11))
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=FS(10))
    plt.tight_layout()
    savefig(fig, out_dir / "horizon_constructor_ranking.png")


# ── Table builders ─────────────────────────────────────────────────────────────


def build_pareto_front_table(df: pd.DataFrame, ctx: dict) -> str:
    """One row per (selection variant, constructor, improver) on any scenario's Pareto front."""
    pareto_rows = []
    for s in ctx["scenarios"]:
        sub = _scen_sub(df, s).reset_index(drop=True)
        if sub.empty:
            continue
        for idx in pareto_indices(sub["overflows"].values, sub["kgkm"].values):
            row = sub.iloc[idx].copy()
            row["_scenario"] = scenario_label(s)
            pareto_rows.append(row)
    if not pareto_rows:
        return "_No Pareto-front data available._"
    pf = pd.DataFrame(pareto_rows)
    table_rows = []
    for (sel, con, imp), grp in pf.groupby(["variant", "constructor", "improver"]):
        scen = sorted(grp["_scenario"].unique())
        table_rows.append(
            {
                "sel": sel,
                "con": con,
                "imp": imp,
                "ov": grp["overflows"].mean(),
                "eff": grp["kgkm"].mean(),
                "scenarios": ", ".join(scen),
                "n": len(scen),
            }
        )
    table_rows.sort(key=lambda r: (-r["n"], r["sel"], r["con"], r["imp"]))
    lines = [
        "| Selection | Constructor | Improver | Overflows | kg/km | Pareto-Front Scenarios |",
        "|-----------|-------------|----------|----------:|------:|------------------------|",
    ]
    for r in table_rows:
        lines.append(f"| {r['sel']} | {r['con']} | {r['imp']} | {r['ov']:.1f} | {r['eff']:.3f} | {r['scenarios']} |")
    return "\n".join(lines)


def build_kpi_table(dfm: pd.DataFrame, ctx: dict, metric: str, fmt: str) -> str:
    """Combined min/max/mean table with one column group per improver."""
    improvers = ctx["improvers"]
    header = "| Config |" + "".join(f" {i} Min | {i} Max | {i} Mean |" for i in improvers)
    sep = "|--------|" + "-----|-----|------|" * len(improvers)
    lines = [header, sep]
    for s in ctx["scenarios"]:
        for strat in ctx["strategies"]:
            cells = []
            any_data = False
            for imp in improvers:
                sub = _scen_sub(dfm[(dfm.improver == imp) & (dfm.strategy == strat)], s)[metric] # pyrefly: ignore [bad-argument-type]
                if len(sub):
                    any_data = True
                    cells.append(f" {sub.min():{fmt}} | {sub.max():{fmt}} | {sub.mean():{fmt}} |")
                else:
                    cells.append(" — | — | — |")
            if any_data:
                lines.append(f"| {scenario_label(s)} / {strat} |" + "".join(cells))
    return "\n".join(lines)


def build_strategy_best(dfm: pd.DataFrame, ctx: dict) -> str:
    """Best constructor per strategy and scenario (improvers pooled)."""
    lines = []
    for strat in ctx["strategies"]:
        lines.append(f"#### {strat} ({META['strategy_names'].get(strat, strat)})")
        lines.append("")
        for s in ctx["scenarios"]:
            sub = _scen_sub(dfm[dfm.strategy == strat], s) # pyrefly: ignore [bad-argument-type]
            if sub.empty:
                continue
            agg = sub.groupby("constructor")[["overflows", "kgkm"]].mean()
            best_ov = agg["overflows"].idxmin()
            best_eff = agg["kgkm"].idxmax()
            lines.append(
                f"**{scenario_label(s)}:** best overflow: **{best_ov}** ({agg['overflows'].min():.1f}); "
                f"best efficiency: **{best_eff}** ({agg['kgkm'].max():.3f} kg/km)."
            )
            lines.append("")
        lines.append(PLACEHOLDER)
        lines.append("")
    return "\n".join(lines)


def _group_spans(keys: list[tuple], level: int) -> list[tuple[int, int, str]]:
    """Consecutive runs of `keys` sharing the same prefix up to `level` → (start, end, label)."""
    spans = []
    start = 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or keys[i][: level + 1] != keys[start][: level + 1]:
            spans.append((start, i, keys[start][level]))
            start = i
    return spans


def _fmt_result_cell(sub: pd.DataFrame) -> str:
    """One matrix cell: mean±std overflows and kg/km, stacked on two lines."""
    if sub.empty:
        return "—"
    ov, kg = sub["overflows"], sub["kgkm"]
    ov_s = f"{ov.mean():.1f}" + (f"±{ov.std():.1f}" if len(ov) > 1 else "")
    kg_s = f"{kg.mean():.3f}" + (f"±{kg.std():.3f}" if len(kg) > 1 else "")
    return f"{ov_s} ov<br>{kg_s} kg/km"


def build_full_results_matrix(
    dfm: pd.DataFrame, ctx: dict, horizon_label: str | None = None
) -> tuple[list[tuple], list[tuple], dict[tuple, str]]:
    """
    Build the full hierarchical results matrix.

    Rows: (city, N, dist) — region x graph size x data distribution (region is
    the outermost grouping, so e.g. all Rio Maior graph sizes merge under one
    region header span).
    Columns: (strategy, constructor, improver), optionally prefixed with a
    horizon label so multiple horizons can be merged into one column
    hierarchy (see build_full_results_table_all_horizons).
    """
    row_keys = sorted({(s["city"], s["N"], s["dist"]) for s in ctx["scenarios"]})
    col_keys = [
        (strat, con, imp) if horizon_label is None else (horizon_label, strat, con, imp)
        for strat in ctx["strategies"]
        for con in ctx["constructors"]
        for imp in ctx["improvers"]
    ]
    cells: dict[tuple, str] = {}
    for city, N, dist in row_keys:
        s = {"city": city, "N": N, "dist": dist}
        for strat in ctx["strategies"]:
            for con in ctx["constructors"]:
                for imp in ctx["improvers"]:
                    sub = _scen_sub(dfm[(dfm.strategy == strat) & (dfm.constructor == con) & (dfm.improver == imp)], s) # pyrefly: ignore [bad-argument-type]
                    ckey = (strat, con, imp) if horizon_label is None else (horizon_label, strat, con, imp)
                    cells[((city, N, dist), ckey)] = _fmt_result_cell(sub)
    return row_keys, col_keys, cells


def render_full_results_table_md(row_keys: list[tuple], col_keys: list[tuple], cells: dict[tuple, str]) -> str:
    """Render the hierarchical results matrix as a GFM pipe table."""
    if not row_keys or not col_keys:
        return "_No data available._"
    n_row_levels = len(row_keys[0])
    row_headers = ["Region", "N", "Distribution"][:n_row_levels] or [f"Level {i + 1}" for i in range(n_row_levels)]
    col_labels = ["<br>".join(str(x) for x in ck) for ck in col_keys]
    lines = [
        "| " + " | ".join(row_headers) + " | " + " | ".join(col_labels) + " |",
        "|" + "---|" * (n_row_levels + len(col_keys)),
    ]
    prev: tuple = ()
    for rk in row_keys:
        row_cells = []
        for lvl in range(n_row_levels):
            row_cells.append("" if prev[: lvl + 1] == rk[: lvl + 1] else str(rk[lvl]))
        prev = rk
        data_cells = [cells.get((rk, ck), "—") for ck in col_keys]
        lines.append("| " + " | ".join(row_cells) + " | " + " | ".join(data_cells) + " |")
    return "\n".join(lines)


def build_full_results_table_for_horizon(dfm: pd.DataFrame, ctx: dict) -> str:
    """Full results table (all scenarios x policy configs) for a single horizon."""
    row_keys, col_keys, cells = build_full_results_matrix(dfm, ctx)
    return render_full_results_table_md(row_keys, col_keys, cells)


def build_full_results_table_all_horizons(horizons: list[dict]) -> str:
    """Full results table across all horizons — horizon prepended to the column hierarchy."""
    row_keys_all: list[tuple] = []
    col_keys_all: list[tuple] = []
    cells_all: dict[tuple, str] = {}
    for h in horizons:
        rk, ck, cells = build_full_results_matrix(h["dfm"], h["ctx"], horizon_label=f"{h['days']}d")
        for r in rk:
            if r not in row_keys_all:
                row_keys_all.append(r)
        col_keys_all += ck
        cells_all.update(cells)
    row_keys_all.sort()
    return render_full_results_table_md(row_keys_all, col_keys_all, cells_all)


# ── Orchestration ──────────────────────────────────────────────────────────────


def gen_horizon_figures(
    df: pd.DataFrame, dfm: pd.DataFrame, ctx: dict, figures_dir: Path, private_dir: Path,
    scenario_heatmap_labels: str = "both",
) -> dict:
    figures_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    charts = ctx["charts"]
    interactive: dict[str, Path] = {}

    def _on(name: str) -> bool:
        return charts.get(name, {}).get("enabled", True)

    if _on("pareto_scatter"):
        gen_pareto_scatter(df, ctx, figures_dir)
    if _on("overflow_bar"):
        gen_kpi_bar(
            dfm, "overflows", "Overflow Count", "overflow_by_config", ctx, figures_dir, charts.get("overflow_bar", {})
        )
    if _on("kgkm_bar"):
        gen_kpi_bar(dfm, "kgkm", f"{KGKM_LABEL} Efficiency", "kgkm_by_config", ctx, figures_dir, charts.get("kgkm_bar", {}))
    if _on("kpi_combined"):
        gen_kpi_combined(dfm, ctx, figures_dir)
    if _on("km_violin"):
        gen_km_violin(df, ctx, figures_dir)
    if _on("policy_scenario_heatmap"):
        gen_policy_scenario_heatmap(df, ctx, figures_dir)
    if _on("scenario_constructor_heatmap"):
        if scenario_heatmap_labels in ("both", "show"):
            gen_scenario_constructor_heatmap(dfm, ctx, figures_dir, shared_axis_labels=True)
        if scenario_heatmap_labels in ("both", "hide"):
            gen_scenario_constructor_heatmap(dfm, ctx, figures_dir, shared_axis_labels=False)
        empirical_scenarios = [s for s in ctx["scenarios"] if s["dist"].lower() == "empirical"]
        if empirical_scenarios:
            empirical_ctx = {**ctx, "scenarios": empirical_scenarios}
            gen_scenario_constructor_heatmap(
                dfm, empirical_ctx, figures_dir, shared_axis_labels=True,
                out_fname="scenario_constructor_heatmap_empirical.png",
            )
    if _on("strategy_bubble"):
        gen_strategy_bubble(dfm, ctx, figures_dir)
    if _on("improver_bubble"):
        gen_improver_bubble(dfm, ctx, figures_dir)
    if _on("constructor_ranking"):
        gen_constructor_ranking(dfm, ctx, figures_dir)
    if _on("radar"):
        gen_radar(dfm, ctx, figures_dir)
    if _on("radar_combined"):
        gen_radar_combined(dfm, ctx, figures_dir)
    if _on("improver_delta") and len(ctx["improvers"]) > 1:
        gen_improver_delta(dfm, ctx, figures_dir)
    if _on("interactive"):
        interactive = gen_interactive_html(df, dfm, ctx, private_dir)
    return interactive


def build_context(df: pd.DataFrame, config: dict, theme: dict, n_days: int) -> dict:
    scenarios = config.get("scenarios") or detect_scenarios(df)
    scenarios = [{"city": s["city"], "N": int(s["N"]), "dist": s["dist"]} for s in scenarios]
    scenarios = [s for s in scenarios if len(_scen_sub(df, s))]
    regions = []
    for s in scenarios:
        r = (s["city"], s["N"])
        if r not in regions:
            regions.append(r)
    pol = config.get("policies") or {}
    return {
        "scenarios": scenarios,
        "regions": regions,
        "dists": sorted({s["dist"] for s in scenarios}),
        "strategies": pol.get("strategies") or sorted(df["strategy"].unique()),
        "constructors": pol.get("constructors") or sorted(df["constructor"].unique()),
        "improvers": pol.get("improvers") or sorted(df["improver"].unique(), reverse=True),
        "charts": config.get("charts", {}),
        "theme": theme,
        "pareto_points": config.get("pareto_points"),
        "n_days": n_days,
    }


def _to_rel(p: Path) -> str:
    """Path relative to out_md's directory (docs/private/markdown/), for
    links inside the generated markdown. out_md sits one level deeper than
    figures_dir/private_dir (docs/private/figures/, docs/private/html/), so
    stripping "docs/private/" alone would produce a broken sibling-style
    link -- the extra "../" accounts for that one level. This assumes the
    fixed docs/private/markdown/ layout from json/simulation_analysis_config.json;
    see gen_dataset_analysis.py's os.path.relpath(..., start=out_md.parent)
    for the general version if out_md's location ever becomes configurable.
    """
    s = str(p)
    return "../" + s.replace("docs/private/", "", 1) if s.startswith("docs/private/") else s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--config", default=None, help="Analysis config JSON (defaults to json/simulation_analysis_config.json)"
    )
    p.add_argument("--theme", choices=["dark", "light"], default=None, help="Chart theme override")
    p.add_argument(
        "--fontsize",
        type=float,
        default=None,
        help="Base fontsize for all matplotlib chart figures (titles/labels/ticks/legends scale together "
        "from this). Overrides the 'base_fontsize' key in the config and the active mplstyle's font.size.",
    )
    p.add_argument("--fontsize-axis", type=float, default=None, help="Override axis tick/label fontsize (FS_AXIS)")
    p.add_argument("--fontsize-labels", type=float, default=None, help="Override annotation label fontsize (FS_LABEL)")
    p.add_argument("--fontsize-legend", type=float, default=None, help="Override legend fontsize (FS_LEGEND)")
    p.add_argument("--fontsize-title", type=float, default=None, help="Override title fontsize (FS_TITLE)")
    p.add_argument(
        "--pareto-points",
        choices=["all", "front"],
        default=None,
        help="Plot all points or only the Pareto-front points in the static Pareto charts",
    )
    p.add_argument(
        "--horizon",
        action="append",
        default=None,
        metavar="DAYS=CSV",
        help="Horizon spec (repeatable), e.g. --horizon 30=path/to.csv; overrides config horizons",
    )
    p.add_argument(
        "--scenarios",
        default=None,
        help="Scenario filter: 'City:N:Dist;City:N:Dist;...' (e.g. 'Rio Maior:100:Gamma-3')",
    )
    p.add_argument("--strategies", default=None, help="Comma-separated selection strategies to include")
    p.add_argument("--constructors", default=None, help="Comma-separated route constructors to include")
    p.add_argument("--improvers", default=None, help="Comma-separated route improvers to include")
    p.add_argument("--acceptance", default=None, help="Comma-separated acceptance criteria to include")
    p.add_argument("--out-md", default=None)
    p.add_argument("--figures-dir", default=None)
    p.add_argument("--private-dir", default=None)
    p.add_argument("--force", action="store_true", help="Overwrite existing markdown")
    p.add_argument("--figures-only", action="store_true", help="Generate figures but do not write markdown")
    p.add_argument(
        "--map-mode",
        default="street",
        choices=["street", "scatter"],
        help="Bin-location maps: 'street' overlays bins on the real OSM road network (needs network "
        "access; default), 'scatter' is a plain lat/lon scatter with no basemap.",
    )
    p.add_argument(
        "--scenario-heatmap-labels",
        default="both",
        choices=["both", "show", "hide"],
        help="Per-scenario constructor heatmap (gen_scenario_constructor_heatmap): 'show' writes only the "
        "shared-axis-labels version, 'hide' only the no-labels version, 'both' (default) writes both.",
    )
    # merged gen_simulation_csv mode
    p.add_argument("--parse-output", action="store_true", help="Parse a raw output tree into a summary CSV and exit")
    p.add_argument("--output-dir", default="assets/output/90days", help="Root of the raw simulation output tree")
    p.add_argument(
        "--out-csv",
        default="docs/private/global/simulation/simulation_summary_90d.csv",
        help="Destination CSV path (with --parse-output)",
    )
    return p.parse_args()


def run_parse_output(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    out_csv = Path(args.out_csv)
    if not output_dir.is_dir():
        raise SystemExit(f"Output dir not found: {output_dir}")
    print(f"Parsing: {output_dir}")
    df = parse_output_dir(output_dir)
    if df.empty:
        raise SystemExit("No log files found — check the output directory structure.")
    print(f"  Rows: {len(df)}")
    for col in ["city", "dist", "strategy", "improver", "constructor"]:
        print(f"  {col.capitalize()}s: {sorted(df[col].unique())}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Written: {out_csv}")


def main() -> None:  # noqa: C901
    args = parse_args()
    if args.parse_output:
        run_parse_output(args)
        return

    config = load_json("simulation_analysis_config.json")
    if args.config:
        config.update(json.loads(Path(args.config).read_text(encoding="utf-8")))
    if args.theme:
        config["theme"] = args.theme
    if args.pareto_points:
        config["pareto_points"] = args.pareto_points
    if args.horizon:
        config["horizons"] = []
        for spec in args.horizon:
            days, _, csv = spec.partition("=")
            config["horizons"].append({"days": int(days), "csv": csv})
    if args.scenarios:
        config["scenarios"] = []
        for part in args.scenarios.split(";"):
            city, n, dist = part.split(":")
            config["scenarios"].append({"city": city.strip(), "N": int(n), "dist": dist.strip()})
    pol = config.setdefault("policies", {})
    for key, val in [
        ("strategies", args.strategies),
        ("constructors", args.constructors),
        ("improvers", args.improvers),
        ("acceptance", args.acceptance),
    ]:
        if val:
            pol[key] = [v.strip() for v in val.split(",")]

    theme = load_theme(config.get("theme", "dark"))
    plt.style.use(theme["mplstyle_path"])

    # Base chart fontsize: --fontsize > config "base_fontsize" > the active mplstyle's
    # own font.size (if it sets one) > the built-in reference default.
    base_fontsize = args.fontsize or config.get("base_fontsize") or matplotlib.rcParams.get("font.size")
    set_chart_fontsize(
        float(base_fontsize) if base_fontsize else _BASE_FONTSIZE_REF,
        axis_fontsize=args.fontsize_axis,
        label_fontsize=args.fontsize_labels,
        legend_fontsize=args.fontsize_legend,
        title_fontsize=args.fontsize_title,
    )

    out_md = Path(args.out_md or config["out_md"])
    figures_base = Path(args.figures_dir or config["figures_dir"])
    private_base = Path(args.private_dir or config["private_dir"])

    # ── Load all horizons (sorted smallest → largest) ───────────────────────────
    horizon_specs = sorted(config["horizons"], key=lambda h: h["days"])
    horizons: list[dict] = []
    for spec in horizon_specs:
        days = int(spec["days"])
        csv_path = Path(spec["csv"]) if spec.get("csv") else None
        if (csv_path is None or not csv_path.exists()) and spec.get("output_dir"):
            print(f"Parsing raw output for {days}d: {spec['output_dir']}")
            df = parse_output_dir(Path(spec["output_dir"]))
            if csv_path:
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(csv_path, index=False)
                print(f"  Written: {csv_path}")
            for col in ("cf", "sl_var", "acceptance"):
                df[col] = df[col].fillna("")
            df["variant"] = df.apply(variant_label, axis=1)
        elif csv_path and csv_path.exists():
            print(f"Reading {days}d CSV: {csv_path}")
            df = load_horizon_csv(csv_path)
        else:
            print(f"  [WARN] No data source for horizon {days}d — skipping")
            continue
        df = filter_data(df, config)
        if df.empty:
            print(f"  [WARN] Horizon {days}d has no rows after filtering — skipping")
            continue
        horizons.append({"days": days, "df": df, "dfm": aggregate(df), "csv": csv_path})
    if not horizons:
        raise SystemExit("No horizon data available.")

    multi = len(horizons) > 1
    suffix = lambda d: f"{d}d"  # noqa: E731

    # ── Figures ─────────────────────────────────────────────────────────────────
    for h in horizons:
        ctx = build_context(h["df"], config, theme, h["days"])
        h["ctx"] = ctx
        fig_dir = figures_base / suffix(h["days"]) if multi else figures_base
        priv_dir = private_base / suffix(h["days"]) if multi else private_base
        print(f"\nGenerating {h['days']}d figures → {fig_dir}")
        h["interactive"] = gen_horizon_figures(
            h["df"], h["dfm"], ctx, fig_dir, priv_dir, scenario_heatmap_labels=args.scenario_heatmap_labels
        )
        h["figures_dir"], h["private_dir"] = fig_dir, priv_dir
        gen_bin_location_maps(fig_dir, mode=args.map_mode)
        gen_selected_bin_maps(fig_dir, mode=args.map_mode)

    cmp_dir = None
    if multi:
        cmp_dir = figures_base / "compare"
        cmp_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nGenerating horizon comparison figures → {cmp_dir}")
        # comparison uses the scenario/strategy space of the largest horizon's ctx
        gen_horizon_comparison(horizons, horizons[-1]["ctx"], cmp_dir)

    if args.figures_only:
        print("--figures-only: skipping markdown generation")
        return
    if out_md.exists() and not args.force:
        print(f"\n{out_md} already exists. Use --force to regenerate. Skipping markdown write.")
        return

    # ── Markdown context ────────────────────────────────────────────────────────
    ctx0 = horizons[0]["ctx"]
    all_scen = ctx0["scenarios"]
    strategies, improvers, constructors = ctx0["strategies"], ctx0["improvers"], ctx0["constructors"]
    acceptance = sorted({a for h in horizons for a in h["df"]["acceptance"].unique() if a})

    toc_items = ["1. [Experimental Setup](#1-experimental-setup)"]
    h_ctxs = []
    for i, h in enumerate(horizons, start=2):
        ctx = h["ctx"]
        anchor = f"{i}-{h['days']}-day-horizon-results"
        toc_items.append(f"{i}. [{h['days']}-Day Horizon Results](#{anchor})")
        h_ctxs.append(
            {
                "days": h["days"],
                "section_num": i,
                "n_logs": len(h["df"]),
                "constructors": ctx["constructors"],
                "figures_rel": _to_rel(h["figures_dir"]),
                "private_rel": _to_rel(h["private_dir"]),
                "pareto_table": build_pareto_front_table(h["df"], ctx),
                "overflow_table": build_kpi_table(h["dfm"], ctx, "overflows", ".1f"),
                "efficiency_table": build_kpi_table(h["dfm"], ctx, "kgkm", ".2f"),
                "km_table": build_kpi_table(h["dfm"], ctx, "km", ".0f"),
                "strategy_best": build_strategy_best(h["dfm"], ctx),
                "full_results_table": build_full_results_table_for_horizon(h["dfm"], ctx),
                "has_pareto_log": (h["figures_dir"] / "pareto_scatter_log.png").exists(),
                "has_overflow_log": (h["figures_dir"] / "overflow_by_config_log.png").exists(),
                "has_bubble_log": (h["figures_dir"] / "strategy_bubble_log.png").exists(),
                "has_improver_delta": (h["figures_dir"] / "improver_delta.png").exists(),
                "interactive": {k: True for k in h["interactive"]},
            }
        )
    comparison = None
    if multi:
        sec = len(horizons) + 2
        label = " vs ".join(f"{h['days']}d" for h in horizons)
        toc_items.append(
            f"{sec}. [Horizon Comparison ({label})](#{sec}-horizon-comparison-{label.replace(' ', '-').lower()})"
        )
        comparison = {
            "section_num": sec,
            "label": label,
            "figures_rel": _to_rel(cmp_dir),  # pyrefly: ignore [bad-argument-type]
            "first_days": horizons[0]["days"],
            "last_days": horizons[-1]["days"],
            "full_results_table": build_full_results_table_all_horizons(horizons),
        }

    scope = (
        f"{' and '.join(str(h['days']) + '-day' for h in horizons)} simulation runs across "
        f"{len(ctx0['regions'])} region/network configurations × {len(ctx0['dists'])} distributions × "
        f"{len(strategies)} selection strategies × {len(improvers)} route improvers × "
        f"{len(constructors)} route constructors"
    )
    md = render_template(
        "simulation_analysis.md.j2",
        scope=scope,
        horizons=h_ctxs,
        horizon_days=[h["days"] for h in horizons],
        scenario_str=", ".join(scenario_label(s) for s in all_scen),
        scenario_regions=", ".join(f"{region_label(c, n)} (N={n})" for c, n in ctx0["regions"]),
        dists=ctx0["dists"],
        strategies=strategies,
        improvers=improvers,
        constructors=constructors,
        acceptance=acceptance,
        comparison=comparison,
        toc="\n".join(toc_items),
        placeholder=PLACEHOLDER,
        figures_base_rel=_to_rel(figures_base),
        csv_list=", ".join(f"`{h['csv']}`" for h in horizons if h["csv"]),
    )
    md = finalize_markdown(md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    print(f"\nWritten: {out_md} ({len(md)} chars)")


if __name__ == "__main__":
    main()
