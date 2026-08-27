"""
Generate the dataset analysis markdown report and all associated figures.

Reads NPZ and TD statistics CSVs, auto-detects available cities and
distributions, loads the raw NPZ waste matrices (when available) to compute
extended statistics — median, variance, quartiles, IQR, minimum, outlier
fences and mode — and richer distribution-shape figures (violin plots, box
plots, histograms with KDE). Line charts that show the evolution across the
Rio Maior network sizes (N=20…170) also display the Figueira da Foz N=350
reference value.

Non-Python content lives in sibling directories:
  jinja/dataset_analysis.md.j2      markdown template
  json/dataset_analysis_config.json labels, colours and default paths
  json/themes.json                  dark/light theme definitions
  style/{dark,light}.mplstyle       matplotlib style sheets

Idempotent: will not overwrite an existing markdown unless --force is passed.

Usage
-----
    uv run python logic/gen/gen_dataset_analysis.py --force
    uv run python logic/gen/gen_dataset_analysis.py \\
        --theme light \\
        --npz-csv docs/private/global/datasets/npz_stats.csv \\
        --td-csv docs/private/global/datasets/td_stats.csv \\
        --npz-dir data/wsr_simulator/datasets \\
        --out-md docs/private/dataset_analysis.md \\
        --figures-dir docs/private/figures/datasets \\
        --private-dir docs/private/html/datasets
"""

from __future__ import annotations

import argparse
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
    load_json,
    load_theme,
    render_template,
    savefig,
)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

CFG = load_json("dataset_analysis_config.json")
CITY_LABELS = CFG["city_labels"]
DIST_LABELS = CFG["dist_labels"]
CITY_COLORS = CFG["city_colors"]
DIST_COLORS = CFG["dist_colors"]

REF_CITY = "figueiradafoz"  # single-N reference city shown as scatter points
LINE_CITY = "riomaior"  # multi-N city shown as evolution lines


# ── Raw NPZ loading & extended statistics ──────────────────────────────────────


def load_raw_waste(npz_dir: Path, npz: pd.DataFrame, horizon: int = 30) -> dict[tuple, np.ndarray]:
    """Return {(city, N, dist): flat waste array} for every entry with a raw NPZ file."""
    raw: dict[tuple, np.ndarray] = {}
    if not npz_dir.is_dir():
        print(f"  [WARN] NPZ dir not found: {npz_dir} — raw-data figures will be skipped")
        return raw
    sub = npz[npz["horizon"] == horizon]
    for _, row in sub.iterrows():
        matches = sorted(npz_dir.glob(f"{row.city}{row.N}_{row.dist}_wsr{horizon}_*.npz"))
        if not matches:
            continue
        try:
            data = np.load(matches[0], allow_pickle=True)
            raw[(row.city, int(row.N), row.dist)] = np.asarray(data["waste"]).ravel()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  [WARN] Failed to read {matches[0].name}: {exc}")
    return raw


def extended_stats(values: np.ndarray) -> dict[str, float]:
    """Median, variance, quartiles, IQR, min, outlier fences and (binned) mode."""
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    hist, edges = np.histogram(values, bins=50)
    mode = float((edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2)
    return {
        "median": float(med),
        "variance": float(np.var(values)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "min": float(np.min(values)),
        "lower_fence": float(q1 - 1.5 * iqr),
        "upper_fence": float(q3 + 1.5 * iqr),
        "mode": mode,
    }


def build_extended_df(raw: dict[tuple, np.ndarray]) -> pd.DataFrame:
    rows = []
    for (city, N, dist), values in sorted(raw.items()):
        rows.append({"city": city, "N": N, "dist": dist, **extended_stats(values)})
    return pd.DataFrame(rows)


# ── Figure helpers ─────────────────────────────────────────────────────────────


def _plot_line_plus_ref(
    ax,
    sub_line: pd.DataFrame,
    sub_ref: pd.DataFrame,
    metric: str,
    dist: str,
    label: str | None = None,
    ls: str = "-",
    alpha: float = 1.0,
) -> None:
    """Rio Maior evolution line plus the Figueira da Foz N=350 reference diamond."""
    color = DIST_COLORS.get(dist, "gray")
    if len(sub_line):
        ax.plot(
            sub_line["N"].values,
            sub_line[metric].values,
            ls,
            marker="o",
            color=color,
            linewidth=2,
            markersize=8,
            alpha=alpha,
            label=label or DIST_LABELS.get(dist, dist),
        )
    if len(sub_ref):
        ax.scatter(
            sub_ref["N"].values,
            sub_ref[metric].values,
            marker="D",
            s=110,
            color=color,
            edgecolors=CITY_COLORS.get(REF_CITY, "#e08030"),
            linewidths=1.8,
            alpha=alpha,
            zorder=4,
            label=(f"{label or DIST_LABELS.get(dist, dist)} (FFZ-350)"),
        )


# ── Figure generators ──────────────────────────────────────────────────────────


def gen_npz_stats_bar(npz: pd.DataFrame, ext: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart of mean / median / std / max per city and distribution (30-day)."""
    sub30 = npz[npz["horizon"] == 30].merge(ext, on=["city", "N", "dist"], how="left")
    cities = sorted(sub30["city"].unique())
    dists = sorted(sub30["dist"].unique())
    metrics = [
        ("mean_kg", "Mean Waste (kg/bin/day)"),
        ("median", "Median Waste (kg/bin/day)"),
        ("std_kg", "Std Waste (kg/bin/day)"),
        ("max_kg", "Max Waste (kg)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("NPZ Dataset Statistics by City and Distribution", fontsize=14, fontweight="bold")
    for ax_i, (metric, ylabel) in enumerate(metrics):
        ax = axes[ax_i // 2][ax_i % 2]
        for di, dist in enumerate(dists):
            sub_d = sub30[sub30["dist"] == dist]
            for ci, city in enumerate(cities):
                vals = sub_d[sub_d["city"] == city][metric].dropna().values
                if len(vals):
                    off = (di - (len(dists) - 1) / 2) * 0.4
                    ax.bar(
                        ci + off,
                        np.mean(vals),
                        0.35,
                        color=DIST_COLORS.get(dist, "gray"),
                        alpha=0.85,
                        label=DIST_LABELS.get(dist, dist) if ci == 0 else "",
                    )
        ax.set_xticks(range(len(cities)))
        ax.set_xticklabels([CITY_LABELS.get(c, c) for c in cities], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(metric, fontsize=11)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
        if ax_i == 0:
            ax.legend(fontsize=9)
    plt.tight_layout()
    savefig(fig, out_dir / "npz_stats_bar.png")


def gen_npz_size_scaling(npz: pd.DataFrame, out_dir: Path) -> None:
    sub30 = npz[npz["horizon"] == 30]
    rm = sub30[sub30["city"] == LINE_CITY].sort_values("N")
    ref = sub30[sub30["city"] == REF_CITY]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "NPZ Statistics vs Network Size (lines: Rio Maior · diamonds: FFZ-350)", fontsize=14, fontweight="bold"
    )
    for ax, (metric, ylabel) in zip(
        axes,
        [
            ("mean_kg", "Mean Waste (kg)"),
            ("std_kg", "Std Waste (kg)"),
            ("skewness", "Skewness"),
        ],
        strict=True,
    ):
        for dist in sorted(sub30["dist"].unique()):
            _plot_line_plus_ref(ax, rm[rm["dist"] == dist], ref[ref["dist"] == dist], metric, dist)
        ax.set_xlabel("Network size N", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{metric} vs N", fontsize=11)
        ax.legend(fontsize=8)
        ax.yaxis.grid(True, alpha=0.4)
        ax.xaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
    plt.tight_layout()
    savefig(fig, out_dir / "npz_size_scaling.png")


def gen_npz_horizon_comparison(npz: pd.DataFrame, out_dir: Path) -> None:
    horizons = sorted(npz["horizon"].unique())
    if len(horizons) < 2:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"{' vs '.join(str(h) + '-day' for h in horizons)} Horizon Comparison (lines: Rio Maior · diamonds: FFZ-350)",
        fontsize=14,
        fontweight="bold",
    )
    for ax, (metric, ylabel) in zip(
        axes,
        [
            ("mean_kg", "Mean Waste (kg)"),
            ("std_kg", "Std Waste (kg)"),
            ("skewness", "Skewness"),
        ],
        strict=True,
    ):
        for dist in sorted(npz["dist"].unique()):
            for horizon, ls in zip(horizons, ["-", "--", ":"], strict=False):
                sub_h = npz[(npz["dist"] == dist) & (npz["horizon"] == horizon)]
                rm = sub_h[sub_h["city"] == LINE_CITY].sort_values("N")
                ref = sub_h[sub_h["city"] == REF_CITY]
                _plot_line_plus_ref(
                    ax,
                    rm,
                    ref,
                    metric,
                    dist,
                    label=f"{DIST_LABELS.get(dist, dist)} {horizon}d",
                    ls=ls,
                    alpha=1.0 if horizon == horizons[0] else 0.7,
                )
        ax.set_xlabel("Network size N", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(metric, fontsize=11)
        ax.legend(fontsize=7)
        ax.yaxis.grid(True, alpha=0.4)
        ax.xaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
    plt.tight_layout()
    savefig(fig, out_dir / "npz_horizon_comparison.png")


def gen_npz_city_comparison(npz: pd.DataFrame, ext: pd.DataFrame, out_dir: Path) -> None:
    sub30 = npz[npz["horizon"] == 30].merge(ext, on=["city", "N", "dist"], how="left")
    cities = sorted(sub30["city"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("City Comparison Overview — NPZ Datasets", fontsize=14, fontweight="bold")
    for ax_i, (metric, ylabel) in enumerate(
        [
            ("mean_kg", "Mean Waste (kg)"),
            ("median", "Median Waste (kg)"),
            ("std_kg", "Std (kg)"),
            ("skewness", "Skewness"),
        ]
    ):
        ax = axes[ax_i // 2][ax_i % 2]
        for dist in sorted(sub30["dist"].unique()):
            sub_d = sub30[sub30["dist"] == dist]
            for city in cities:
                sub_cd = sub_d[sub_d["city"] == city].sort_values("N")
                for _, row in sub_cd.iterrows():
                    if pd.isna(row.get(metric)):
                        continue
                    marker = "o" if city == LINE_CITY else "D"
                    ec = "white" if dist == "gamma3" else "none"
                    ax.scatter(
                        row["N"],
                        row[metric],
                        c=CITY_COLORS.get(city, "#a0a0a0"),
                        marker=marker,
                        s=150,
                        alpha=0.85,
                        edgecolors=ec,
                        linewidths=1.5,
                    )
        ax.set_xlabel("Network size N", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(metric, fontsize=11)
        ax.yaxis.grid(True, alpha=0.4)
        ax.xaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
    city_patches = [mpatches.Patch(color=CITY_COLORS.get(c, "gray"), label=CITY_LABELS.get(c, c)) for c in cities]
    dist_patches = [
        plt.scatter([], [], c="gray", s=60, marker="o", edgecolors="none", label=DIST_LABELS.get("emp", "emp")),
        plt.scatter(
            [],
            [],
            c="gray",
            s=60,
            marker="o",
            edgecolors="white",
            linewidths=1.5,
            label=DIST_LABELS.get("gamma3", "gamma3"),
        ),
    ]
    fig.legend(handles=city_patches + dist_patches, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout()
    savefig(fig, out_dir / "npz_city_comparison.png")


def gen_npz_td_alignment(npz: pd.DataFrame, td: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title("Training (TD) vs Simulator (NPZ) Mean Waste Alignment", fontsize=13, fontweight="bold")
    sub30 = npz[npz["horizon"] == 30]
    rm_npz = sub30[sub30["city"] == LINE_CITY]
    ref_npz = sub30[sub30["city"] == REF_CITY]
    for dist in sorted(sub30["dist"].unique()):
        rm_sub = rm_npz[rm_npz["dist"] == dist].sort_values("N")
        ref_sub = ref_npz[ref_npz["dist"] == dist]
        td_sub = td[td["dist"] == dist].sort_values("N")
        color = DIST_COLORS.get(dist, "gray")
        dist_label = DIST_LABELS.get(dist, dist)
        ax.plot(
            rm_sub["N"].values,
            rm_sub["mean_kg"].values,
            "o-",
            color=color,
            linewidth=2,
            markersize=8,
            label=f"NPZ {dist_label} (RM)",
        )
        if len(ref_sub):
            ax.scatter(
                ref_sub["N"].values,
                ref_sub["mean_kg"].values,
                marker="D",
                s=110,
                color=color,
                edgecolors=CITY_COLORS.get(REF_CITY, "#e08030"),
                linewidths=1.8,
                zorder=4,
                label=f"NPZ {dist_label} (FFZ-350)",
            )
        if len(td_sub):
            ax.plot(
                td_sub["N"].values,
                td_sub["waste_mean"].values * 100,
                "s--",
                color=color,
                linewidth=1.5,
                markersize=6,
                alpha=0.7,
                label=f"TD {dist_label} (×100)",
            )
    ax.set_xlabel("Network size N", fontsize=11)
    ax.set_ylabel("Mean waste (kg/bin/day)", fontsize=11)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.4)
    ax.xaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    savefig(fig, out_dir / "npz_td_alignment.png")


def gen_td_stats(td: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Training Data (TD) Statistics Comparison", fontsize=14, fontweight="bold")
    for ax, (metric, ylabel) in zip(
        axes,
        [
            ("waste_mean", "Mean waste fraction"),
            ("waste_std", "Std waste fraction"),
            ("waste_skew", "Skewness"),
        ],
        strict=True,
    ):
        for dist in sorted(td["dist"].unique()):
            sub = td[td["dist"] == dist].sort_values("N")
            ax.plot(
                sub["N"].values,
                sub[metric].values,
                "o-",
                color=DIST_COLORS.get(dist, "gray"),
                linewidth=2,
                markersize=8,
                label=DIST_LABELS.get(dist, dist),
            )
        ax.set_xlabel("Network size N", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(metric, fontsize=11)
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, alpha=0.4)
        ax.xaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
    plt.tight_layout()
    savefig(fig, out_dir / "td_stats_comparison.png")

    ns = sorted(td["N"].unique())
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle("Training Data Waste Distributions", fontsize=14, fontweight="bold")
    for ax, metric in zip(axes2, ["waste_mean", "waste_std"], strict=True):
        for dist in sorted(td["dist"].unique()):
            sub = td[td["dist"] == dist].sort_values("N")
            offset = 0.2 if dist == "gamma3" else -0.2
            ax.bar(
                np.arange(len(sub)) + offset,
                sub[metric].values,
                0.35,
                color=DIST_COLORS.get(dist, "gray"),
                alpha=0.85,
                label=DIST_LABELS.get(dist, dist),
            )
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels([f"N={n}" for n in ns], fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(metric, fontsize=11)
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
    plt.tight_layout()
    savefig(fig2, out_dir / "td_waste_distributions.png")


def _raw_groups(raw: dict[tuple, np.ndarray]) -> tuple[list[str], dict[str, list]]:
    """Group raw arrays by distribution: returns (dists, {dist: [(label, values), ...]})."""
    dists = sorted({k[2] for k in raw})
    groups: dict[str, list] = {d: [] for d in dists}
    for (city, N, dist), values in sorted(raw.items(), key=lambda kv: kv[0][1]):
        short = "FFZ" if city == REF_CITY else "RM"
        groups[dist].append((f"{short}-{N}", values))
    return dists, groups


def gen_npz_violin(raw: dict[tuple, np.ndarray], theme: dict, out_dir: Path) -> None:
    dists, groups = _raw_groups(raw)
    fig, axes = plt.subplots(1, len(dists), figsize=(9 * len(dists), 7), squeeze=False)
    fig.suptitle("Raw Waste Distribution — Violin Plots (30-day horizon)", fontsize=14, fontweight="bold")
    for ax, dist in zip(axes[0], dists, strict=True):
        labels = [lbl for lbl, _ in groups[dist]]
        data = [vals for _, vals in groups[dist]]
        parts = ax.violinplot(data, positions=range(len(data)), showmedians=True, showextrema=True)
        for body in parts["bodies"]:
            body.set_facecolor(DIST_COLORS.get(dist, "gray"))
            body.set_alpha(0.7)
        for key in ["cmedians", "cmins", "cmaxes", "cbars"]:
            parts[key].set_color(theme["accent_line"] if key == "cmedians" else theme["guide_line"])
        for i, vals in enumerate(data):
            q1, q3 = np.percentile(vals, [25, 75])
            ax.scatter([i, i], [q1, q3], marker="_", s=250, color=theme["accent_line"], zorder=4)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Waste (kg/bin/day)", fontsize=10)
        ax.set_title(DIST_LABELS.get(dist, dist), fontsize=11)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
    plt.tight_layout()
    savefig(fig, out_dir / "npz_violin.png")


def gen_npz_box(raw: dict[tuple, np.ndarray], theme: dict, out_dir: Path) -> None:
    dists, groups = _raw_groups(raw)
    fig, axes = plt.subplots(1, len(dists), figsize=(9 * len(dists), 7), squeeze=False)
    fig.suptitle("Raw Waste Distribution — Box Plots (30-day horizon)", fontsize=14, fontweight="bold")
    for ax, dist in zip(axes[0], dists, strict=True):
        labels = [lbl for lbl, _ in groups[dist]]
        data = [vals for _, vals in groups[dist]]
        bp = ax.boxplot(
            data,
            positions=range(len(data)),
            widths=0.6,
            patch_artist=True,
            showfliers=True,
            whis=1.5,
            flierprops=dict(
                marker=".", markersize=3, alpha=0.35, markerfacecolor=theme["muted"], markeredgecolor="none"
            ),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(DIST_COLORS.get(dist, "gray"))
            patch.set_alpha(0.7)
        for med in bp["medians"]:
            med.set_color(theme["accent_line"])
        for el in bp["whiskers"] + bp["caps"]:
            el.set_color(theme["guide_line"])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Waste (kg/bin/day)", fontsize=10)
        ax.set_title(DIST_LABELS.get(dist, dist), fontsize=11)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
    plt.tight_layout()
    savefig(fig, out_dir / "npz_box.png")


def gen_npz_hist_kde(raw: dict[tuple, np.ndarray], theme: dict, out_dir: Path) -> None:
    from scipy import stats as sps

    dists, groups = _raw_groups(raw)
    palette = ["#4e88d9", "#e09020", "#20b2aa", "#a060e0", "#e05c5c", "#5cb85c"]
    fig, axes = plt.subplots(1, len(dists), figsize=(9 * len(dists), 7), squeeze=False)
    fig.suptitle("Raw Waste Histograms with KDE (30-day horizon)", fontsize=14, fontweight="bold")
    for ax, dist in zip(axes[0], dists, strict=True):
        pooled = np.concatenate([vals for _, vals in groups[dist]])
        ax.hist(
            pooled, bins=60, density=True, color=DIST_COLORS.get(dist, "gray"), alpha=0.35, label="All sizes (pooled)"
        )
        xs = np.linspace(pooled.min(), pooled.max(), 400)
        for i, (lbl, vals) in enumerate(groups[dist]):
            kde = sps.gaussian_kde(vals)
            ax.plot(xs, kde(xs), color=palette[i % len(palette)], linewidth=1.8, label=f"KDE {lbl}")
        ax.set_xlabel("Waste (kg/bin/day)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(DIST_LABELS.get(dist, dist), fontsize=11)
        ax.legend(fontsize=8)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
    plt.tight_layout()
    savefig(fig, out_dir / "npz_hist_kde.png")


def gen_npz_extended_stats(ext: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        ("median", "Median (kg)"),
        ("variance", "Variance (kg²)"),
        ("iqr", "IQR (kg)"),
        ("min", "Minimum (kg)"),
        ("upper_fence", "Upper Outlier Fence (kg)"),
        ("mode", "Mode (kg, binned)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(
        "Extended NPZ Statistics vs Network Size (lines: Rio Maior · diamonds: FFZ-350)", fontsize=14, fontweight="bold"
    )
    rm = ext[ext["city"] == LINE_CITY].sort_values("N")
    ref = ext[ext["city"] == REF_CITY]
    for ax_i, (metric, ylabel) in enumerate(metrics):
        ax = axes[ax_i // 3][ax_i % 3]
        for dist in sorted(ext["dist"].unique()):
            _plot_line_plus_ref(ax, rm[rm["dist"] == dist], ref[ref["dist"] == dist], metric, dist)
        ax.set_xlabel("Network size N", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(metric, fontsize=11)
        if ax_i == 0:
            ax.legend(fontsize=8)
        ax.yaxis.grid(True, alpha=0.4)
        ax.xaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
    plt.tight_layout()
    savefig(fig, out_dir / "npz_extended_stats.png")


def gen_dataset_interactive_html(npz: pd.DataFrame, td: pd.DataFrame, theme: dict, out_dir: Path) -> dict[str, Path]:
    if not HAS_PLOTLY:
        print("  [WARN] Plotly not available — skipping interactive HTML")
        return {}
    template = theme["plotly_template"]
    paths: dict[str, Path] = {}

    sub30 = npz[npz["horizon"] == 30].copy()
    sub30["city_label"] = sub30["city"].map(CITY_LABELS).fillna(sub30["city"])
    sub30["dist_label"] = sub30["dist"].map(DIST_LABELS).fillna(sub30["dist"])

    fig = go.Figure()
    for city in sub30["city_label"].unique():
        for dist in sub30["dist_label"].unique():
            sub = sub30[(sub30["city_label"] == city) & (sub30["dist_label"] == dist)]
            if not len(sub):
                continue
            sym = "circle" if dist == "Empirical" else "square"
            color = CITY_COLORS.get("figueiradafoz" if "Figueira" in city else "riomaior", "#a0a0a0")
            hover = (
                sub["city_label"]
                + "<br>N="
                + sub["N"].astype(str)
                + "<br>Dist: "
                + sub["dist_label"]
                + "<br>Mean kg: "
                + sub["mean_kg"].round(3).astype(str)
                + "<br>Std kg: "
                + sub["std_kg"].round(3).astype(str)
                + "<br>Max kg: "
                + sub["max_kg"].round(1).astype(str)
                + "<br>Skewness: "
                + sub["skewness"].round(3).astype(str)
            )
            fig.add_trace(
                go.Scatter(
                    x=sub["mean_kg"],
                    y=sub["std_kg"],
                    mode="markers",
                    name=f"{city} — {dist}",
                    marker=dict(
                        color=color,
                        symbol=sym,
                        size=(sub["N"] / 20 + 8).tolist(),
                        line=dict(width=1, color=theme["fg"]),
                        opacity=0.85,
                    ),
                    text=hover,
                    hovertemplate="%{text}<extra></extra>",
                )
            )
    fig.update_layout(
        title="NPZ Dataset Statistics — Mean vs Std Waste (30-day horizon)",
        xaxis_title="Mean waste (kg/bin/day)",
        yaxis_title="Std waste (kg/bin/day)",
        template=template,
        height=600,
        hovermode="closest",
        legend_title="City — Distribution",
    )
    p = out_dir / "npz_stats_interactive.html"
    fig.write_html(str(p), include_plotlyjs="cdn")
    paths["npz_stats"] = p
    print(f"  Saved: {p.name}")

    cities = sorted(sub30["city"].unique())
    fig2 = make_subplots(
        rows=1, cols=len(cities), subplot_titles=[CITY_LABELS.get(c, c) for c in cities], shared_yaxes=True
    )
    for col_i, city in enumerate(cities, 1):
        sub = sub30[sub30["city"] == city].sort_values("N")
        for dist in sorted(sub["dist"].unique()):
            dist_l = DIST_LABELS.get(dist, dist)
            sub_d = sub[sub["dist"] == dist]
            hover = (
                sub_d["city"].map(CITY_LABELS).fillna(sub_d["city"])
                + ", N="
                + sub_d["N"].astype(str)
                + ", "
                + dist_l
                + "<br>Mean: "
                + sub_d["mean_kg"].round(3).astype(str)
                + " kg"
                + "<br>Std: "
                + sub_d["std_kg"].round(3).astype(str)
                + " kg"
            )
            fig2.add_trace(
                go.Bar(
                    name=dist_l,
                    x=[f"N={n}" for n in sub_d["N"]],
                    y=sub_d["mean_kg"],
                    error_y=dict(type="data", array=sub_d["std_kg"].tolist(), visible=True),
                    marker_color=DIST_COLORS.get(dist, "gray"),
                    marker_opacity=0.85,
                    text=hover,
                    hovertemplate="%{text}<extra></extra>",
                    legendgroup=dist_l,
                    showlegend=(col_i == 1),
                ),
                row=1,
                col=col_i,
            )
    fig2.update_layout(
        title="Waste Distribution Statistics per City and Network Size",
        yaxis_title="Mean waste (kg/bin/day)",
        template=template,
        barmode="group",
        height=550,
        legend_title="Distribution",
    )
    p2 = out_dir / "waste_distribution_interactive.html"
    fig2.write_html(str(p2), include_plotlyjs="cdn")
    paths["waste_dist"] = p2
    print(f"  Saved: {p2.name}")

    metrics = ["mean_kg", "std_kg", "skewness", "max_kg"]
    fig3 = make_subplots(
        rows=2, cols=2, subplot_titles=["Mean Waste (kg/bin)", "Std Waste (kg/bin)", "Skewness", "Max Waste (kg)"]
    )
    for mi, metric in enumerate(metrics):
        row, col = mi // 2 + 1, mi % 2 + 1
        for city in cities:
            city_l = CITY_LABELS.get(city, city)
            for dist in sorted(sub30["dist"].unique()):
                dist_l = DIST_LABELS.get(dist, dist)
                sub = sub30[(sub30["city"] == city) & (sub30["dist"] == dist)].sort_values("N")
                if not len(sub):
                    continue
                hover = (
                    city_l
                    + "<br>N="
                    + sub["N"].astype(str)
                    + "<br>Dist: "
                    + dist_l
                    + "<br>"
                    + metric
                    + ": "
                    + sub[metric].round(3).astype(str)
                )
                fig3.add_trace(
                    go.Bar(
                        name=f"{city_l} {dist_l}",
                        x=[f"{city_l[:3]} N={n}" for n in sub["N"]],
                        y=sub[metric],
                        marker_color=CITY_COLORS.get(city, "gray"),
                        marker_opacity=0.85 if dist == "emp" else 0.55,
                        text=hover,
                        hovertemplate="%{text}<extra></extra>",
                        legendgroup=f"{city}{dist}",
                        showlegend=(mi == 0),
                    ),
                    row=row,
                    col=col,
                )
    fig3.update_layout(
        title="City & Network Comparison — NPZ Dataset Statistics",
        template=template,
        height=800,
        barmode="group",
        legend_title="City & Distribution",
    )
    p3 = out_dir / "city_network_comparison_interactive.html"
    fig3.write_html(str(p3), include_plotlyjs="cdn")
    paths["city_net"] = p3
    print(f"  Saved: {p3.name}")
    return paths


# ── Table builders ─────────────────────────────────────────────────────────────


def build_npz_table(npz: pd.DataFrame, ext: pd.DataFrame, horizon: int = 30) -> str:
    sub = npz[npz["horizon"] == horizon].merge(ext, on=["city", "N", "dist"], how="left")
    sub["city_label"] = sub["city"].map(CITY_LABELS).fillna(sub["city"])
    sub["dist_label"] = sub["dist"].map(DIST_LABELS).fillna(sub["dist"])
    rows = [
        "| City | N | Distribution | Mean kg | Median kg | Std kg | Max kg | IQR kg | Skewness |",
        "|------|---|-------------|---------|-----------|--------|--------|--------|---------|",
    ]
    for _, row in sub.sort_values(["city", "N", "dist"]).iterrows():
        med = f"{row['median']:.2f}" if pd.notna(row.get("median")) else "—"
        iqr = f"{row['iqr']:.2f}" if pd.notna(row.get("iqr")) else "—"
        rows.append(
            f"| {row.city_label} | {row.N} | {row.dist_label} | "
            f"{row.mean_kg:.2f} | {med} | {row.std_kg:.2f} | {row.max_kg:.1f} | "
            f"{iqr} | {row.skewness:.3f} |"
        )
    return "\n".join(rows)


def build_extended_table(ext: pd.DataFrame) -> str:
    if ext.empty:
        return "_Raw NPZ data unavailable — extended statistics could not be computed._"
    rows = [
        "| City | N | Distribution | Median | Variance | Q1 | Q3 | IQR | Min | Fences (lo/hi) | Mode |",
        "|------|---|-------------|--------|----------|----|----|-----|-----|----------------|------|",
    ]
    for _, r in ext.sort_values(["city", "N", "dist"]).iterrows():
        rows.append(
            f"| {CITY_LABELS.get(r.city, r.city)} | {r.N} | {DIST_LABELS.get(r.dist, r.dist)} | "
            f"{r['median']:.2f} | {r.variance:.2f} | {r.q1:.2f} | {r.q3:.2f} | {r.iqr:.2f} | "
            f"{r['min']:.2f} | {max(r.lower_fence, 0):.2f} / {r.upper_fence:.2f} | {r['mode']:.2f} |"
        )
    return "\n".join(rows)


def build_td_table(td: pd.DataFrame) -> str:
    rows = [
        "| N | Distribution | Instances | Mean Waste | Std Waste | Skewness |",
        "|---|-------------|-----------|------------|-----------|---------|",
    ]
    for _, row in td.sort_values(["N", "dist"]).iterrows():
        dist_l = DIST_LABELS.get(row["dist"], row["dist"])
        rows.append(
            f"| {row.N} | {dist_l} | {int(row.instances):,} | "
            f"{row.waste_mean:.4f} | {row.waste_std:.4f} | {row.waste_skew:.3f} |"
        )
    return "\n".join(rows)


# ── Main ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--theme", choices=["dark", "light"], default=None, help="Chart theme")
    p.add_argument("--npz-csv", default=CFG["npz_csv"])
    p.add_argument("--td-csv", default=CFG["td_csv"])
    p.add_argument("--npz-dir", default=CFG["npz_dir"], help="Directory with raw NPZ dataset files")
    p.add_argument("--out-md", default=CFG["out_md"])
    p.add_argument("--figures-dir", default=CFG["figures_dir"])
    p.add_argument("--private-dir", default=CFG["private_dir"])
    p.add_argument("--force", action="store_true", help="Overwrite existing markdown")
    p.add_argument("--figures-only", action="store_true", help="Generate figures only, skip markdown")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    theme = load_theme(args.theme or CFG.get("theme", "dark"))
    plt.style.use(theme["mplstyle_path"])

    npz_path, td_path = Path(args.npz_csv), Path(args.td_csv)
    out_md = Path(args.out_md)
    figures_dir, private_dir = Path(args.figures_dir), Path(args.private_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {npz_path}")
    npz = pd.read_csv(npz_path)
    td = pd.DataFrame()
    if td_path.exists():
        print(f"Reading: {td_path}")
        td = pd.read_csv(td_path)
    else:
        print(f"  [WARN] TD CSV not found: {td_path}")

    print(f"  NPZ cities: {sorted(npz['city'].unique())}")
    print(f"  NPZ distributions: {sorted(npz['dist'].unique())}")
    print(f"  NPZ horizons: {sorted(npz['horizon'].unique())}")

    print(f"Loading raw NPZ waste data: {args.npz_dir}")
    raw = load_raw_waste(Path(args.npz_dir), npz, horizon=30)
    print(f"  Raw datasets loaded: {len(raw)}")
    ext = build_extended_df(raw)

    print("\nGenerating figures...")
    gen_npz_stats_bar(npz, ext, figures_dir)
    gen_npz_size_scaling(npz, figures_dir)
    gen_npz_city_comparison(npz, ext, figures_dir)
    gen_npz_horizon_comparison(npz, figures_dir)
    if len(td):
        gen_td_stats(td, figures_dir)
        gen_npz_td_alignment(npz, td, figures_dir)
    if raw:
        gen_npz_violin(raw, theme, figures_dir)
        gen_npz_box(raw, theme, figures_dir)
        gen_npz_hist_kde(raw, theme, figures_dir)
        gen_npz_extended_stats(ext, figures_dir)
    html_paths = gen_dataset_interactive_html(npz, td, theme, private_dir)
    print(f"  Generated {len(html_paths)} interactive HTML files")

    if args.figures_only:
        print("--figures-only: skipping markdown generation")
        return
    if out_md.exists() and not args.force:
        print(f"\n{out_md} already exists. Use --force to regenerate. Skipping.")
        return

    # ── Markdown context ────────────────────────────────────────────────────────
    cities = sorted(npz["city"].unique())
    dists = sorted(npz["dist"].unique())
    horizons = sorted(npz["horizon"].unique())
    has_td = bool(len(td))

    toc_items = []
    sec = 1
    if has_td:
        toc_items.append(f"{sec}. [Training Data (TD)](#{sec}-training-data-td)")
        sec += 1
    city_ctxs = []
    for city in cities:
        label = CITY_LABELS.get(city, city)
        anchor = label.lower().replace(" ", "-")
        toc_items.append(f"{sec}. [{label} NPZ Datasets](#{sec}-{anchor}-npz-datasets)")
        npz_sub = npz[npz["city"] == city]
        city_ctxs.append(
            {
                "section_num": sec,
                "label": label,
                "Ns": sorted(npz_sub["N"].unique()),
                "table": build_npz_table(npz_sub, ext),
            }
        )
        sec += 1
    shapes_section_num = sec
    toc_items.append(f"{sec}. [Waste Distribution Shapes](#{sec}-waste-distribution-shapes)")
    sec += 1
    has_city_cmp = len(cities) > 1
    city_cmp_section_num = sec
    if has_city_cmp:
        toc_items.append(f"{sec}. [City Comparison](#{sec}-city-comparison)")
        sec += 1
    alignment_section_num = sec
    if has_td:
        toc_items.append(f"{sec}. [TD vs NPZ Alignment](#{sec}-td-vs-npz-alignment)")

    figures_rel = str(figures_dir).replace("docs/private/", "", 1)
    private_rel = str(private_dir).replace("docs/private/", "", 1)

    print(f"\nGenerating markdown: {out_md}")
    md = render_template(
        "dataset_analysis.md.j2",
        city_labels=[CITY_LABELS.get(c, c) for c in cities],
        dist_labels=[DIST_LABELS.get(d, d) for d in dists],
        horizon_str=", ".join(f"{h} days" for h in horizons),
        has_multiple_horizons=len(horizons) > 1,
        total_npz=len(npz),
        toc="\n".join(toc_items),
        has_td=has_td,
        td_table=build_td_table(td) if has_td else "",
        cities=city_ctxs,
        shapes_section_num=shapes_section_num,
        extended_table=build_extended_table(ext),
        has_city_cmp=has_city_cmp,
        city_cmp_section_num=city_cmp_section_num,
        all_table=build_npz_table(npz, ext),
        alignment_section_num=alignment_section_num,
        figures_rel=figures_rel,
        private_rel=private_rel,
        td_csv=str(td_path),
        npz_csv=str(npz_path),
        interactive=bool(html_paths),
        placeholder=PLACEHOLDER,
    )
    md = finalize_markdown(md)
    out_md.write_text(md, encoding="utf-8")
    print(f"  Written: {out_md} ({len(md)} chars)")


if __name__ == "__main__":
    main()
