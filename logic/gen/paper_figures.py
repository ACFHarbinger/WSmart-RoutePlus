"""Figures of the paper generator: Pareto fronts, strategy trade-offs, improver pairs, network maps, scale and fill trajectories.

Attributes:
    fig_pareto: Cost-service Pareto front per horizon.
    fig_networks: The three road networks.
    fig_fill_trajectory: Fill-level trajectories under Last-Minute.

Example:
    >>> fig_pareto(clean, 30, out_path, colors, cfg)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from logic.gen.paper_data import (
    REPO_ROOT,
    balance_marginal,
    display_name,
    improver_pairs,
)
from logic.gen.paper_diagrams import fig_policy_space, fig_simulation_loop
from logic.gen.paper_networks import (
    COORD_DIR,
    classical_mds,
    embedding_stress,
    fig_networks,
    find_distance_matrix,
    read_distance_matrix,
    read_network_layout,
)
from logic.gen.report_utils import savefig


def pareto_front(points: pd.DataFrame) -> pd.DataFrame:
    """Non-dominated set: maximise kg/km, minimise overflows."""
    ordered = points.sort_values("kgkm", ascending=False)
    best, keep = np.inf, []
    for idx, row in ordered.iterrows():
        if row.overflows < best:
            keep.append(idx)
            best = row.overflows
    return points.loc[keep].sort_values("kgkm")


def fig_pareto(clean: pd.DataFrame, horizon: int, out: Path, colors: dict, cfg: dict) -> None:
    """kg/km against overflows, per routing algorithm, with the non-dominated front drawn."""
    sub = clean[clean.horizon == horizon]
    agg = sub.groupby("constructor").agg(kgkm=("kgkm", "mean"), overflows=("overflows", "mean"))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, row in agg.iterrows():
        ax.scatter(row.overflows, row.kgkm, s=120, zorder=3, color=colors.get(name, "#4e88d9"), edgecolor="none")
        # Points are annotated in place rather than via a legend: eight labels in
        # a legend box would cover the front on a figure this small.
        ax.annotate(
            display_name(name, cfg), (row.overflows, row.kgkm), textcoords="offset points", xytext=(7, 4), fontsize=8
        )
    front = pareto_front(agg)
    ax.plot(
        front.overflows, front.kgkm, drawstyle="steps-post", linewidth=1.4, linestyle="--", color="#888888", zorder=2
    )
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
    # body text uses (LA, LM-CF70, LM-CF90, SL1, SL2).
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
    ax.bar(np.arange(len(delta)), delta, width=1.0, color=np.where(delta > 0, "#5cb85c", "#c04070"))
    ax.axhline(0, color="#555555", linewidth=0.9)
    ax.set_xlabel(cfg["axis"]["pair_index"].format(pairs=len(delta)))
    ax.set_ylabel(cfg["axis"]["kgkm_delta"])
    ax.set_title(
        cfg["figure_titles"]["improver"].format(horizon=horizon, wins=int((delta > 0).sum()), pairs=len(delta))
    )
    ax.grid(True, axis="y", alpha=0.25)
    savefig(fig, out / f"improver_delta_{horizon}d.png")


def fig_scale(clean: pd.DataFrame, horizon: int, out: Path, colors: dict, cfg: dict) -> None:
    """Runtime against instance size, per routing algorithm -- the cost side of the trade-off."""
    sub = clean[clean.horizon == horizon]
    agg = sub.pivot_table(index="N", columns="constructor", values="time", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name in agg.columns:
        ax.plot(
            agg.index,
            agg[name],
            marker="o",
            linewidth=1.6,
            color=colors.get(name, "#4e88d9"),
            label=display_name(name, cfg),
        )
    ax.set_xlabel(cfg["axis"]["n_bins"])
    ax.set_ylabel(cfg["axis"]["time"])
    ax.set_yscale("log")
    ax.set_xticks(sorted(agg.index))
    ax.set_title(cfg["figure_titles"]["scale"].format(horizon=horizon))
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    savefig(fig, out / f"runtime_scaling_{horizon}d.png")


def _simulate_last_minute(increments: np.ndarray, tau: float, capacity: float) -> np.ndarray:
    """Record post-arrival levels, then immediately serve threshold requests."""
    n_bins, n_days = increments.shape
    level = np.zeros(n_bins)
    out = np.zeros_like(increments)
    for d in range(n_days):
        level = np.minimum(level + increments[:, d], capacity)
        out[:, d] = level.copy()
        collect = level >= tau
        level[collect] = 0.0
    return out


def fig_fill_trajectory(out_dir: Path) -> None:
    """
    The multi-period mechanism behind the paper's central claim, as a figure.

    This threshold-only illustration deposits waste before selection and serves
    every selected bin immediately. It omits routing and optional visits; a
    large arrival can cause overflow before either threshold triggers service.
    Bins are chosen by input-accumulation ranks, not observed policy outcomes.
    """
    # The scenario's daily deposits, in % of bin capacity, as (bins, days). They are the
    # same for every policy; the simulator's fill_history workbooks hold this same matrix.
    dataset = REPO_ROOT / "data/simulator/datasets/riomaior100_gamma3_wsr30_N1_seed42.npz"
    if not dataset.exists():
        print(f"  Skipped fill-trajectory figure (missing {dataset.name}).")
        return
    increments = np.load(dataset)["waste"][0].T.astype(float)
    capacity = 100.0

    levels_70 = _simulate_last_minute(increments, 70.0, capacity)
    levels_90 = _simulate_last_minute(increments, 90.0, capacity)

    # Match the manuscript's one-based input ranks: 90th, 50th and 10th of 100.
    total = increments.sum(axis=1)
    order = np.argsort(total, kind="stable")
    heavy, median, light = (int(order[rank - 1]) for rank in (90, 50, 10))

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
    fig.suptitle("Last-Minute selection at two thresholds (Rio Maior, $N=100$, Gamma-3)", fontsize=10, y=1.02)
    fig.tight_layout()
    savefig(fig, out_dir / "fill_trajectory_30d.png")


__all__ = [
    "pareto_front",
    "fig_pareto",
    "fig_strategy_tradeoff",
    "fig_improver_paired",
    "find_distance_matrix",
    "read_distance_matrix",
    "classical_mds",
    "embedding_stress",
    "read_network_layout",
    "fig_networks",
    "fig_scale",
    "fig_fill_trajectory",
    "fig_policy_space",
    "fig_simulation_loop",
    "COORD_DIR",
]
