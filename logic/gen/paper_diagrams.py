"""Schematic figures of the paper: the policy space and the simulation loop.

Attributes:
    fig_policy_space: The packing x routing x improver policy space.
    fig_simulation_loop: The daily simulation loop.

Example:
    >>> fig_simulation_loop(out_dir, cfg)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from logic.gen.report_utils import savefig


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
                (centre - box_w / 2, box_bot),
                box_w,
                box_top - box_bot,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                facecolor=stage["fill"],
                edgecolor=colour,
                linewidth=1.4,
                zorder=2,
            )
        )
        ax.text(
            centre,
            box_top - 0.07,
            stage["title"],
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=colour,
            zorder=3,
        )
        ax.text(
            centre,
            box_top - 0.135,
            stage["scope"],
            ha="center",
            va="center",
            fontsize=7.8,
            style="italic",
            color="#5a6673",
            zorder=3,
        )
        ax.text(
            centre,
            (box_top + box_bot) / 2 - 0.055,
            "\n".join(stage["members"]),
            ha="center",
            va="center",
            fontsize=8.6,
            color="#26313d",
            zorder=3,
            linespacing=1.5,
        )
        ax.text(
            centre,
            box_bot + 0.045,
            stage["footer"].format(**counts),
            ha="center",
            va="center",
            fontsize=7.6,
            color="#5a6673",
            zorder=3,
        )

    for a, b in zip(centres, centres[1:], strict=False):
        ax.annotate(
            "",
            xy=(b - box_w / 2 - 0.008, (box_top + box_bot) / 2),
            xytext=(a + box_w / 2 + 0.008, (box_top + box_bot) / 2),
            arrowprops=dict(arrowstyle="-|>", linewidth=1.6, color="#94a2b0"),
            zorder=1,
        )

    ax.text(
        0.5,
        0.20,
        cfg["policy_space"]["benchmarked"].format(**counts),
        ha="center",
        va="center",
        fontsize=9.2,
        color="#26313d",
        zorder=3,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#eef2f5", edgecolor="#d5dde4"),
    )
    ax.text(
        0.5,
        0.045,
        cfg["policy_space"]["footnote"].format(**counts),
        ha="center",
        va="center",
        fontsize=7.8,
        color="#5a6673",
        zorder=3,
    )

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
    c_policy_border = "#059669"
    c_policy_fill = "#f0fdf4"
    c_eval_border = "#b91c1c"
    c_eval_fill = "#fef2f2"
    c_text_dark = "#1e293b"

    # Outer Day Container
    day_box = mpatches.FancyBboxPatch(
        (0.012, 0.03),
        0.976,
        0.94,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
        linewidth=1.5,
        zorder=0,
    )
    ax.add_patch(day_box)
    ax.text(
        0.035,
        0.93,
        "SIMULATED DAY CYCLE  (Day $d \\in \\{1, \\dots, D\\}$)",
        fontsize=9.5,
        fontweight="bold",
        color="#334155",
        zorder=1,
    )

    # 1. Environment & Accumulation Box
    box_env = mpatches.FancyBboxPatch(
        (0.055, 0.54),
        0.25,
        0.35,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=c_env_fill,
        edgecolor=c_env_border,
        linewidth=1.3,
        zorder=1,
    )
    ax.add_patch(box_env)
    ax.text(
        0.180, 0.845, "1. Arrival and service flag", ha="center", fontsize=8.6, fontweight="bold", color=c_env_border
    )
    ax.text(
        0.180,
        0.74,
        "Add $a_{i,d}$, then cap at $\\mathrm{Cap}_i$.\n$o_{i,d}=1$ if that level equals $\\mathrm{Cap}_i$.\nLost mass is only the excess.\nScored before the route.",
        ha="center",
        va="center",
        fontsize=7.2,
        color=c_text_dark,
        linespacing=1.25,
    )
    ax.text(
        0.180, 0.585, "Decision state, pre-collection", ha="center", fontsize=7.0, fontweight="bold", color="#64748b"
    )

    # 2. Noisy Sensing Box (Asymmetry Highlight)
    box_noise = mpatches.FancyBboxPatch(
        (0.055, 0.13),
        0.25,
        0.34,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor="#f8fafc",
        edgecolor="#94a3b8",
        linewidth=1.3,
        zorder=1,
    )
    ax.add_patch(box_noise)
    ax.text(0.180, 0.43, "2. Sensor (unused)", ha="center", fontsize=9.2, fontweight="bold", color="#64748b")
    # The sensor model is a framework capability; the reported runs do not
    # exercise it. Earlier versions of this figure asserted the opposite in a
    # highlighted badge, contradicting the simulation protocol -- so the noise
    # term is now shown greyed, with the operative sigma = 0 identity in black.
    ax.text(
        0.180,
        0.365,
        "Optional sensor noise:\n$\\epsilon_{i,d} \\sim \\mathcal{N}(0, \\sigma^2)$",
        ha="center",
        va="center",
        fontsize=7.6,
        color="#94a3b8",
        linespacing=1.3,
    )
    ax.text(
        0.180,
        0.285,
        "This study: $\\sigma = 0$",
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color=c_text_dark,
    )

    # Scope Callout Badge 1 -- capability vs. exercised path
    ax.text(
        0.180,
        0.19,
        "Noise supported, not exercised:\nevery reported run sets $\\sigma = 0$,\nso policies observe true levels",
        ha="center",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        color="#475569",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f1f5f9", edgecolor="#94a3b8", linewidth=0.8),
    )

    # Arrow 1 -> 2
    ax.annotate(
        "", xy=(0.180, 0.48), xytext=(0.180, 0.535), arrowprops=dict(arrowstyle="-|>", linewidth=1.4, color="#94a3b8")
    )

    # 3. Policy Pipeline Big Container (Middle)
    box_policy_bg = mpatches.FancyBboxPatch(
        (0.34, 0.12),
        0.35,
        0.76,
        boxstyle="round,pad=0.015,rounding_size=0.015",
        facecolor=c_policy_fill,
        edgecolor=c_policy_border,
        linewidth=1.4,
        zorder=1,
    )
    ax.add_patch(box_policy_bg)
    ax.text(
        0.515, 0.84, "3. Modular Policy Pipeline", ha="center", fontsize=9.6, fontweight="bold", color=c_policy_border
    )
    ax.text(
        0.515,
        0.80,
        "Three Independently Swappable Plugin Registries",
        ha="center",
        fontsize=7.0,
        style="italic",
        color="#047857",
    )

    # 3a. Stage 1: Packing
    box_p1 = mpatches.FancyBboxPatch(
        (0.36, 0.58),
        0.31,
        0.18,
        boxstyle="round,pad=0.01,rounding_size=0.012",
        facecolor="#ffffff",
        edgecolor="#059669",
        linewidth=1.1,
        zorder=2,
    )
    ax.add_patch(box_p1)
    ax.text(0.515, 0.72, "Stage 1: Packing", ha="center", fontsize=8.4, fontweight="bold", color="#065f46")
    ax.text(
        0.515,
        0.64,
        "Input: capped level\nOutput: mandatory set $\\mathcal{M}_d$\n(LM-CF70/LM-CF90, LA, SL1/SL2)",
        ha="center",
        va="center",
        fontsize=7.2,
        color=c_text_dark,
        linespacing=1.2,
    )

    # 3b. Stage 2: Routing
    box_p2 = mpatches.FancyBboxPatch(
        (0.36, 0.36),
        0.31,
        0.18,
        boxstyle="round,pad=0.01,rounding_size=0.012",
        facecolor="#ffffff",
        edgecolor="#059669",
        linewidth=1.1,
        zorder=2,
    )
    ax.add_patch(box_p2)
    ax.text(0.515, 0.50, "Stage 2: Routing", ha="center", fontsize=8.4, fontweight="bold", color="#065f46")
    ax.text(
        0.515,
        0.42,
        "Input: $\\mathcal{M}_d$, directed $\\mathrm{dist}_{ij}$\nOutput: a tour (load not re-checked)\n(eight classical routing algorithms)",
        ha="center",
        va="center",
        fontsize=7.2,
        color=c_text_dark,
        linespacing=1.2,
    )

    # 3c. Stage 3: Route Improvement
    box_p3 = mpatches.FancyBboxPatch(
        (0.36, 0.14),
        0.31,
        0.18,
        boxstyle="round,pad=0.01,rounding_size=0.012",
        facecolor="#ffffff",
        edgecolor="#059669",
        linewidth=1.1,
        zorder=2,
    )
    ax.add_patch(box_p3)
    ax.text(0.515, 0.28, "Stage 3: Route Improvement", ha="center", fontsize=8.4, fontweight="bold", color="#065f46")
    ax.text(
        0.515,
        0.20,
        "Input: constructed tour\nOutput: refined tour\n(Classical Local Search, Fast-TSP)",
        ha="center",
        va="center",
        fontsize=7.2,
        color=c_text_dark,
        linespacing=1.2,
    )

    # Internal pipeline arrows
    ax.annotate(
        "", xy=(0.515, 0.55), xytext=(0.515, 0.58), arrowprops=dict(arrowstyle="-|>", linewidth=1.3, color="#059669")
    )
    ax.annotate(
        "", xy=(0.515, 0.33), xytext=(0.515, 0.36), arrowprops=dict(arrowstyle="-|>", linewidth=1.3, color="#059669")
    )

    # Arrow Sensing -> Stage 1
    ax.annotate(
        "",
        xy=(0.355, 0.67),
        xytext=(0.315, 0.30),
        arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=-0.25", linewidth=1.5, color="#94a3b8"),
    )

    # 4. Route Execution Box (Top Right)
    box_exec = mpatches.FancyBboxPatch(
        (0.73, 0.54),
        0.23,
        0.34,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=c_env_fill,
        edgecolor=c_env_border,
        linewidth=1.3,
        zorder=1,
    )
    ax.add_patch(box_exec)
    ax.text(0.845, 0.84, "4. Route Execution", ha="center", fontsize=9.2, fontweight="bold", color=c_env_border)
    ax.text(
        0.845,
        0.71,
        "Empties visited bins.\nKilometers use directed $\\mathrm{dist}_{ij}$.\nThese runs do not reject\na load above $Q$.",
        ha="center",
        va="center",
        fontsize=7.3,
        color=c_text_dark,
        linespacing=1.25,
    )
    ax.text(0.845, 0.58, "Physical Collection", ha="center", fontsize=7.0, fontweight="bold", color="#64748b")

    # Arrow Policy -> Execution
    ax.annotate(
        "",
        xy=(0.725, 0.71),
        xytext=(0.675, 0.23),
        arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=-0.3", linewidth=1.5, color="#059669"),
    )

    # 5. Exact Accounting Box (Bottom Right)
    box_eval = mpatches.FancyBboxPatch(
        (0.73, 0.12),
        0.23,
        0.34,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=c_eval_fill,
        edgecolor=c_eval_border,
        linewidth=1.3,
        zorder=1,
    )
    ax.add_patch(box_eval)
    ax.text(0.845, 0.42, "5. Collection log", ha="center", fontsize=9.2, fontweight="bold", color=c_eval_border)
    ax.text(
        0.845,
        0.32,
        "Profit $= R\\cdot\\mathrm{kg} - C\\cdot\\mathrm{km}$.\nEfficiency $=$ total kg $/$ total km.\nRevised time: all three policy stages.",
        ha="center",
        va="center",
        fontsize=7.1,
        color=c_text_dark,
        linespacing=1.25,
    )

    # Callout Badge 2 -- accounting reads the true state, not the sensed signal.
    # (With sigma = 0 the two coincide, so this is a statement about the
    # accounting contract, not an observed asymmetry in these runs.)
    ax.text(
        0.845,
        0.18,
        "Overflow and lost mass\nare recorded in step 1,\nnot after the route",
        ha="center",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fee2e2", edgecolor="#ef4444", linewidth=0.8),
    )

    # Arrow Execution -> Accounting
    ax.annotate(
        "",
        xy=(0.845, 0.47),
        xytext=(0.845, 0.53),
        arrowprops=dict(arrowstyle="-|>", linewidth=1.4, color=c_eval_border),
    )

    # Loop back arrow: 5 -> Next Day State -> 1.
    # Routed as an orthogonal polyline through the clear lane below the boxes
    # and up the left margin.
    lane_y, lane_x = 0.055, 0.024
    ax.plot(
        [0.845, 0.845, lane_x, lane_x],
        [0.108, lane_y, lane_y, 0.715],
        color="#475569",
        linewidth=1.6,
        linestyle="--",
        zorder=1,
        solid_capstyle="butt",
    )
    ax.annotate(
        "", xy=(0.042, 0.715), xytext=(lane_x, 0.715), arrowprops=dict(arrowstyle="-|>", linewidth=1.6, color="#475569")
    )
    # The label sits on the lane; its opaque bbox breaks the dashed line, which
    # reads as one routed path rather than as a line colliding with text.
    ax.text(
        0.48,
        lane_y,
        "Next day: capped residual carries forward to day $d+1$",
        fontsize=7.4,
        style="italic",
        color="#475569",
        ha="center",
        va="center",
        zorder=3,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )

    savefig(fig, out_dir / "simulation_loop.png")
