"""Network maps of the paper: bin layouts recovered from the road-distance matrices by classical MDS.

Attributes:
    fig_networks: The three road networks side by side.

Example:
    >>> fig_networks(out_dir, cfg)"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from logic.gen.paper_data import (
    NETWORK_MAP_DIR,
    OUTPUT_DIR,
    REPO_ROOT,
)
from logic.gen.report_utils import savefig

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
        print(
            f"  Note: {scenario['dm']} for {scenario['key']} has {len(digests)} distinct "
            f"correctly sized variants; taking the {votes}-way majority (issue #48)"
        )
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
        image = image[max(0, ys.min() - pad) : ys.max() + pad, max(0, xs.min() - pad) : xs.max() + pad]

        # Retain the real geometry while giving the paper figure a legible,
        # cartographic palette instead of a large blank white canvas.
        background = np.all(image > 0.965, axis=2)
        roads = (image[..., 0] > 0.62) & (image[..., 0] < 0.9) & (np.max(image, axis=2) - np.min(image, axis=2) < 0.12)
        bins = (image[..., 0] > 0.55) & (image[..., 1] < 0.52) & (image[..., 2] < 0.52)
        image[background] = np.array([0.965, 0.945, 0.89])
        image[roads] = np.array([0.48, 0.58, 0.65])
        rgb = np.array([int(scenario["color"][i : i + 2], 16) for i in (1, 3, 5)]) / 255
        image[bins] = rgb

        ax.set_facecolor("#f6f2e9")
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#91a0aa")
            spine.set_linewidth(0.8)
        ax.set_title(f"{scenario['city']}  ($N={scenario['N']}$)", fontsize=10, fontweight="bold", pad=7)
        ax.text(
            0.03,
            0.035,
            "selected plastic bins  |  OSM roads",
            transform=ax.transAxes,
            fontsize=6.7,
            color="#35434c",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#fffdf7", "edgecolor": "#aab5bb", "alpha": 0.92},
        )
        ax.annotate(
            "",
            xy=(0.94, 0.94),
            xytext=(0.94, 0.84),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "-|>", "color": "#35434c", "lw": 0.9},
        )
        ax.text(
            0.94,
            0.955,
            "N",
            ha="center",
            va="bottom",
            transform=ax.transAxes,
            fontsize=7,
            fontweight="bold",
            color="#35434c",
        )

    fig.text(0.995, 0.006, "© OpenStreetMap contributors", ha="right", va="bottom", fontsize=5.5, color="#5d6970")
    savefig(fig, out_dir / "networks.png")
    print("    coordinate-derived selected-bin maps over OpenStreetMap road geometry")
