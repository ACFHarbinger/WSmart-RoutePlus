"""
K-Means Geographic Sector Selection Module.

Pre-partitions bins into G geographic sectors via K-means clustering on bin
coordinates. On each operational day t, exactly one sector is mandated:

    sector_index = (t - 1) % G

This mirrors the zone-day schedules used in real-world waste collection while
ensuring spatially proximate bins share the same collection day, naturally
compressing per-day routing distance compared to a random or index-based
partition.

Clustering is performed lazily on the first call and cached for the lifetime
of the process; repeated calls with identical coordinates are free.

Attributes:
    _CLUSTER_CACHE: Module-level dict mapping (n_bins, n_sectors, coord_hash)
                    to a per-bin integer label array.

Example:
    >>> from logic.src.policies.mandatory.selection_kmeans_sector import KMeansGeographicSectorSelection
    >>> strategy = KMeansGeographicSectorSelection()
    >>> bins, ctx = strategy.pack(context)
"""

import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context import PackingContext, SearchContext
from logic.src.interfaces.packing import IPackingAlgorithm
from logic.src.policies.packing.base import PackingRegistry

# Module-level cache: (n_bins, n_sectors, coord_hash) -> integer label array.
# Avoids re-running K-means on every call within a simulation episode.
_CLUSTER_CACHE: Dict[Tuple[int, int, str], np.ndarray] = {}


def _fit_kmeans(coordinates: np.ndarray, n_sectors: int) -> np.ndarray:
    """
    Assign each bin a sector label in {0, ..., n_sectors - 1} via K-means.

    Results are cached on a (shape, n_sectors, MD5-of-data) key so that
    re-runs within the same process incur no additional clustering cost.

    If scikit-learn is unavailable, falls back to a deterministic quantile
    partition on the first principal coordinate (sorted x-axis), which
    preserves contiguous spatial grouping at zero additional cost.

    Args:
        coordinates: Float array of shape (n_bins, 2) — (x, y) or (lat, lon).
        n_sectors:   Number of clusters G ≥ 1.

    Returns:
        Integer array of shape (n_bins,) with values in {0, ..., n_sectors-1}.
    """
    n_bins = coordinates.shape[0]
    coord_hash = hashlib.md5(coordinates.tobytes()).hexdigest()
    cache_key = (n_bins, n_sectors, coord_hash)

    if cache_key in _CLUSTER_CACHE:
        return _CLUSTER_CACHE[cache_key]

    # Guard: more sectors than bins is degenerate — each bin becomes its own sector.
    effective_sectors = min(n_sectors, n_bins)

    try:
        km = KMeans(n_clusters=effective_sectors, n_init=10, random_state=0)
        labels: np.ndarray = km.fit_predict(coordinates)
    except ImportError:
        # Fallback: sort bins by x-coordinate and assign contiguous quantile
        # bands. Not as spatially optimal as K-means but fully deterministic.
        order = np.argsort(coordinates[:, 0])
        labels = np.empty(n_bins, dtype=int)
        labels[order] = np.arange(n_bins) * effective_sectors // n_bins

    _CLUSTER_CACHE[cache_key] = labels
    return labels


@GlobalRegistry.register(
    PolicyTag.PACKING,
    PolicyTag.HEURISTIC,
    PolicyTag.DETERMINISTIC,
)
@PackingRegistry.register("kmeans_sector")
class KMeansGeographicSectorSelection(IPackingAlgorithm):
    """Cyclic geographic-sector collection strategy.

    Bins are pre-partitioned into G sectors via K-means on their coordinates.
    On operational day t exactly one sector is mandated.

    Attributes:
        min_fill (float): Minimum fill ratio for a bin to be eligible.
                          Defaults to 0.0.
    """

    def pack(self, context: PackingContext) -> Tuple[List[int], SearchContext]:
        """Return bin IDs belonging to today's active geographic sector.

        On days where coordinates are unavailable the method returns an empty
        selection rather than raising, preserving graceful degradation in
        partially-initialised environments.

        Args:
            context: PackingContext providing coordinates, current_day,
                threshold (number of sectors), and optionally current_fill.

        Returns:
            Tuple[List[int], SearchContext]: A 2-tuple containing:
                - List of 1-based bin IDs in the active sector meeting min_fill.
                - SearchContext populated with strategy metrics.
        """
        current_day: int = getattr(context, "current_day", 1)
        n_sectors: int = max(1, int(getattr(context, "threshold", 5)))
        min_fill: float = float(getattr(self, "min_fill", 0.0))

        coordinates: Optional[np.ndarray] = getattr(context, "coordinates", None)
        current_fill: Optional[np.ndarray] = getattr(context, "current_fill", None)

        active_sector: int = (current_day - 1) % n_sectors

        metrics: dict = {
            "strategy": "KMeansGeographicSectorSelection",
            "active_sector": active_sector,
            "n_sectors": n_sectors,
        }

        if coordinates is None or len(coordinates) == 0:
            metrics["n_selected"] = 0
            return [], SearchContext.initialize(selection_metrics=metrics)

        coordinates = np.asarray(coordinates, dtype=float)
        labels = _fit_kmeans(coordinates, n_sectors)

        eligible_bins: List[int] = []
        n_bins = len(labels)

        for i in range(n_bins):
            if labels[i] != active_sector:
                continue
            # Apply fill-level gate if fill data is present.
            if current_fill is not None and len(current_fill) > i and current_fill[i] < min_fill:
                continue
            eligible_bins.append(i + 1)  # 1-based indexing for routing

        metrics["n_selected"] = len(eligible_bins)
        return eligible_bins, SearchContext.initialize(selection_metrics=metrics)
