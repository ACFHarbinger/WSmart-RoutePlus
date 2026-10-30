"""Depot placement for the simulator networks.

The municipal networks keep the depot at a remote site. A central depot makes one of the bins
(or a synthetic point at the centroid) the depot, so the effect of the access distance can be
separated from that of the policies.

Attributes:
    DEPOT_MODES: Accepted placement modes.
    select_depot_bin: Index of the bin that becomes the depot under a bin-based mode.
    relocate_depot: Distance matrix with a bin promoted to the depot.
    bins_to_load: Number of bins to load so that ``num_loc`` customers remain after relocation.
    extend_subset: Bin subset with one spare bin, so a promoted bin leaves the customer count unchanged.
    promote_central_depot: Processed simulator frames with a bin promoted to the depot.

Example:
    >>> node = select_depot_bin("centroid", coords=np.array([[0.0, 0.0], [0.0, 2.0], [0.1, 1.0]]))
    >>> node
    2
"""

from __future__ import annotations

from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEPOT_MODES = ("remote", "centroid", "medoid", "synthetic")
DepotMode = Literal["remote", "centroid", "medoid", "synthetic"]

_EARTH_RADIUS_KM = 6371.0088


def _great_circle_km(points: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Haversine distance (km) from each ``[lat, lng]`` row of ``points`` to ``target``.

    Args:
        points: Array of shape (n, 2) of latitudes and longitudes in degrees.
        target: Array of shape (2,) with the latitude and longitude in degrees.

    Returns:
        np.ndarray: Distances of shape (n,).
    """
    lat1, lng1 = np.radians(points[:, 0]), np.radians(points[:, 1])
    lat2, lng2 = np.radians(target[0]), np.radians(target[1])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def centroid(coords: np.ndarray) -> np.ndarray:
    """Mean ``[lat, lng]`` of the bins (adequate at the scale of one municipality).

    Args:
        coords: Array of shape (n, 2) with the bin latitudes and longitudes in degrees.

    Returns:
        np.ndarray: Centroid of shape (2,).
    """
    return np.asarray(coords, dtype=np.float64).mean(axis=0)


def select_depot_bin(
    mode: DepotMode,
    coords: Optional[np.ndarray] = None,
    distance_matrix: Optional[np.ndarray] = None,
) -> int:
    """Pick the bin that becomes the depot.

    Args:
        mode: ``centroid`` takes the bin nearest to the centroid of the bins by great-circle
            distance, and ``medoid`` the bin with the smallest total road distance to the others
            in both directions.
        coords: Bin coordinates of shape (n, 2), required by ``centroid``.
        distance_matrix: Bin-to-bin road distances of shape (n, n) without the depot, required by
            ``medoid``.

    Returns:
        int: Zero-based index of the chosen bin.

    Raises:
        ValueError: If the mode is not bin-based or its input is missing.
    """
    if mode == "centroid":
        if coords is None or len(coords) == 0:
            raise ValueError("centroid depot needs the bin coordinates")
        points = np.asarray(coords, dtype=np.float64)
        return int(np.argmin(_great_circle_km(points, centroid(points))))
    if mode == "medoid":
        if distance_matrix is None:
            raise ValueError("medoid depot needs the bin-to-bin distance matrix")
        matrix = np.asarray(distance_matrix, dtype=np.float64)
        return int(np.argmin(matrix.sum(axis=0) + matrix.sum(axis=1)))
    raise ValueError(f"{mode!r} does not select a bin; use 'centroid' or 'medoid'")


def relocate_depot(distance_matrix: np.ndarray, bin_node: int) -> Tuple[np.ndarray, np.ndarray]:
    """Promote bin ``bin_node`` to the depot and drop the original depot.

    Args:
        distance_matrix: Matrix of shape (n + 1, n + 1) where node 0 is the depot and nodes
            1..n are the bins.
        bin_node: Node (1..n) of the bin that becomes the depot.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The (n, n) matrix whose node 0 is the new depot followed by
        the remaining bins in their original order, and the original node ids of its rows.

    Raises:
        ValueError: If ``bin_node`` is not a bin of the matrix.
    """
    matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = matrix.shape[0] - 1
    if not 1 <= bin_node <= n:
        raise ValueError(f"bin_node must be in 1..{n}, got {bin_node}")
    nodes = np.array([bin_node, *[i for i in range(1, n + 1) if i != bin_node]])
    return matrix[np.ix_(nodes, nodes)], nodes


def bins_to_load(num_loc: int, mode: DepotMode) -> int:
    """Number of bins to load so that ``num_loc`` customers remain after the depot is placed.

    Args:
        num_loc: Customer count requested by the graph configuration.
        mode: Depot placement mode.

    Returns:
        int: ``num_loc`` for the remote depot, ``num_loc + 1`` when a bin becomes the depot.
    """
    return num_loc if mode == "remote" else num_loc + 1


def extend_subset(indices: Sequence[int], n_available: int) -> List[int]:
    """Add the lowest unused bin position to a bin subset.

    A network defined by a bin subset loses one customer when a bin becomes the depot. The spare
    bin replaces it, so the routed network keeps its customer count.

    Args:
        indices: Positions of the subset bins in the loaded bin frame.
        n_available: Number of bins in the loaded frame.

    Returns:
        List[int]: The sorted subset positions plus the lowest position outside the subset.

    Raises:
        ValueError: If every available bin is already in the subset.
    """
    taken = set(int(i) for i in indices)
    spare = next((i for i in range(n_available) if i not in taken), None)
    if spare is None:
        raise ValueError("no spare bin is left to replace the promoted depot")
    return sorted([*taken, spare])


def promote_central_depot(
    data: pd.DataFrame,
    coords: pd.DataFrame,
    mode: DepotMode,
    candidate_ids: Optional[Sequence[int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Promote one bin of the processed simulator frames to the depot.

    The frames are the output of ``process_data``: row 0 is the depot and the other rows are the
    bins. The chosen bin takes the depot row, keeping the depot's stock and accumulation values,
    and leaves the bin set. The remaining bins keep their order and the ``#bin`` column is
    renumbered from 0. The distance matrix is read by ID, so it needs no change.

    Args:
        data: Frame with ``ID``, ``Stock`` and ``Accum_Rate`` columns.
        coords: Frame with ``ID``, ``Lat`` and ``Lng`` columns, in the row order of ``data``.
        mode: ``remote`` returns the frames unchanged and ``centroid`` promotes the bin nearest
            to the centroid of the bins.
        candidate_ids: Bin IDs that define the centroid and may become the depot (default: every
            bin). A spare bin added by ``extend_subset`` is left out of them.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The data and coordinate frames with the new depot.

    Raises:
        NotImplementedError: For ``medoid`` and ``synthetic``, which need distances that are not
            available before the distance matrix is built.
        ValueError: If the mode is unknown or the frames disagree on the node IDs.
    """
    if mode not in DEPOT_MODES:
        raise ValueError(f"depot mode must be one of {DEPOT_MODES}, got {mode!r}")
    if mode == "remote":
        return data, coords
    if mode != "centroid":
        raise NotImplementedError(f"depot mode {mode!r} is not available at initialization")
    if not (data["ID"].to_numpy() == coords["ID"].to_numpy()).all():
        raise ValueError("data and coords must list the same node IDs in the same order")
    bins = coords.iloc[1:]
    if candidate_ids is not None:
        bins = bins[bins["ID"].isin(list(candidate_ids))]
    chosen = int(bins.index[select_depot_bin("centroid", coords=bins[["Lat", "Lng"]].to_numpy())])
    order = [chosen, *[row for row in range(1, len(coords)) if row != chosen]]

    def relocate(frame: pd.DataFrame, depot_values: pd.Series) -> pd.DataFrame:
        moved = frame.iloc[order].reset_index(drop=True)
        for column in depot_values.index:
            moved.loc[0, column] = depot_values[column]
        if "#bin" in moved.columns:
            moved["#bin"] = np.arange(len(moved)).astype(frame["#bin"].dtype)
        return moved.astype(frame.dtypes.to_dict())

    return (
        relocate(data, data.iloc[0][[c for c in ("Stock", "Accum_Rate") if c in data.columns]]),
        relocate(coords, pd.Series(dtype=float)),
    )
