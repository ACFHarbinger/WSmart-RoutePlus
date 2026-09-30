"""
Strategy for loading distance matrices from files.

Attributes:
    FileStrategy: Strategy for loading distance matrices from disk.

Example:
    from logic.src.data.network.file import FileStrategy
    strategy = FileStrategy()
    strategy.calculate(coords, dm_filepath="distance_matrix.csv")
"""

import os
from typing import Any, Iterable

import numpy as np
import pandas as pd

from logic.src.constants import ROOT_DIR

from .base import DistanceStrategy


class FileStrategy(DistanceStrategy):
    """Strategy for loading distance matrices from disk.

    Attributes:
        None
    """

    def calculate(self, coords: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """
        Loads a pre-computed distance matrix from a CSV file.



        Args:
            coords: DataFrame with coordinates (must contain 'ID', 'lat', 'lng' columns).
            kwargs: Additional arguments for the distance strategy.

        Returns:
            np.ndarray: Loaded distance matrix.
        """
        assert self._eval_kwarg("dm_filepath", kwargs), "Missing 'dm_filepath' in kwargs for FileStrategy."
        dm_filepath = kwargs["dm_filepath"]

        # Path resolution (consistent with compute_distance_matrix)
        filename_only = os.path.basename(dm_filepath) == dm_filepath and not os.path.isabs(dm_filepath)
        matrix_path = (
            os.path.join(
                ROOT_DIR,
                "data",
                "simulator",
                "distance_matrix",
                dm_filepath,
            )
            if filename_only
            else dm_filepath
        )

        if not os.path.isfile(matrix_path):
            raise FileNotFoundError(f"Distance matrix file not found: {matrix_path}")

        raw = np.loadtxt(matrix_path, delimiter=",")
        if getattr(raw, "ndim", 0) != 2:
            raise ValueError(
                f"Distance matrix {matrix_path} must be a 2-d array, got shape {getattr(raw, 'shape', None)}"
            )

        req_ids = coords["ID"].to_numpy()
        # Archived osm_distmat.csv and the simulator saver write one header row of bin
        # ids and then an N×N body with no id column. loadtxt sees (N+1, N). Dropping
        # the first column as well builds an N×(N-1) matrix (index 350 on a length-350 axis).
        if raw.shape[0] == raw.shape[1] + 1:
            return _matrix_for_bin_ids(raw[1:], raw[0], req_ids)

        # Square files that carry an id column as well as an id row (the gmaps dumps).
        distance_matrix = raw[1:, 1:]

        # Handle focus_idx if present
        if self._eval_kwarg("focus_idx", kwargs) and kwargs["focus_idx"] is not None:
            focus_idx = kwargs["focus_idx"]
            idx_list = list(focus_idx[0]) if isinstance(focus_idx[0], Iterable) else list(focus_idx)
            # Add 0 for depot (assumed to be at index 0 in the original matrix)
            # The matrix loaded [1:, 1:] already maps to indices [0...N-1] where 0 is depot
            idx = np.array([-1] + idx_list) + 1
            idx = idx[idx < distance_matrix.shape[0]]  # ensure bounds
            return distance_matrix[np.ix_(idx, idx)]

        # Try to slice safely by ID matching if shapes differ
        if distance_matrix.shape[0] != len(req_ids):
            try:
                with open(matrix_path, "r") as f:
                    first_line = f.readline().strip().split(",")
                # Safely assign IDs to the columns of distance_matrix
                matrix_ids = np.array([float(x) for x in first_line[-distance_matrix.shape[1] :]])
                float_req_ids = np.array([float(x) for x in req_ids])

                if np.isin(float_req_ids, matrix_ids).all():
                    indices = [np.where(matrix_ids == i)[0][0] for i in float_req_ids]
                    return distance_matrix[np.ix_(indices, indices)]
            except Exception:
                pass

            # Fallback to direct top-N slicing
            if len(req_ids) <= distance_matrix.shape[0]:
                return distance_matrix[: len(req_ids), : len(req_ids)]

        return distance_matrix


def _bin_id(value: Any) -> Any:
    """Normalise a header cell and a coordinate ID onto one comparable key.

    Args:
        value: Raw id from a matrix header or a coordinate frame.

    Returns:
        An ``int`` when the value is an integral number, otherwise the original value.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if np.isfinite(number) and number == int(number):
        return int(number)
    return value


def _matrix_for_bin_ids(distance_matrix: np.ndarray, matrix_ids: np.ndarray, req_ids: np.ndarray) -> np.ndarray:
    """Return the square submatrix whose rows and columns follow ``req_ids``.

    Args:
        distance_matrix: Square distances aligned with ``matrix_ids``.
        matrix_ids: Bin id of each row and column, in matrix order.
        req_ids: Bin ids to keep, in the order the caller asked for.

    Returns:
        Submatrix of shape ``(len(req_ids), len(req_ids))``.

    Raises:
        ValueError: The body is not square, or a requested bin id is absent.
    """
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError(f"Labelled distance body has shape {distance_matrix.shape}; expected a square matrix")
    if len(matrix_ids) != distance_matrix.shape[0]:
        raise ValueError(f"Distance header has {len(matrix_ids)} ids for a body of shape {distance_matrix.shape}")

    index_of = {}
    for pos, raw_id in enumerate(matrix_ids):
        index_of[_bin_id(raw_id)] = pos

    missing = []
    positions = []
    for raw_id in req_ids:
        key = _bin_id(raw_id)
        if key not in index_of:
            missing.append(key)
        else:
            positions.append(index_of[key])
    if missing:
        raise ValueError(f"Distance matrix has no row for bin id(s) {missing}")

    idx = np.asarray(positions, dtype=int)
    return distance_matrix[np.ix_(idx, idx)]
