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

        # Load matrix, skipping first row and column (IDs)
        distance_matrix = np.loadtxt(matrix_path, delimiter=",")[1:, 1:]

        req_ids = coords["ID"].to_numpy()

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
