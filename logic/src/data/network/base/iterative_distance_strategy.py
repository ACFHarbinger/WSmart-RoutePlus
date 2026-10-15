"""iterative_distance_streategy.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

"""

from abc import abstractmethod
from typing import Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .distance_strategy import DistanceStrategy


class IterativeDistanceStrategy(DistanceStrategy):
    """Base class for strategies that compute distances pairwise iteratively.

    Attributes:
        None
    """

    @abstractmethod
    def calculate_pair(self, coords_i: Tuple[float, float], coords_j: Tuple[float, float]) -> float:
        """
        Computes distance between a single pair of coordinates.
        Args:
            coords_i: Coordinates of the first point (lat, lng).
            coords_j: Coordinates of the second point (lat, lng).
        """
        pass

    def calculate(self, coords: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """
        Iterates through all pairs to compute a full distance matrix.
        Args:
            coords: DataFrame with coordinates (must contain 'Lat', 'Lng' columns).
            kwargs: Additional arguments for the distance strategy.

        Returns:
            Distance matrix as a NumPy array.
        """
        size = len(coords)
        distance_matrix = np.zeros((size, size))

        verbose = kwargs.get("verbose", False)
        rows = list(coords.iterrows())
        for id_i, row_i in tqdm(rows, total=size, desc="Outer Loop", disable=not verbose):
            coords_i = (row_i["Lat"], row_i["Lng"])
            for id_j, row_j in tqdm(rows, total=size, desc="Inner Loop", leave=False, disable=not verbose):
                if id_i != id_j:
                    coords_j = (row_j["Lat"], row_j["Lng"])
                    distance_matrix[id_i, id_j] = self.calculate_pair(coords_i, coords_j)

        return distance_matrix
