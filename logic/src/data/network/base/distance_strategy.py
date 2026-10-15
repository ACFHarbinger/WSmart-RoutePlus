"""distance_strategy.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import pandas as pd


class DistanceStrategy(ABC):
    """
    Abstract base class for distance matrix calculation strategies.

    Defines the interface for computing pairwise distances between
    geographic coordinates. Each strategy implements a different
    calculation method (API-based, formula-based, or network-based).

    Attributes:
        None
    """

    @abstractmethod
    def calculate(self, coords: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """
        Computes pairwise distance matrix for given coordinates.

        Args:
            coords: DataFrame with coordinates (must contain 'Lat', 'Lng' columns).
            kwargs: Additional arguments for the distance strategy.

        Returns:
            np.ndarray: n×n symmetric distance matrix in kilometers

        """
        pass

    def _eval_kwarg(self, kwarg: str, kwargs: Dict[str, Any]) -> bool:
        """Check if keyword argument exists and is not None.

        Args:
            kwarg: Keyword argument to check.
            kwargs: Dictionary of keyword arguments.

        Returns:
            bool: True if kwarg exists and is not None, False otherwise
        """
        return bool(kwarg in kwargs and kwargs[kwarg] is not None)
