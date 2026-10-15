"""
Distance Metric interface module.

Attributes:
    IDistanceMetric: Interface for distance metrics

Example:
    >>> from logic.src.interfaces.distance_metric import IDistanceMetric
    >>> class MyDistanceMetric(IDistanceMetric):
    ...     def compute(self, current: Any, candidate: Any) -> float:
    ...         return 0.0
    ...
    >>> distance_metric = MyDistanceMetric()
    >>> distance_metric.compute(None, None)
    0.0
"""

from abc import ABC, abstractmethod
from typing import Any


class IDistanceMetric(ABC):
    """
    Abstract Base Class for structural distance metrics between optimization solutions.
    Used for topography and diversity aware exploration (e.g., Skewed VNS).

    Attributes:
        None: No attributes
    """

    @abstractmethod
    def compute(self, current: Any, candidate: Any) -> float:
        """
        Computes the structural distance between two solutions.

        Args:
            current: The current solution representation (e.g., list of routes).
            candidate: The candidate solution representation.

        Returns:
            float: The scalar structural distance.
        """
        pass
