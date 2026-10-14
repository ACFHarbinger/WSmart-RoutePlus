"""
Genetic Algorithm with Stochastic Tournament Selection for VRPP.

Provides a memetic algorithm that utilizes tolerance-based selection
mechanisms to solve the Vehicle Routing Problem with Profits.

Attributes:
    MemeticAlgorithmToleranceBasedSelectionParams: Configuration parameters for MATBS.
    MemeticAlgorithmToleranceBasedSelectionSolver: Core solver implementation for MATBS.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.memetic_algorithm_tolerance_based_selection import MemeticAlgorithmToleranceBasedSelectionSolver
"""

from .params import MemeticAlgorithmToleranceBasedSelectionParams
from .solver import MemeticAlgorithmToleranceBasedSelectionSolver

__all__ = ["MemeticAlgorithmToleranceBasedSelectionSolver", "MemeticAlgorithmToleranceBasedSelectionParams"]
