"""
Memetic Island Model Genetic Algorithm for VRPP.

Uses an island model architecture with memetic local search to solve the
Vehicle Routing Problem with Profits.

Attributes:
    MemeticAlgorithmIslandModelParams: Configuration parameters for MAIM.
    MemeticAlgorithmIslandModelSolver: Core solver implementation for MAIM.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.memetic_algorithm_island_model import MemeticAlgorithmIslandModelSolver
"""

from .params import MemeticAlgorithmIslandModelParams
from .solver import MemeticAlgorithmIslandModelSolver

__all__ = ["MemeticAlgorithmIslandModelSolver", "MemeticAlgorithmIslandModelParams"]
