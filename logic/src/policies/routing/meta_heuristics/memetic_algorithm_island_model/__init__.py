"""
Memetic Island Model Genetic Algorithm for VRPP.

Uses an island model architecture with memetic local search to solve the
Vehicle Routing Problem with Profits.

Attributes:
    MemeticAlgorithmIslandModelConfig: Configuration parameters for MAIM.
    MemeticAlgorithmIslandModelSolver: Core solver implementation for MAIM.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.memetic_algorithm_island_model import MemeticAlgorithmIslandModelSolver
"""

from logic.src.configs.policies.ma_im import MemeticAlgorithmIslandModelConfig
from .solver import MemeticAlgorithmIslandModelSolver

__all__ = ["MemeticAlgorithmIslandModelSolver", "MemeticAlgorithmIslandModelConfig"]
