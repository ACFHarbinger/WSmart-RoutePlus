"""
Genetic Algorithm with Stochastic Tournament Selection for VRPP.

Provides a memetic algorithm that utilizes tolerance-based selection
mechanisms to solve the Vehicle Routing Problem with Profits.

Attributes:
    MemeticAlgorithmToleranceBasedSelectionConfig: Configuration parameters for MATBS.
    MemeticAlgorithmToleranceBasedSelectionSolver: Core solver implementation for MATBS.

"""

from logic.src.configs.policies.routing.meta_heuristics.ma_ts import MemeticAlgorithmToleranceBasedSelectionConfig

from .solver import MemeticAlgorithmToleranceBasedSelectionSolver

__all__ = ["MemeticAlgorithmToleranceBasedSelectionSolver", "MemeticAlgorithmToleranceBasedSelectionConfig"]
