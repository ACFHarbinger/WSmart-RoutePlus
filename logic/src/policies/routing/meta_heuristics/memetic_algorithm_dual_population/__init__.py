"""
Memetic Algorithm with Dual Population (MA-DP) for VRPP.

Maintains dual population structure (active + reserve) to balance
intensification and diversification.

Attributes:
    MemeticAlgorithmDualPopulationParams: Configuration parameters for MA-DP.
    MemeticAlgorithmDualPopulationPolicy: Policy adapter for MA-DP.
    MemeticAlgorithmDualPopulationSolver: Core solver logic for MA-DP.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.memetic_algorithm_dual_population import MemeticAlgorithmDualPopulationPolicy
"""

from .params import MemeticAlgorithmDualPopulationParams
from .policy_ma_dp import MemeticAlgorithmDualPopulationPolicy
from .solver import MemeticAlgorithmDualPopulationSolver

__all__ = [
    "MemeticAlgorithmDualPopulationSolver",
    "MemeticAlgorithmDualPopulationParams",
    "MemeticAlgorithmDualPopulationPolicy",
]
