"""
Memetic Algorithm with Dual Population (MA-DP) for VRPP.

Maintains dual population structure (active + reserve) to balance
intensification and diversification.

Attributes:
    MemeticAlgorithmDualPopulationConfig: Configuration parameters for MA-DP.
    MemeticAlgorithmDualPopulationPolicy: Policy adapter for MA-DP.
    MemeticAlgorithmDualPopulationSolver: Core solver logic for MA-DP.

"""

from logic.src.configs.policies.routing.meta_heuristics.ma_dp import MemeticAlgorithmDualPopulationConfig

from .policy_ma_dp import MemeticAlgorithmDualPopulationPolicy
from .solver import MemeticAlgorithmDualPopulationSolver

__all__ = [
    "MemeticAlgorithmDualPopulationSolver",
    "MemeticAlgorithmDualPopulationConfig",
    "MemeticAlgorithmDualPopulationPolicy",
]
