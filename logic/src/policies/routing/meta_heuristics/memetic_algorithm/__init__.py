"""Memetic Algorithm (MA) implementation for VRPP.

Attributes:
    MASolver: Main solver class for the Memetic Algorithm.
    MAPolicy: Policy class for the Memetic Algorithm.
    MAConfig: Parameter dataclass for the Memetic Algorithm.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.memetic_algorithm import MAPolicy
    >>> policy = MAPolicy()
"""

from logic.src.configs.policies.ma import MAConfig

from .policy_ma import MAPolicy
from .solver import MASolver

__all__ = [
    "MASolver",
    "MAConfig",
    "MAPolicy",
]
