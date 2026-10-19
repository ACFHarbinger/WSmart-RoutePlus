"""
Base environment definitions.

Attributes:
    RL4COEnvBase: Base class for RL4CO environments.
    ImprovementEnvBase: Base class for improvement environments.

Example:
    None
"""

from .base import RL4COEnvBase
from .improvement import ImprovementEnvBase

__all__ = ["RL4COEnvBase", "ImprovementEnvBase"]
