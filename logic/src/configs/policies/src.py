"""
SRC (Sequential Route Constructor) configuration dataclass.

Attributes:
    constructors: List of route constructor names to run in sequence.
    time_limit: Maximum wall-clock time (seconds) for the entire sequence.
    seed: Random seed for reproducibility.

Example:
    >>> from configs.policies.src import SRCConfig
    >>> config = SRCConfig()
    >>> config.constructors
    ['tsp', 'alns']
    >>> config.time_limit
    60.0
    >>> config.seed
    42
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SRCConfig:
    """Configuration for the Sequential Route Constructor (SRC).

    Attributes:
        constructors: List of route constructor names to run in sequence.
        time_limit: Maximum wall-clock time (seconds) for the entire sequence.
    """

    constructors: List[str] = field(default_factory=lambda: ["tsp", "alns"])
    time_limit: float = 60.0
    seed: int = 42
