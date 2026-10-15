"""
SRC (Sequential Routing Algorithm) configuration dataclass.

Attributes:
    constructors: List of routing algorithm names to run in sequence.
    time_limit: Maximum wall-clock time (seconds) for the entire sequence.
    seed: Random seed for reproducibility.

Example:
    >>> from configs.policies.src import SRCConfig
    >>> config = SRCConfig()
    >>> config.routing_algorithms
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
    """Configuration for the Sequential Routing Algorithm (SRC).

    Attributes:
        constructors: List of routing algorithm names to run in sequence.
        time_limit: Maximum wall-clock time (seconds) for the entire sequence.
    """

    routing_algorithms: List[str] = field(default_factory=lambda: ["tsp", "alns"])
    time_limit: float = 60.0
    seed: int = 42
    vrpp: bool = True
