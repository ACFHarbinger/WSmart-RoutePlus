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
from typing import Any, List


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


    @classmethod
    def from_config(cls, config: Any) -> "SRCConfig":
        """Create parameters from a configuration object.

        Args:
            config: Configuration object.

        Returns:
            "SRCConfig": Configuration parameters for the Sequential Routing Algorithm.
        """
        return cls(
            routing_algorithms=getattr(
                config, "routing_algorithms", getattr(config, "constructors", ["tsp", "alns"])
            ),
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", 42),
        )
