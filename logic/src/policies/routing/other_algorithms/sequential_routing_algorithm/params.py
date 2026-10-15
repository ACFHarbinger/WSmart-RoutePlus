"""
Configuration parameters for the Sequential Routing Algorithm (SRC).

Attributes:
    SRCParams: Configuration parameters for the Sequential Routing Algorithm (SRC).

Example:
    >>> from logic.src.policies.routing.other_algorithms.sequential_routing_algorithm import SRCParams
    >>> params = SRCParams()
    >>> params.routing_algorithms
    ['tsp', 'alns']
    >>> params.time_limit
    60.0
    >>> params.seed
    42
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List


@dataclass
class SRCParams:
    """
    Configuration parameters for the Sequential Routing Algorithm.

    Attributes:
        constructors: List of routing algorithm names to run in sequence.
        time_limit: Maximum wall-clock time (seconds) for the entire sequence.
        seed: Random seed for deterministic execution.
    """

    routing_algorithms: List[str]
    time_limit: float = 60.0
    seed: int = 42

    @classmethod
    def from_config(cls, config: Any) -> SRCParams:
        """Create parameters from a configuration object.

        Args:
            config: Configuration object.

        Returns:
            SRCParams: Configuration parameters for the Sequential Routing Algorithm.
        """
        return cls(
            routing_algorithms=getattr(
                config, "routing_algorithms", getattr(config, "constructors", ["tsp", "alns"])
            ),
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", 42),
        )
