"""Profitable nearest-neighbour expert configuration.

The profitable nearest-neighbour heuristic exists only as a vectorised imitation
expert (``logic.src.policies.vectorized.routing.profitable_nearest_neighbour``); it has no
simulator counterpart, so this is its only configuration.

Attributes:
    ProfitableNNConfig: Configuration for the profitable nearest-neighbour expert.

Example:
    >>> ProfitableNNConfig(two_opt_passes=5)
"""

from dataclasses import dataclass


@dataclass
class ProfitableNNConfig:
    """Configuration for the profitable nearest-neighbour expert.

    Attributes:
        two_opt_passes: Maximum number of full 2-opt sweeps per tour.
    """

    two_opt_passes: int = 5
