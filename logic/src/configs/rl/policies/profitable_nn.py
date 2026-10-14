"""
Profitable nearest-neighbour expert configuration.

Attributes:
    ProfitableNNConfig: Configuration for the selective nearest-neighbour expert.

Example:
    profitable_nn_config = ProfitableNNConfig(two_opt_passes=5)
"""

from dataclasses import dataclass


@dataclass
class ProfitableNNConfig:
    """Configuration for the profitable nearest-neighbour expert.

    Attributes:
        two_opt_passes: Maximum number of full 2-opt sweeps per tour.
    """

    two_opt_passes: int = 5
