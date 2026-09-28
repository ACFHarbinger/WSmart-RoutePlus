"""GRPO specific configuration.

Attributes:
    GRPOConfig: Configuration for GRPO algorithm.

Example:
    grpo_config = GRPOConfig(
        epsilon=0.2,
        epochs=3,
    )
"""

from dataclasses import dataclass


@dataclass
class GRPOConfig:
    """GRPO specific configuration.

    Attributes:
        epsilon: Epsilon value for the algorithm.
        epochs: Number of epochs to train.
    """

    epsilon: float = 0.2
    epochs: int = 3
