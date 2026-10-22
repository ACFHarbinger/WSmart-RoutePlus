"""SymNCO specific configuration.

Attributes:
    SymNCOConfig: Configuration for SymNCO algorithm.

Example:
    symnco_config = SymNCOConfig(
        alpha=0.2,
        beta=1.0,
    )
"""

from dataclasses import dataclass


@dataclass
class SymNCOConfig:
    """SymNCO specific configuration.

    Attributes:
        alpha: Alpha parameter for the algorithm.
        beta: Beta parameter for the algorithm.
    """

    alpha: float = 0.2
    beta: float = 1.0
