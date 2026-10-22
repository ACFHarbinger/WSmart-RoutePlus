"""SAPO specific configuration.

Attributes:
    SAPOConfig: Configuration for SAPO algorithm.

Example:
    sapo_config = SAPOConfig(
        tau_pos=0.1,
        tau_neg=1.0,
    )
"""

from dataclasses import dataclass


@dataclass
class SAPOConfig:
    """SAPO specific configuration.

    Attributes:
        tau_pos: Positive learning rate.
        tau_neg: Negative learning rate.
    """

    tau_pos: float = 0.1
    tau_neg: float = 1.0
