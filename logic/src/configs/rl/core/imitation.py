"""Imitation specific configuration.

Attributes:
    ImitationConfig: Configuration for Imitation algorithm.

Example:
    imitation_config = ImitationConfig(
        policy_config=None,
        loss_fn="nll",
    )
"""

from dataclasses import dataclass
from typing import Any, Union

from logic.src.configs.rl.policies import (
    ACOConfig,
    ProfitableNNConfig,
    RLSConfig,
    VectorALNSConfig,
    VectorHGSALNSConfig,
    VectorHGSConfig,
    VectorILSConfig,
)

# Type alias for expert policy configurations
ExpertPolicyConfig = Union[
    VectorHGSConfig, VectorALNSConfig, VectorILSConfig, RLSConfig, ACOConfig, VectorHGSALNSConfig, ProfitableNNConfig
]


@dataclass
class ImitationConfig:
    """Imitation learning configuration.

    Attributes:
        policy_config: Configuration for the expert policy to imitate.
                      Can be any of: HGSConfig, ALNSConfig, ILSConfig, RLSConfig, ACOConfig, HGSALNSConfig.
        loss_fn: Loss function to use ('nll' for negative log-likelihood, 'mse' for mean squared error).
    """

    policy_config: Any = None
    loss_fn: str = "nll"
