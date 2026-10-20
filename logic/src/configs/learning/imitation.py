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
from typing import Any, Optional, Union

from .experts import (
    ACOConfig,
    ProfitableNNConfig,
    RLSConfig,
    VectorALNSConfig,
    VectorHGSALNSConfig,
    VectorHGSConfig,
    VectorHVPLConfig,
    VectorILSConfig,
)

# Type alias for expert policy configurations
ExpertPolicyConfig = Union[
    VectorHGSConfig,
    VectorALNSConfig,
    VectorILSConfig,
    VectorHVPLConfig,
    RLSConfig,
    ACOConfig,
    VectorHGSALNSConfig,
    ProfitableNNConfig,
]


@dataclass
class ImitationConfig:
    """Imitation learning configuration.

    Attributes:
        policy_config: Configuration for the expert policy to imitate.
                      One of the classes in ``logic.src.configs.learning.experts``.
        loss_fn: Loss function to use ('nll' for negative log-likelihood, 'mse' for mean squared error).
        fill_from: Whose routes empty the bins between time-training days: 'model' (the model's
                   sampled rollout; trained the paper models) or 'expert' (the expert's tours;
                   the model rollout is skipped).
        expert_cache: Directory of the expert solve cache (profitable nearest-neighbour expert
                      only), e.g. 'data/cache/expert/cptp100_emp'; None disables it.
        precompute: Only fill the expert cache, without training the model (needs
                    fill_from='expert' and expert_cache).
    """

    policy_config: Any = None
    loss_fn: str = "nll"
    fill_from: str = "model"
    expert_cache: Optional[str] = None
    precompute: bool = False
