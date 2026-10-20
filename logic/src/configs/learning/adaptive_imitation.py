"""Adaptive Imitation Learning specific configuration.

Attributes:
    AdaptiveImitationConfig: Configuration for Adaptive Imitation Learning.

Example:
    adaptive_imitation_config = AdaptiveImitationConfig(
        il_weight=1.0,
        il_decay=0.95,
        patience=5,
        threshold=0.05,
        decay_step=1,
        epsilon=1e-5,
        policy_config=None,
        loss_fn="nll",
    )
"""

from dataclasses import dataclass
from typing import Any, Union

from logic.src.configs.policies.aco_ks import KSparseACOConfig
from logic.src.configs.policies.alns import ALNSConfig
from logic.src.configs.policies.hgs import HGSConfig
from logic.src.configs.policies.hgs_alns import HGSALNSConfig
from logic.src.configs.policies.hvpl import HVPLConfig
from logic.src.configs.policies.ils import ILSConfig
from logic.src.configs.policies.other.route_improvement import RandomLocalSearchPostConfig
from logic.src.configs.policies.profitable_nn import ProfitableNNConfig

# Type alias for expert policy configurations: the simulator policy configs of the
# vectorised experts (``logic.src.policies.vector``).
ExpertPolicyConfig = Union[
    HGSConfig,
    ALNSConfig,
    ILSConfig,
    HVPLConfig,
    RandomLocalSearchPostConfig,
    KSparseACOConfig,
    HGSALNSConfig,
    ProfitableNNConfig,
]


@dataclass
class AdaptiveImitationConfig:
    """Adaptive Imitation Learning specific configuration.

    Attributes:
        il_weight: Weight for the imitation learning loss.
        il_decay: Decay factor for the imitation learning weight.
        patience: Number of epochs to wait for improvement before stopping.
        threshold: Threshold for improvement to continue training.
        decay_step: Step size for the decay of the imitation learning weight.
        epsilon: Small value to prevent division by zero.
        policy_config: Configuration for the expert policy.
        loss_fn: Loss function to use for training.
    """

    il_weight: float = 1.0
    il_decay: float = 0.95
    patience: int = 5
    threshold: float = 0.05
    decay_step: int = 1
    epsilon: float = 1e-5
    policy_config: Any = None
    loss_fn: str = "nll"
