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

from logic.src.configs.policies.routing.meta_heuristics.aco_ks import KSparseACOConfig
from logic.src.configs.policies.routing.meta_heuristics.alns import ALNSConfig
from logic.src.configs.policies.routing.meta_heuristics.hgs import HGSConfig
from logic.src.configs.policies.routing.meta_heuristics.hgs_alns import HGSALNSConfig
from logic.src.configs.policies.routing.meta_heuristics.hvpl import HVPLConfig
from logic.src.configs.policies.routing.meta_heuristics.ils import ILSConfig
from logic.src.configs.policies.route_improvement import RandomLocalSearchPostConfig
from logic.src.configs.policies.vectorized.routing.profitable_nn import ProfitableNNConfig

# Type alias for expert policy configurations: the simulator policy configs of the
# vectorised experts (``logic.src.policies.vectorized.routing``).
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
class ImitationConfig:
    """Imitation learning configuration.

    Attributes:
        policy_config: Configuration for the expert policy to imitate.
                      A simulator policy config (``ExpertPolicyConfig``).
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
