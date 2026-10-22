"""Learning algorithm configuration.

``RLConfig`` is the training-loop configuration. Algorithm dataclasses sit in this
package (imitation experts use the simulator policy configs). The package used
to live at ``logic.src.configs.rl``.

Attributes:
    RLConfig: Configuration for RL algorithms.
    PPOConfig: Configuration for PPO.
    SAPOConfig: Configuration for SAPO.
    GRPOConfig: Configuration for GRPO.
    POMOConfig: Configuration for POMO.
    SymNCOConfig: Configuration for SymNCO.
    ImitationConfig: Configuration for imitation learning.
    GDPOConfig: Configuration for GDPO.
    AdaptiveImitationConfig: Configuration for adaptive imitation.

Example:
    RLConfig(
        algorithm='reinforce',
        baseline='rollout',
        entropy_weight=0.0,
        max_grad_norm=1.0,
    )
"""

from dataclasses import dataclass, field

from .adaptive_imitation import AdaptiveImitationConfig
from .gdpo import GDPOConfig
from .grpo import GRPOConfig
from .imitation import ImitationConfig
from .pomo import POMOConfig
from .ppo import PPOConfig
from .sapo import SAPOConfig
from .symnco import SymNCOConfig


@dataclass
class RLConfig:
    """RL algorithm configuration.

    Attributes:
        algorithm: RL algorithm name ('reinforce', 'ppo', 'sapo', etc.).
        baseline: Baseline type ('rollout', 'critic', 'pomo', 'warmup').
        entropy_weight: Weight for entropy regularization.
        normalize_advantage: Divide the REINFORCE advantage by its batch standard deviation, so
            the entropy weight means the same at any reward scale.
        max_grad_norm: Maximum gradient norm for clipping.
    """

    algorithm: str = "reinforce"
    baseline: str = "rollout"
    bl_warmup_epochs: int = 0
    entropy_weight: float = 0.0
    normalize_advantage: bool = False
    max_grad_norm: float = 1.0
    gamma: float = 0.99
    exp_beta: float = 0.8
    bl_alpha: float = 0.05

    # -------------------------------------------------------------------------
    # Potential-Based Reward Shaping (PBRS)
    # Reference: Ng, Harada & Russell (1999) — ICML
    # -------------------------------------------------------------------------
    # F(s, a, s') = gamma * Phi(s') - Phi(s)
    # R_total = R_base + pbrs_shaping_weight * F
    # gamma is reused from the field above — no separate key needed.
    use_pbrs: bool = False
    """Enable episode-level PBRS shaping (default: False)."""
    pbrs_shaping_weight: float = 1.0
    """Scale factor applied to F before adding to R_base (default: 1.0)."""
    pbrs_potential: str = "ptp"
    """Potential function key. Currently supported: 'ptp'. Others log a warning
    and fall back to zero shaping."""

    # Algorithm specific sub-configs
    ppo: PPOConfig = field(default_factory=PPOConfig)
    sapo: SAPOConfig = field(default_factory=SAPOConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    pomo: POMOConfig = field(default_factory=POMOConfig)
    symnco: SymNCOConfig = field(default_factory=SymNCOConfig)
    imitation: ImitationConfig = field(default_factory=ImitationConfig)
    gdpo: GDPOConfig = field(default_factory=GDPOConfig)
    adaptive_imitation: AdaptiveImitationConfig = field(default_factory=AdaptiveImitationConfig)


__all__ = [
    "RLConfig",
    "PPOConfig",
    "SAPOConfig",
    "GRPOConfig",
    "POMOConfig",
    "SymNCOConfig",
    "ImitationConfig",
    "GDPOConfig",
    "AdaptiveImitationConfig",
]
