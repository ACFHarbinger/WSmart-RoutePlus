"""RL algorithm configuration.

Attributes:
    RLConfig: Configuration for RL algorithms.

Example:
    RLConfig(
        algorithm='reinforce',
        baseline='rollout',
        entropy_weight=0.0,
        max_grad_norm=1.0,
    )
"""

from dataclasses import dataclass, field

from .core.adaptive_imitation import AdaptiveImitationConfig
from .core.gdpo import GDPOConfig
from .core.grpo import GRPOConfig
from .core.imitation import ImitationConfig
from .core.pomo import POMOConfig
from .core.ppo import PPOConfig
from .core.sapo import SAPOConfig
from .core.symnco import SymNCOConfig


@dataclass
class RLConfig:
    """RL algorithm configuration.

    Attributes:
        algorithm: RL algorithm name ('reinforce', 'ppo', 'sapo', etc.).
        baseline: Baseline type ('rollout', 'critic', 'pomo', 'warmup').
        entropy_weight: Weight for entropy regularization.
        max_grad_norm: Maximum gradient norm for clipping.
    """

    algorithm: str = "reinforce"
    baseline: str = "rollout"
    bl_warmup_epochs: int = 0
    entropy_weight: float = 0.0
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
