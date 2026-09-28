"""
Meta-RL Config module.

Attributes:
    MetaRLConfig: Configuration for meta-reinforcement learning pipeline.

Example:
    meta_rl_config = MetaRLConfig(
        use_meta=True,
        meta_strategy="rnn",
        meta_lr=1e-3,
        meta_hidden_dim=64,
        meta_history_length=10,
        graph=GraphConfig(),
        reward=ObjectiveConfig(),
    )
"""

from dataclasses import dataclass, field
from typing import Any, List

from logic.src.configs.envs.env import EnvConfig


@dataclass
class MetaRLConfig:
    """Meta-RL and HRL algorithm configuration.

    Attributes:
        use_meta: Whether to use meta-learning wrapper.
        meta_strategy: Meta-learning strategy ('rnn', 'bandit', 'morl', etc.).
        meta_lr: Learning rate for meta-optimizer.
        meta_hidden_dim: Hidden dimension for meta-network.
        meta_history_length: History length for meta-learning.
        graph: Graph configuration.
        reward: Objective/reward configuration.
    """

    # Meta-RL
    use_meta: bool = False
    meta_strategy: str = "rnn"  # rnn|bandit|morl|tdl|hypernet|hrl
    meta_lr: float = 1e-3
    meta_hidden_dim: int = 64
    meta_history_length: int = 10

    # HRL
    shared_encoder: bool = True
    lr_critic_value: float = 1e-4

    env: Any = field(default_factory=EnvConfig)
