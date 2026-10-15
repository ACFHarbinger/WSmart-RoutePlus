"""
RL-ALNS (Reinforcement Learning-augmented Adaptive Large Neighborhood Search) configuration.

Attributes:
    RLALNSConfig: Configuration for the RL-ALNS policy.

Example:
    >>> from configs.policies.rl_alns import RLALNSConfig
    >>> config = RLALNSConfig()
    >>> config.max_iterations
    1000
    >>> config.profit_aware_operators
    False
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from logic.src.configs.policies.other import RLConfig

from .alns import ALNSConfig


@dataclass
class RLALNSConfig(ALNSConfig):
    """Configuration for Reinforcement Learning-augmented ALNS (RL-ALNS) policy.

    Attributes:
        rl_config: Centralized Reinforcement Learning configuration.
    """

    rl_config: RLConfig = field(default_factory=RLConfig)
    profit_aware_operators: bool = False
    time_limit: float = 60.0
    max_iterations: int = 5000
    start_temp: float = 100.0
    cooling_rate: float = 0.995
    min_removal: int = 1
    max_removal_pct: float = 0.3
    seed: Optional[int] = None
