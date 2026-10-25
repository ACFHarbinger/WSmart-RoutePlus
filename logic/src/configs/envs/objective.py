"""
Objective/Reward configuration module.

Attributes:
    ObjectiveConfig: Objective configuration.

Example:
    >>> from logic.src.configs.envs import ObjectiveConfig
    >>> config = ObjectiveConfig()
    >>> print(config)
    ObjectiveConfig(cost_weight=1.0, waste_weight=1.0, overflow_penalty=1.0)
"""

from dataclasses import dataclass


@dataclass
class ObjectiveConfig:
    """Configuration for problem objectives and reward weights.

    Attributes:
        cost_weight: Weight for length/distance in cost function.
        waste_weight: Weight for waste collection in cost function.
        overflow_penalty: Penalty factor for overflows.
    """

    cost_weight: float = 1.0
    waste_weight: float = 1.0
    overflow_penalty: float = 1.0
