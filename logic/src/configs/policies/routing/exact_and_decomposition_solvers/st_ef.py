"""
Configuration for the Scenario-Tree Extensive Form policy.

Attributes:
    ScenarioTreeExtensiveFormConfig: Configuration for the Scenario-Tree Extensive Form (ST-EF) policy.

Example:
    >>> from configs.policies.st_ef import ScenarioTreeExtensiveFormConfig
    >>> config = ScenarioTreeExtensiveFormConfig()
    >>> config.num_days
    3
    >>> config.num_realizations
    3
    >>> config.mean_increment
    0.2
    >>> config.time_limit
    300.0
    >>> config.mip_gap
    0.05
    >>> config.waste_weight
    1.0
    >>> config.cost_weight
    1.0
    >>> config.overflow_penalty
    10.0
    >>> config.discount_factor
    0.95
    >>> config.use_mtz
    True
    >>> config.seed
    42
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ScenarioTreeExtensiveFormConfig:
    """
    Configuration for the Scenario-Tree Extensive Form policy.

    Attributes:
        num_days: Planning horizon in days.
        num_realizations: Number of discrete joint realizations per day.
        mean_increment: Mean fractional increment per day (0.0 to 1.0).
        time_limit: Maximum solve time in seconds.
        mip_gap: Target optimality gap (e.g., 0.05 for 5%).
        waste_weight: Weight for the waste component in the objective.
        cost_weight: Weight for the cost component in the objective.
        overflow_penalty: Penalty cost for overflowing capacity.
        discount_factor: Discount factor for future expected rewards.
        use_mtz: Whether to use MTZ constraints (True) or lazy SEC separation (False).
        seed: Random seed for scenario generation.
    """

    # Planning horizon (number of days to look ahead)
    num_days: int = 3

    # Number of discrete joint realizations per day (branching factor)
    num_realizations: int = 3

    # Mean fractional increment per day (0.0 to 1.0)
    # This guides the distribution used for scenario generation.
    mean_increment: float = 0.2

    # Maximum solve time for the full tree MILP in seconds
    time_limit: float = 300.0

    # Target optimality gap (0.01 = 1%)
    mip_gap: float = 0.05

    # Weights for the objective function (matches sim loop defaults)
    waste_weight: float = 1.0
    cost_weight: float = 1.0
    overflow_penalty: float = 10.0

    # Discount factor for future expected rewards
    discount_factor: float = 0.95

    # Whether to use MTZ constraints (standard) or lazy SEC separation
    # Note: Lazy constraints are generally slower for multi-scenario models
    # if the number of nodes is very small.
    use_mtz: bool = True

    # Seed for scenario tree generation reproducibility
    seed: Optional[int] = 42

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.num_days < 1:
            raise ValueError("num_days must be at least 1")
        if self.num_realizations < 1:
            raise ValueError("num_realizations must be at least 1")
