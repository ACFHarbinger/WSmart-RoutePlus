"""GDPO specific configuration.

Attributes:
    GDPOConfig: Configuration for GDPO algorithm.

Example:
    gdpo_config = GDPOConfig(
        objective_keys=["reward_waste", "reward_cost"],
        objective_weights=None,
        conditional_key=None,
        renormalize=True,
    )
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GDPOConfig:
    """GDPO specific configuration.

    Attributes:
        objective_keys: List of objective keys to use.
        objective_weights: Weights for the objective functions.
        conditional_key: Conditional key for the objective functions.
        renormalize: Whether to renormalize the objective functions.
    """

    objective_keys: List[str] = field(default_factory=lambda: ["reward_waste", "reward_cost"])
    objective_weights: Optional[List[float]] = None
    conditional_key: Optional[str] = None
    renormalize: bool = True
