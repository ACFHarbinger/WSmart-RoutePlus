"""
Joint Greedy Orienteering Configuration for Hydra.

Attributes:
    JointGreedyConfig: Configuration for the Joint Greedy Orienteering policy.

Example:
    >>> from configs.policies.jgo import JointGreedyConfig
    >>> config = JointGreedyConfig()
    >>> config.k_best
    3
    >>> config.time_limit
    30.0
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class JointGreedyConfig:
    """
    Hydra configuration for the Joint Greedy Orienteering policy.

    Attributes:
        k_best (int): Number of best customers to consider.
        n_starts (int): Number of starting points.
        distance_weight (float): Weight for distance.
        seed (Optional[int]): Seed for random number generator.
        time_limit (float): Time limit in seconds.
        packing (Optional[List[Any]]): Mandatory customers/requests.
        route_improvement (Optional[List[Any]]): Route improvement strategies.
    """

    k_best: int = 3
    n_starts: int = 10
    distance_weight: float = 1.0
    seed: Optional[int] = 42
    time_limit: float = 30.0

    # Hydra compatibility placeholders
    packing: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
