"""
GENIUS (GENI + US) configuration for Hydra.

Reference:
    Gendreau, M., Hertz, A., & Laporte, G. (1992).
    "New Insertion and Postoptimization Procedures for the Traveling Salesman Problem"
    Operations Research, 40(6), 1086-1094.

Attributes:
    GENIUSConfig: Configuration for the GENIUS (GENI + US) policy.

Example:
    >>> from configs.policies.genius import GENIUSConfig
    >>> config = GENIUSConfig()
    >>> config.time_limit
    60.0
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class GENIUSConfig:
    """Configuration for the GENIUS (GENI + US) policy.

    Attributes:
        neighborhood_size (int): Neighborhood size.
        unstring_type (int): Type of unstring.
        string_type (int): Type of string.
        n_iterations (int): Number of iterations.
        random_us_sampling (bool): Whether to use random US sampling.
        time_limit (float): Time limit in seconds.
        seed (Optional[int]): Seed for the random number generator.
        packing (Optional[List[PackingConfig]]): Packing configurations.
        route_improvement (Optional[List[RouteImprovingConfig]]): Route improvement configurations.
    """

    neighborhood_size: int = 5
    unstring_type: int = 1
    string_type: int = 1
    n_iterations: int = 1
    random_us_sampling: bool = False
    time_limit: float = 60.0
    seed: Optional[int] = None
    packing: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
    profit_aware_operators: bool = False
