"""
TSP (Traveling Salesman Problem) policy configuration.

Attributes:
    TSPConfig: Configuration for the Traveling Salesman Problem (TSP) policy.

Example:
    >>> from configs.policies.tsp import TSPConfig
    >>> config = TSPConfig()
    >>> config.cache
    True
    >>> config.time_limit
    60.0
    >>> config.packing
    None
    >>> config.route_improvement
    None
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.packing import PackingConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class TSPConfig:
    """Configuration for Traveling Salesman Problem (TSP) policy.

    Attributes:
        cache: Whether to cache solutions.
        time_limit: Maximum time in seconds for the solver.
        packing: List of packing algorithm config files.
        route_improvement: List of route improvement operations to apply.
    """

    cache: bool = True
    engine: str = "fast_tsp"
    time_limit: float = 60.0
    seed: Optional[int] = None
    packing: Optional[List[PackingConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
