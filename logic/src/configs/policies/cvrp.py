"""
CVRP (Capacitated Vehicle Routing Problem) policy configuration.

Attributes:
    CVRPConfig: Attributes for CVRP configuration.

Example:
    >>> from configs.policies.cvrp import CVRPConfig
    >>> config = CVRPConfig()
    >>> config.time_limit
    60.0
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.packing import PackingConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class CVRPConfig:
    """Configuration for Capacitated Vehicle Routing Problem (CVRP) policy.

    Attributes:
        cache: Whether to cache solutions.
        time_limit: Maximum time in seconds for the solver.
        seed: Random seed for reproducibility.
        engine: Solver engine to use ('ortools', 'gurobi').
        packing: List of packing algorithm config files.
        route_improvement: List of route improvement operations to apply.
        solution_limit: Deterministic OR-Tools budget (WP-12 gate-(a)).
            None (default) keeps the time_limit behaviour unchanged.
    """

    cache: bool = False
    time_limit: float = 60.0
    engine: str = "ortools"
    seed: Optional[int] = None
    packing: Optional[List[PackingConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
    solution_limit: Optional[int] = None
