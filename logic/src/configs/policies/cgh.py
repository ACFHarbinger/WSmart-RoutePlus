"""
Configuration for the Column Generation Heuristic (CGH).

Attributes:
    CGHConfig: Attributes for CGH configuration.

Example:
    >>> from configs.policies.cgh import CGHConfig
    >>> config = CGHConfig()
    >>> config.cg_iters
    10
"""

from dataclasses import dataclass
from typing import Optional

from .other.packing import PackingConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class CGHConfig:
    """
    Configuration for the Column Generation Heuristic (CGH) policy.

    CGH uses a master problem (set packing/partitioning) and generates routes
    heuristically. Instead of solving the exact subproblem (ESPPRC), it uses
    randomized greedy and local search to find negative reduced-cost columns.

    Attributes:
        cg_iters (int): Number of column generation iterations. Defaults to 10.
        routes_per_iter (int): Number of heuristic routes to generate in each
            column generation iteration. Defaults to 50.
        seed (int): Random seed for reproducibility. Defaults to 42.
        packing (Optional[PackingConfig]): Configuration for
            mandatory node selection policies.
        route_improvement (Optional[RouteImprovingConfig]): Optional configuration
            for local search refinement steps.
    """

    cg_iters: int = 10
    routes_per_iter: int = 50
    seed: int = 42

    packing: Optional[PackingConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None
