"""
Configuration for the Multi-Period Iterated Local Search (MP-ILS).

Attributes:
    MP_ILS_Config: Configuration for the Multi-Period Iterated Local Search (MP-ILS) policy.

Example:
    >>> from configs.policies.mp_ils import MP_ILS_Config
    >>> config = MP_ILS_Config()
    >>> config.iters
    50
    >>> config.perturb_size
    3
    >>> config.seed
    42
    >>> config.vrpp
    True
"""

from dataclasses import dataclass
from typing import Optional

from .other.packing import PackingConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class MP_ILS_Config:
    """
    Configuration for the Multi-Period Iterated Local Search (MP-ILS) policy.

    Attributes:
        iters (int): Number of iterations. Defaults to 50.
        perturb_size (int): Number of nodes to remove during perturbation. Defaults to 3.
        seed (int): Random seed for reproducibility. Defaults to 42.
        vrpp (bool): Whether the problem is a VRP with Profits. Defaults to True.
        packing (Optional[PackingConfig]): Configuration for
            mandatory node selection policies.
        route_improvement (Optional[RouteImprovingConfig]): Optional configuration
            for local search refinement steps.
    """

    iters: int = 50
    perturb_size: int = 3
    seed: int = 42
    vrpp: bool = True

    packing: Optional[PackingConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None
