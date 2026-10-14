"""
Configuration for the Multi-Period Ant Colony Optimization (MP-ACO).

Attributes:
    MP_ACO_Config: Configuration for the Multi-Period Ant Colony Optimization (MP-ACO) policy.

Example:
    >>> from configs.policies.mp_aco import MP_ACO_Config
    >>> config = MP_ACO_Config()
    >>> config.iters
    50
    >>> config.vrpp
    True
"""

from dataclasses import dataclass
from typing import Optional

from .other.packing import PackingConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class MP_ACO_Config:
    """
    Configuration for the Multi-Period Ant Colony Optimization (MP-ACO) policy.

    Attributes:
        n_ants (int): Number of ants in the colony. Defaults to 10.
        iters (int): Number of iterations. Defaults to 50.
        alpha (float): Influence of pheromone on path selection. Defaults to 1.0.
        beta (float): Influence of heuristic information on path selection. Defaults to 2.0.
        rho (float): Pheromone evaporation rate. Defaults to 0.1.
        seed (int): Random seed for reproducibility. Defaults to 42.
        vrpp (bool): Whether the problem is a VRP with Profits. Defaults to True.
        packing (Optional[PackingConfig]): Configuration for
            mandatory node selection policies.
        route_improvement (Optional[RouteImprovingConfig]): Optional configuration
            for local search refinement steps.
    """

    n_ants: int = 10
    iters: int = 50
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.1
    seed: int = 42
    vrpp: bool = True

    packing: Optional[PackingConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None
