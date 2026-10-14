"""
Configuration for the Multi-Period Particle Swarm Optimization (MP-PSO).

Attributes:
    MP_PSO_Config: Configuration for the Multi-Period Particle Swarm Optimization (MP-PSO) policy.

Example:
    >>> from configs.policies.mp_pso import MP_PSO_Config
    >>> config = MP_PSO_Config()
    >>> config.swarm_size
    20
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
class MP_PSO_Config:
    """
    Configuration for the Multi-Period Particle Swarm Optimization (MP-PSO) policy.

    Attributes:
        swarm_size (int): Number of particles in the swarm. Defaults to 20.
        iters (int): Number of iterations. Defaults to 50.
        w (float): Inertia weight. Defaults to 0.8.
        c1 (float): Cognitive coefficient. Defaults to 2.0.
        c2 (float): Social coefficient. Defaults to 2.0.
        seed (int): Random seed for reproducibility. Defaults to 42.
        vrpp (bool): Whether the problem is a VRP with Profits. Defaults to True.
        packing (Optional[PackingConfig]): Configuration for
            mandatory node selection policies.
        route_improvement (Optional[RouteImprovingConfig]): Optional configuration
            for local search refinement steps.
    """

    swarm_size: int = 20
    iters: int = 50
    w: float = 0.8
    c1: float = 2.0
    c2: float = 2.0
    seed: int = 42
    vrpp: bool = True

    packing: Optional[PackingConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None
