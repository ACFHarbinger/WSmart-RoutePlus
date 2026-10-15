"""
Configuration for the Multi-Period Simulated Annealing (MP-SA).

Attributes:
    MP_SA_Config: Configuration for the Multi-Period Simulated Annealing (MP-SA) policy.

Example:
    >>> from configs.policies.mp_sa import MP_SA_Config
    >>> config = MP_SA_Config()
    >>> config.iters
    500
    >>> config.init_temp
    100.0
    >>> config.cooling_rate
    0.95
    >>> config.seed
    42
"""

from dataclasses import dataclass
from typing import Optional

from .other.packing import PackingConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class MP_SA_Config:
    """
    Configuration for the Multi-Period Simulated Annealing (MP-SA) policy.

    Attributes:
        iters (int): Number of iterations. Defaults to 500.
        init_temp (float): Initial temperature. Defaults to 100.0.
        cooling_rate (float): Cooling rate. Defaults to 0.95.
        seed (int): Random seed for reproducibility. Defaults to 42.
        packing (Optional[PackingConfig]): Configuration for
            mandatory node selection policies.
        route_improvement (Optional[RouteImprovingConfig]): Optional configuration
            for local search refinement steps.
    """

    iters: int = 500
    init_temp: float = 100.0
    cooling_rate: float = 0.95
    seed: int = 42

    packing: Optional[PackingConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None
