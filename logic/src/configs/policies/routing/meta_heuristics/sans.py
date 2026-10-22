"""
SANS (Simulated Annealing Neighborhood Search) configuration.

Attributes:
    SANSConfig: Configuration for the Simulated Annealing Neighborhood Search (SANS) policy.

Example:
    >>> from configs.policies.sans import SANSConfig
    >>> config = SANSConfig()
    >>> config.engine
    'new'
    >>> config.time_limit
    60.0
    >>> config.seed
    42
    >>> config.perc_bins_can_overflow
    0.0
    >>> config.T_min
    0.01
    >>> config.T_init
    75.0
    >>> config.iterations_per_T
    5000
    >>> config.alpha
    0.95
    >>> config.combination
    None
    >>> config.packing
    None
    >>> config.route_improvement
    None
"""

from dataclasses import dataclass
from typing import List, Literal, Optional

from logic.src.configs.policies.packing.dispatch import PackingConfig

from logic.src.configs.policies.route_improvement import RouteImprovingConfig


@dataclass
class SANSConfig:
    """Configuration for Simulated Annealing Neighborhood Search (SANS) policy.

    Attributes:
        engine: Engine type - 'new' for improved SA, 'og' for original LAC.
        time_limit: Maximum time in seconds for the solver.
        perc_bins_can_overflow: Percentage of bins allowed to overflow.
        T_min: Minimum temperature for simulated annealing.
        T_init: Initial temperature for simulated annealing.
        iterations_per_T: Number of iterations at each temperature level.
        alpha: Cooling rate (temperature multiplier).
        max_iterations: Total neighbour moves of the 'new' engine before it stops (0 = no cap).
            With a cap the run is reproducible: the stop no longer depends on machine speed;
            time_limit stays a guard.
        combination: Seven numbers read by position by the 'og' engine (iterations, initial
            temperature, cooling factor, vehicle, load, route-difference and shift penalties).
        packing: List of packing algorithm config files.
        route_improvement: List of route improvement operations to apply.
    """

    engine: Literal["new", "og"] = "new"
    time_limit: float = 60.0
    seed: Optional[int] = None
    perc_bins_can_overflow: float = 0.0
    T_min: float = 0.01
    T_init: float = 75.0
    iterations_per_T: int = 5000
    alpha: float = 0.95
    max_iterations: int = 0
    combination: Optional[List[float]] = None
    shift_duration: float = 28800.0  # 8 hours
    packing: Optional[List[PackingConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
