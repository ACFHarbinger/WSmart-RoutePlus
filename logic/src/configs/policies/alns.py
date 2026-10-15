"""
ALNS (Adaptive Large Neighborhood Search) configuration.

Attributes:
    ALNSConfig: Attributes for ALNS configuration.

Example:
    >>> from configs.policies.alns import ALNSConfig
    >>> config = ALNSConfig()
    >>> config.time_limit
    60.0
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .other.acceptance_criteria import AcceptanceConfig, BoltzmannAcceptanceConfig
from .other.packing import PackingConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class ALNSConfig:
    """Configuration for Adaptive Large Neighborhood Search (ALNS) policy.

    Attributes:
        time_limit: Maximum time in seconds for the solver.
        max_iterations: Maximum number of ALNS iterations.
        start_temp: Initial temperature for simulated annealing acceptance.
        cooling_rate: Rate at which temperature decreases each iteration.
        reaction_factor: Weight update reaction factor for operator scores.
        start_temp_control: 'w' parameter: accept a 5% worse solution with 0.5 probability
        xi: 'xi' parameter: fraction of total nodes for max removal cap
        segment_size: Iterations before updating operator weights.
        noise_factor: Noise level for repair operators.
        worst_removal_randomness: Randomness parameter p >= 1 for worst removal.
        shaw_randomization: Shaw removal randomization factor.
        max_removal_cap: Hard upper cap on removal count.
        regret_pool: Regret operator pool size.
        sigma_1: Score for new global best solution.
        sigma_2: Score for improving solution.
        sigma_3: Score for accepted worse solution.
        engine: Solver engine to use ('custom', 'alns').
        packing: List of packing algorithm config files.
        route_improvement: List of route improvement operations to apply.
        acceptance_criterion: Acceptance criterion config for local search.
    """

    time_limit: float = 60.0
    seed: Optional[int] = None
    max_iterations: int = 5000
    start_temp: float = 0.0  # 0 = Ropke & Pisinger calibration from the initial profit
    cooling_rate: float = 0.995
    reaction_factor: float = 0.1
    start_temp_control: float = 0.05
    xi: float = 0.4
    segment_size: int = 100
    noise_factor: float = 0.025
    worst_removal_randomness: float = 3.0
    shaw_randomization: float = 6.0
    max_removal_cap: int = 100
    regret_pool: str = "regret234"
    sigma_1: float = 33.0
    sigma_2: float = 9.0
    sigma_3: float = 13.0
    engine: str = "custom"
    profit_aware_operators: bool = True
    extended_operators: bool = False
    packing: Optional[List[PackingConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
    acceptance_criterion: AcceptanceConfig = field(
        default_factory=lambda: AcceptanceConfig(
            method="bmc", params=BoltzmannAcceptanceConfig(initial_temp=100.0, alpha=0.995)
        )
    )
