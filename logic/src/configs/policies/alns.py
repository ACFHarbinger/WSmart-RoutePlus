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
from .other.mandatory_selection import MandatorySelectionConfig
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
        min_removal: Minimum number of nodes to remove per destroy operation.
        start_temp_control: 'w' parameter: accept a 5% worse solution with 0.5 probability
        xi: 'xi' parameter: fraction of total nodes for max removal cap
        engine: Solver engine to use ('custom', 'alns').
        mandatory_selection: List of mandatory strategy config files.
        route_improvement: List of route improvement operations to apply.
        acceptance_criterion: Acceptance criterion config for local search.
    """

    time_limit: float = 60.0
    seed: Optional[int] = None
    max_iterations: int = 5000
    start_temp: float = 0.0  # 0 = Ropke & Pisinger calibration from the initial profit
    cooling_rate: float = 0.995
    reaction_factor: float = 0.1
    min_removal: int = 1
    start_temp_control: float = 0.05
    xi: float = 0.4
    engine: str = "custom"
    vrpp: bool = True
    profit_aware_operators: bool = False
    extended_operators: bool = False
    mandatory_selection: Optional[List[MandatorySelectionConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
    acceptance_criterion: AcceptanceConfig = field(
        default_factory=lambda: AcceptanceConfig(
            method="bmc", params=BoltzmannAcceptanceConfig(initial_temp=100.0, alpha=0.995)
        )
    )
