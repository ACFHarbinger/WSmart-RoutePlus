"""
RTS (Reactive Tabu Search) configuration for Hydra.

Attributes:
    RTSConfig: Configuration for the Reactive Tabu Search (RTS) policy.

Example:
    >>> from configs.policies.rts import RTSConfig
    >>> config = RTSConfig()
    >>> config.initial_tenure
    7
    >>> config.max_iterations
    500
    >>> config.vrpp
    True
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .other.acceptance_criteria import AcceptanceConfig, AspirationConfig


@dataclass
class RTSConfig:
    """Configuration for the Reactive Tabu Search policy.

    Attributes:
        initial_tenure: Initial tenure for the tabu list.
        min_tenure: Minimum tenure for the tabu list.
        max_tenure: Maximum tenure for the tabu list.
        tenure_increase: Rate of increase for the tenure.
        tenure_decrease: Rate of decrease for the tenure.
        max_iterations: Maximum number of iterations.
        n_removal: Number of nodes to remove in each iteration.
        n_llh: Number of local search operators to apply in each iteration.
        time_limit: Time limit for the algorithm in seconds.
        seed: Random seed for the algorithm.
        vrpp: Whether to use VRPP.
        profit_aware_operators: Whether to use profit-aware operators.
        packing: List of mandatory nodes.
        route_improvement: List of route improvements.
        acceptance_criterion: Acceptance criterion configuration.
    """

    initial_tenure: int = 7
    min_tenure: int = 3
    max_tenure: int = 20
    tenure_increase: float = 1.5
    tenure_decrease: float = 0.9
    max_iterations: int = 500
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False
    packing: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
    acceptance_criterion: AcceptanceConfig = AcceptanceConfig(method="ac", params=AspirationConfig())
