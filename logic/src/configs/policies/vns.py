"""
VNS (Variable Neighborhood Search) configuration for Hydra.

Attributes:
    VNSConfig: Configuration for the Variable Neighborhood Search (VNS) policy.

Example:
    >>> from configs.policies.vns import VNSConfig
    >>> config = VNSConfig()
    >>> config.k_max
    5
    >>> config.max_iterations
    200
    >>> config.local_search_iterations
    500
    >>> config.n_removal
    2
    >>> config.n_llh
    5
    >>> config.time_limit
    60.0
    >>> config.seed
    None
    >>> config.profit_aware_operators
    False
    >>> config.packing
    []
    >>> config.route_improvement
    []
    >>> config.acceptance_criterion.method
    'oi'
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from logic.src.configs.policies.other.acceptance_criteria import AcceptanceConfig


@dataclass
class VNSConfig:
    """Configuration for the Variable Neighborhood Search policy.

    Attributes:
        k_max: Maximum neighborhood index (number of different neighborhood structures to explore).
        max_iterations: Maximum number of iterations to run the search.
        local_search_iterations: Number of iterations to run local search at each neighborhood level.
        n_removal: Number of routes/sequences to remove in each iteration (for destroying and rebuilding).
        n_llh: Number of local search operators to apply in each iteration.
        time_limit: Maximum time in seconds to run the search.
        seed: Random seed for reproducibility.
        profit_aware_operators: Whether to use profit-aware considerations in neighborhood operators.
        packing: List of packing algorithms to apply.
        route_improvement: List of route improvement strategies to apply.
        acceptance_criterion: Configuration for the acceptance criterion (e.g., simulated annealing).
    """

    k_max: int = 5
    max_iterations: int = 200
    local_search_iterations: int = 500
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
    seed: Optional[int] = None
    profit_aware_operators: bool = False
    packing: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
    acceptance_criterion: AcceptanceConfig = field(default_factory=lambda: AcceptanceConfig(method="oi"))
