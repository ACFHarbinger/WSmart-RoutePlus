"""
GIHH (Hyper-Heuristic with Two Guidance Indicators) configuration.

Attributes:
    GIHHConfig: Configuration for the GIHH (Hyper-Heuristic with Two Guidance Indicators) policy.

Example:
    >>> from configs.policies.gihh import GIHHConfig
    >>> config = GIHHConfig()
    >>> config.time_limit
    60.0
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.acceptance_criteria import AcceptanceConfig, OnlyImprovingConfig
from .other.packing import PackingConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class GIHHConfig:
    """Configuration for GIHH (Hyper-Heuristic with Two Guidance Indicators) policy.

    Attributes:
        time_limit: Maximum time in seconds for the solver.
        seed: Random seed for reproducibility.
        max_iterations: Maximum number of iterations.
        vrpp: If True, enable VRPP mode (visit subset profitably).

        # Episodic Learning Parameters (Chen et al. 2018)
        seg: Segment size for episodic weight updates.
        alpha: Weight momentum parameter.
        beta: Quality reward weight parameter.
        gamma: Directional penalty parameter.
        min_prob: Minimum selection probability for any operator.

        # Stopping criteria
        nonimp_threshold: Maximum iterations without improvement before stop.

        packing: List of packing algorithm config files.
        route_improvement: List of route improvement operations to apply.
        acceptance_criterion: Acceptance criterion config for local search.
    """

    time_limit: float = 60.0
    seed: Optional[int] = None
    max_iterations: int = 1000
    vrpp: bool = True
    profit_aware_operators: bool = False

    # Episodic Learning Parameters
    seg: int = 80
    alpha: float = 0.5
    beta: float = 0.4
    gamma: float = 0.1
    min_prob: float = 0.05

    # Stopping criteria
    nonimp_threshold: int = 150

    packing: Optional[List[PackingConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
    acceptance_criterion: AcceptanceConfig = AcceptanceConfig(method="only_improving", params=OnlyImprovingConfig())
