"""
GLS (Guided Local Search) configuration for Hydra.

Attributes:
    GLSConfig: Configuration for the Guided Local Search policy.

Example:
    >>> from configs.policies.gls import GLSConfig
    >>> config = GLSConfig()
    >>> config.time_limit
    60.0
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class GLSConfig:
    """Configuration for the Guided Local Search policy.

    Attributes:
        lambda_param (float): Lambda parameter for GLS.
        alpha_param (float): Alpha parameter for GLS.
        penalty_cycles (int): Number of penalty cycles.
        n_removal (int): Number of routes to remove.
        n_llh (int): Number of local low-level heuristics.
        inner_iterations (int): Number of inner iterations.
        fls_coupling_prob (float): Probability of coupling FLS.
        time_limit (float): Time limit in seconds.
        seed (Optional[int]): Seed for the random number generator.
        profit_aware_operators (bool): Whether to use profit-aware operators.
        packing (Optional[List[PackingConfig]]): Packing configurations.
        route_improvement (Optional[List[RouteImprovingConfig]]): Route improvement configurations.
    """

    lambda_param: float = 1.0
    alpha_param: float = 0.3
    penalty_cycles: int = 1000
    n_removal: int = 2
    n_llh: int = 6
    inner_iterations: int = 100
    fls_coupling_prob: float = 0.8
    time_limit: float = 60.0
    seed: Optional[int] = None
    profit_aware_operators: bool = False
    packing: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
