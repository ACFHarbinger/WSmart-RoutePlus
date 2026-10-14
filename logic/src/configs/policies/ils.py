"""
ILS (Iterated Local Search) configuration for Hydra.

Attributes:
    ILSConfig: Configuration for the Iterated Local Search policy.

Example:
    >>> from configs.policies.ils import ILSConfig
    >>> config = ILSConfig()
    >>> config.time_limit
    60.0
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from logic.src.configs.policies.other.acceptance_criteria import AcceptanceConfig


@dataclass
class ILSConfig:
    """Configuration for the Iterated Local Search policy.

    Attributes:
        engine (str): Name of the engine.
        n_restarts (int): Number of restarts.
        inner_iterations (int): Number of inner iterations.
        n_removal (int): Number of removals for perturbation.
        n_llh (int): Number of local heuristic calls per restart.
        perturbation_strength (float): Strength of perturbation (0-1).
        time_limit (float): Time limit in seconds.
        seed (Optional[int]): Random seed for reproducibility.
        vrpp (bool): Whether the problem is a VRRP.
        profit_aware_operators (bool): Use profit-aware operators.
        packing (Optional[List[Any]]): Mandatory customers/requests selection.
        route_improvement (Optional[List[Any]]): Route improvement strategies.
        acceptance_criterion (AcceptanceConfig): Acceptance criteria for new solutions.
    """

    engine: str = "ils"
    n_restarts: int = 30
    inner_iterations: int = 20
    n_removal: int = 2
    n_llh: int = 5
    perturbation_strength: float = 0.15
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False
    packing: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
    acceptance_criterion: AcceptanceConfig = field(default_factory=lambda: AcceptanceConfig(method="oi"))
