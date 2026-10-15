"""
KGLS (Knowledge-Guided Local Search) configuration dataclasses.

Attributes:
    KGLSConfig: Configuration for the Knowledge-Guided Local Search policy.

Example:
    >>> from configs.policies.kgls import KGLSConfig
    >>> config = KGLSConfig()
    >>> config.time_limit
    60.0
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .abc import ABCConfig


@dataclass
class KGLSConfig(ABCConfig):
    """Configuration for the Knowledge-Guided Local Search algorithm.

    Attributes:
        time_limit (float): Time limit in seconds.
        num_perturbations (int): Number of perturbations.
        neighborhood_size (int): Size of the neighborhood.
        local_search_iterations (int): Number of local search iterations.
        moves (List[str]): List of moves to apply.
        penalization_cycle (List[str]): Cycle of penalization criteria.
        seed (Optional[int]): Random seed for reproducibility.
        profit_aware_operators (bool): Whether to use profit-aware operators.
        packing (Optional[List[Any]]): Mandatory customers/requests selection.
        route_improvement (Optional[List[Any]]): Route improvement strategies.
    """

    time_limit: float = 60.0
    num_perturbations: int = 3
    neighborhood_size: int = 20
    local_search_iterations: int = 100

    # Operators to apply during LS phase
    moves: List[str] = field(default_factory=lambda: ["relocate", "swap", "two_opt", "cross_exchange"])

    # Sequence of criteria to evaluate "badness" of an edge
    penalization_cycle: List[str] = field(default_factory=lambda: ["width", "length", "width_length"])

    seed: Optional[int] = 42
    profit_aware_operators: bool = False


    @classmethod
    def from_config(cls, config: Any) -> "KGLSConfig":
        """Build parameters from a configuration instance.

        Args:
            config: Configuration source (dataclass or object).

        Returns:
            KGLSConfig: Initialized runtime parameters.
        """
        seed = getattr(config, "seed", None)
        profit_aware_operators = getattr(config, "profit_aware_operators", False)

        return cls(
            time_limit=getattr(config, "time_limit", 300.0),
            num_perturbations=getattr(config, "num_perturbations", 10),
            neighborhood_size=getattr(config, "neighborhood_size", 20),
            moves=getattr(config, "moves", []),
            penalization_cycle=getattr(config, "penalization_cycle", []),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            seed=seed,
            profit_aware_operators=profit_aware_operators,
        )
