"""
KGLS algorithm parameters.

Attributes:
    KGLSParams: Parameters for the KGLS solver.

Example:
    >>> params = KGLSParams(time_limit=300, num_perturbations=10, neighborhood_size=20, moves=["2opt"], penalization_cycle=[], local_search_iterations=100, seed=42)
"""

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class KGLSParams:
    """
    Runtime parameters for the KGLS solver.

    Attributes:
        time_limit: Maximum execution time in seconds.
        num_perturbations: Number of perturbation steps per iteration.
        neighborhood_size: Size of the neighborhood to explore.
        moves: List of local search move types to apply.
        penalization_cycle: List of moves during which penalization is applied.
        local_search_iterations: Number of local search iterations.
        seed: Random seed for reproducibility.
        vrpp: Whether to solve as a VRP with profits.
        profit_aware_operators: Whether to use profit-aware heuristics.
    """

    time_limit: float
    num_perturbations: int
    neighborhood_size: int
    moves: List[str]
    penalization_cycle: List[str]
    local_search_iterations: int
    seed: Optional[int]
    vrpp: bool = True
    profit_aware_operators: bool = True

    @classmethod
    def from_config(cls, config: Any) -> "KGLSParams":
        """Build parameters from a configuration instance.

        Args:
            config: Configuration source (dataclass or object).

        Returns:
            KGLSParams: Initialized runtime parameters.
        """
        seed = getattr(config, "seed", None)
        vrpp = getattr(config, "vrpp", True)
        profit_aware_operators = getattr(config, "profit_aware_operators", False)

        return cls(
            time_limit=getattr(config, "time_limit", 300.0),
            num_perturbations=getattr(config, "num_perturbations", 10),
            neighborhood_size=getattr(config, "neighborhood_size", 20),
            moves=getattr(config, "moves", []),
            penalization_cycle=getattr(config, "penalization_cycle", []),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            seed=seed,
            vrpp=vrpp,
            profit_aware_operators=profit_aware_operators,
        )
