"""
LCA (League Championship Algorithm) configuration for Hydra.

Attributes:
    LCAConfig: Configuration for the League Championship Algorithm policy.

Example:
    >>> from configs.policies.lca import LCAConfig
    >>> config = LCAConfig()
    >>> config.time_limit
    60.0
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class LCAConfig:
    """
    Configuration for the League Championship Algorithm policy.

    Attributes:
        n_teams: Number of teams in the championship.
        max_iterations: Maximum number of weeks (outer loop).
        tolerance_pct: Infeasibility tolerance fraction for match decisions.
        crossover_prob: Probability of OX crossover vs. perturbation after loss.
        n_removal: Nodes removed per perturbation step.
        local_search_iterations: Number of local search iterations.
        time_limit: Wall-clock time limit in seconds.
        packing: Packing algorithm config list.
        route_improvement: Route improvement operation config list.
    """

    n_teams: int = 10
    max_iterations: int = 100
    tolerance_pct: float = 0.05
    crossover_prob: float = 0.6
    n_removal: int = 2
    local_search_iterations: int = 500
    time_limit: float = 60.0
    seed: Optional[int] = None
    profit_aware_operators: bool = False
    packing: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)


    @classmethod
    def from_config(cls, config: Any) -> "LCAConfig":
        """Create parameters from a configuration object.

        Args:
            config: Configuration source (dataclass or object).

        Returns:
            LCAConfig: Initialized runtime parameters.
        """
        return cls(
            n_teams=getattr(config, "n_teams", 10),
            max_iterations=getattr(config, "max_iterations", 100),
            tolerance_pct=getattr(config, "tolerance_pct", 0.05),
            crossover_prob=getattr(config, "crossover_prob", 0.6),
            n_removal=getattr(config, "n_removal", 2),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", None),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
