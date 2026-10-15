"""
SLC (Soccer League Competition) configuration for Hydra.

Attributes:
    SLCConfig: Configuration for the Soccer League Competition (SLC) policy.

Example:
    >>> from configs.policies.slc import SLCConfig
    >>> config = SLCConfig()
    >>> config.n_teams
    5
    >>> config.team_size
    4
    >>> config.max_iterations
    50
    >>> config.packing
    []
    >>> config.route_improvement
    []
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class SLCConfig:
    """
    Configuration for the Soccer League Competition policy.

    Attributes:
        n_teams: Number of teams in the league.
        team_size: Number of players per team.
        max_iterations: Maximum number of seasons.
        stagnation_limit: Seasons without improvement before team regeneration.
        n_removal: Nodes removed per perturbation step.
        local_search_iterations: Number of local search iterations.
        time_limit: Wall-clock time limit in seconds.
        packing: Packing algorithm config list.
        route_improvement: Route improvement operation config list.
    """

    n_teams: int = 5
    team_size: int = 4
    max_iterations: int = 50
    stagnation_limit: int = 5
    n_removal: int = 1
    local_search_iterations: int = 500
    time_limit: float = 60.0
    seed: Optional[int] = None
    packing: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
    profit_aware_operators: bool = False


    @classmethod
    def from_config(cls, config: Any) -> "SLCConfig":
        """Create parameters from a configuration object.

        Args:
            config: Configuration object.

        Returns:
            SLCConfig: Parameters for the SLC algorithm.
        """
        return cls(
            n_teams=getattr(config, "n_teams", 5),
            team_size=getattr(config, "team_size", 4),
            max_iterations=getattr(config, "max_iterations", 50),
            stagnation_limit=getattr(config, "stagnation_limit", 5),
            n_removal=getattr(config, "n_removal", 1),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", None),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
