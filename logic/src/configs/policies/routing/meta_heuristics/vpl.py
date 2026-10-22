"""
VPL Configuration for Hydra.

Reference:
    Moghdani, R., & Salimifard, K. (2018). "Volleyball Premier League
    Algorithm." Applied Soft Computing, 64, 161-185.
    DOI: https://doi.org/10.1016/j.asoc.2017.11.043

Attributes:
    VPLConfig: Configuration for the Volleyball Premier League policy.

Example:
    >>> from configs.policies.vpl import VPLConfig
    >>> config = VPLConfig()
    >>> config.n_teams
    30
    >>> config.max_iterations
    200
    >>> config.substitution_rate
    0.2
    >>> config.coaching_weight_1
    0.5
    >>> config.coaching_weight_2
    0.3
    >>> config.coaching_weight_3
    0.2
    >>> config.elite_size
    3
    >>> config.local_search_iterations
    100
    >>> config.time_limit
    300.0
    >>> config.seed
    None
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class VPLConfig:
    """
    Configuration for the Volleyball Premier League policy.

    VPL maintains a dual population structure with active and passive teams.
    The algorithm features four main phases: Team Formation, Competition,
    Substitution, and Coaching & Learning.

    Attributes:
        n_teams: Number of active teams (total = 2 * n_teams).
        max_iterations: Maximum number of iterations to run the search.
        substitution_rate: Diversity injection rate from passive teams.
        coaching_weight_1: Learning weight for the best team.
        coaching_weight_2: Learning weight for the 2nd best team.
        coaching_weight_3: Learning weight for the 3rd best team.
        elite_size: Number of elite teams to preserve.
        local_search_iterations: Number of iterations to run local search at each neighborhood level.
        time_limit: Maximum time in seconds to run the search.
        seed: Random seed for reproducibility.
        packing: List of packing algorithms to apply.
        route_improvement: List of route improvement strategies to apply.
    """

    # VPL Population Parameters
    n_teams: int = 30  # Number of active teams (total = 2 * n_teams)
    max_iterations: int = 200
    substitution_rate: float = 0.2  # Diversity injection rate from passive teams

    # Coaching and Learning Parameters
    coaching_weight_1: float = 0.5  # Learning weight for best team
    coaching_weight_2: float = 0.3  # Learning weight for 2nd best team
    coaching_weight_3: float = 0.2  # Learning weight for 3rd best team
    elite_size: int = 3  # Number of elite teams preserved

    # Local Search Parameters
    local_search_iterations: int = 100

    # Global Parameters
    time_limit: float = 300.0
    seed: Optional[int] = None

    # Common policy fields
    packing: List[str] = field(default_factory=list)
    route_improvement: List[Any] = field(default_factory=list)
    profit_aware_operators: bool = False


    def __post_init__(self):
        """Validate parameter constraints.

        Raises:
            AssertionError: If parameters are invalid.
        """
        assert self.n_teams > 0, "n_teams must be positive"
        assert 0 <= self.substitution_rate <= 1, "substitution_rate must be in [0, 1]"
        assert self.elite_size >= 1, "elite_size must be at least 1"

        # Validate coaching weights sum to 1.0
        total_weight = self.coaching_weight_1 + self.coaching_weight_2 + self.coaching_weight_3
        assert abs(total_weight - 1.0) < 1e-6, f"Coaching weights must sum to 1.0, got {total_weight}"

    @classmethod
    def from_config(cls, config: Any) -> "VPLConfig":
        """Create parameters from a configuration object.

        Args:
            config: Configuration object or dictionary.

        Returns:
            VPLConfig: Parameter instance.
        """
        return cls(
            n_teams=getattr(config, "n_teams", 30),
            max_iterations=getattr(config, "max_iterations", 200),
            substitution_rate=getattr(config, "substitution_rate", 0.2),
            coaching_weight_1=getattr(config, "coaching_weight_1", 0.5),
            coaching_weight_2=getattr(config, "coaching_weight_2", 0.3),
            coaching_weight_3=getattr(config, "coaching_weight_3", 0.2),
            elite_size=getattr(config, "elite_size", 3),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            time_limit=getattr(config, "time_limit", 300.0),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
