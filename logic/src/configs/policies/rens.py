"""
Relaxation Enforced Neighborhood Search (RENS) configuration.

Attributes:
    RENSConfig: Configuration for the RENS matheuristic.

Example:
    >>> from configs.policies.rens import RENSConfig
    >>> config = RENSConfig()
    >>> config.time_limit
    60.0
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .other.packing import PackingConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class RENSConfig:
    """
    Configuration for Relaxation Enforced Neighborhood Search (RENS) matheuristic.

    RENS (Berthold, 2009) is a start heuristic that explores the space of
    feasible roundings of an LP relaxation. It works by fixing all variables
    that take integer values in the LP relaxation and solving a restricted
    MIP on the remaining (fractional) variables.

    Attributes:
        time_limit (float): Total maximum runtime for the entire RENS process
            (including LP and MIP phases). Typical range is 30-120s.
        lp_time_limit (float): Hard limit for the initial continuous relaxation
            solve. LP relaxation is usually fast, so 5-10s is standard.
        mip_limit_nodes (int): Limit on branch-and-bound nodes for the sub-MIP.
            Used to prevent stalling on difficult neighborhoods.
        mip_gap (float): Optimality gap threshold for the sub-MIP. Setting this
            slightly higher (e.g., 0.05) often helps find a "good enough"
            primal solution quickly.
        seed (int): Random seed for the underlying Gurobi solver to ensure
            deterministic and reproducible results.
        framework: Optimization framework to use ('ortools' or 'pyomo').
        engine: Optimization engine to use ('gurobi', 'scip', 'highs', or 'cplex').
        packing (Optional[PackingConfig]): Composition handle
            for packing algorithms (e.g., collecting bins at risk of overflow).
        route_improvement (Optional[RouteImprovingConfig]): Settings for
            post-optimization refinement (e.g., Local Search or Route-Smoothing).
    """

    time_limit: float = 60.0
    lp_time_limit: float = 10.0
    mip_limit_nodes: int = 1000
    mip_gap: float = 0.01
    seed: int = 42

    # Infrastructure
    framework: str = "ortools"
    engine: str = "gurobi"
    packing: Optional[PackingConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None


    @classmethod
    def from_config(cls, config: Any) -> "RENSConfig":
        """Create RENSConfig from a configuration object or dictionary.

        Args:
            config: Configuration object.

        Returns:
            "RENSConfig": RENS parameters.
        """
        from dataclasses import fields
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            time_limit=getattr(config, "time_limit", 300.0),
            lp_time_limit=getattr(config, "lp_time_limit", 60.0),
            mip_gap=getattr(config, "mip_gap", 0.01),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert RENSConfig to a dictionary.

        Returns:
            Dict[str, Any]: RENS parameters as dictionary.
        """
        from dataclasses import fields
        return {f.name: getattr(self, f.name) for f in fields(self)}
