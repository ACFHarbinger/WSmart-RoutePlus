"""
Configuration for the Local Branching (LB) matheuristic.

Attributes:
    LocalBranchingConfig: Configuration for the Local Branching policy.

Example:
    >>> from configs.policies.lb import LocalBranchingConfig
    >>> config = LocalBranchingConfig()
    >>> config.time_limit
    300.0
    >>> config.k
    10
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .other.packing import PackingConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class LocalBranchingConfig:
    """
    Configuration for the Local Branching (LB) matheuristic.

    Local Branching (Fischetti and Lodi, 2003) is a matheuristic that uses a
    general-purpose MIP solver to explore the k-neighborhood of a given
    incumbent solution.

    Attributes:
        time_limit (float): Total wall-clock time limit for the policy.
        time_limit_per_iteration (float): Time limit for each sub-MIP solve.
        k (int): Neighborhood size (Hamming distance).
        max_iterations (int): Maximum number of improvement iterations.
        node_limit_per_iteration (int): Branch-and-bound node limit per sub-MIP.
        mip_gap (float): Targeted relative optimality gap for sub-problems.
        seed (int): Random seed for reproducibility.

        # Infrastructure
        engine (str): Solver engine to use ('gurobi', 'scip', 'highs', or 'cplex').
        framework (str): Solver framework to use ('ortools', 'pyomo').
        packing (Optional[PackingConfig]): Configuration for mandatory
            node selection.
        route_improvement (Optional[RouteImprovingConfig]): Optional local search refinement.
    """

    time_limit: float = 300.0
    time_limit_per_iteration: float = 30.0
    k: int = 10
    max_iterations: int = 20
    node_limit_per_iteration: int = 5000
    mip_gap: float = 0.01
    seed: int = 42

    # Infrastructure
    engine: str = "gurobi"
    framework: str = "ortools"
    packing: Optional[PackingConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None


    @classmethod
    def from_config(cls, config: Any) -> "LocalBranchingConfig":
        """Create LocalBranchingConfig from a configuration object or dictionary.

        Args:
            config (Any): Configuration object or dictionary.

        Returns:
            "LocalBranchingConfig": Configuration parameters for the Local Branching (LB) policy.
        """
        from dataclasses import fields
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            k=getattr(config, "k", 10),
            max_iterations=getattr(config, "max_iterations", 100),
            time_limit=getattr(config, "time_limit", 60.0),
            time_limit_per_iteration=getattr(config, "time_limit_per_iteration", 10.0),
            node_limit_per_iteration=getattr(config, "node_limit_per_iteration", 1000),
            mip_gap=getattr(config, "mip_gap", 0.01),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Params to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary of LocalBranchingConfig.
        """
        from dataclasses import fields
        return {f.name: getattr(self, f.name) for f in fields(self)}
