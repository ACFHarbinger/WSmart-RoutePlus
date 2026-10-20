"""
BP (Branch-and-Price) configuration.

Attributes:
    BPConfig: Attributes for BP configuration.

Example:
    >>> from configs.policies.bp import BPConfig
    >>> config = BPConfig()
    >>> config.time_limit
    300.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from logic.src.configs.policies.other.packing_dispatch import PackingConfig

from .other.route_improvement import RouteImprovingConfig


@dataclass
class BPConfig:
    """Configuration for Branch-and-Price (BP) policy.

    Branch-and-Price is a column generation method that handles large-scale
    optimization by implicitly enumerating exponentially many variables (routes).

    The algorithm uses:
    1. Set partitioning master problem (select routes to cover nodes)
    2. Pricing subproblem (generate profitable routes via RCSPP)
    3. Column generation (iteratively add routes with positive reduced cost)
    4. Branch-and-bound for integrality (Ryan-Foster branching)

    Key advantages:
    - Handles problems with huge solution spaces
    - Provides strong LP bounds (tighter than compact formulations)
    - Scalable to large instances

    Attributes:
        time_limit: Maximum time in seconds for the solver.
        max_iterations: Maximum column generation iterations per node.
        max_routes_per_iteration: Maximum routes to generate per pricing call.
        optimality_gap: Convergence tolerance for column generation.
        use_ryan_foster_branching: Use Ryan-Foster branching (not yet implemented).
        packing: List of packing algorithm config files.
        route_improvement: List of route improvement operations to apply.
        seed: Random seed for reproducibility.
    """

    time_limit: float = 300.0
    max_iterations: int = 100
    max_routes_per_iteration: int = 10
    optimality_gap: float = 1e-4
    use_ryan_foster_branching: bool = False
    multi_day_mode: bool = False
    packing: Optional[List[PackingConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
    seed: Optional[int] = None
    branching_strategy: str = "edge"
    max_branch_nodes: int = 1000
    use_exact_pricing: bool = False
    use_ng_routes: bool = True
    ng_neighborhood_size: int = 8
    tree_search_strategy: str = "best_first"
    vehicle_limit: Optional[int] = None
    cleanup_frequency: int = 20
    cleanup_threshold: float = -100.0
    early_termination_gap: float = 1e-3
    multiple_waste_types: bool = False
    allow_heuristic_ryan_foster: bool = False
    use_ryan_foster: bool = False

    @classmethod
    def from_config(cls, config: Any) -> "BPConfig":
        """Create a BPConfig instance from a configuration object or dictionary.

        Args:
            config: A configuration object (Hydra/OmegaConf) or a raw dictionary.

        Returns:
            A BPConfig instance with values mapped from the config.
        """
        from dataclasses import fields

        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            max_iterations=getattr(config, "max_iterations", 100),
            max_routes_per_iteration=getattr(config, "max_routes_per_iteration", 10),
            optimality_gap=getattr(config, "optimality_gap", 1e-4),
            branching_strategy=getattr(config, "branching_strategy", "edge"),
            max_branch_nodes=getattr(config, "max_branch_nodes", 1000),
            use_exact_pricing=getattr(config, "use_exact_pricing", False),
            use_ng_routes=getattr(config, "use_ng_routes", True),
            ng_neighborhood_size=getattr(config, "ng_neighborhood_size", 8),
            tree_search_strategy=getattr(config, "tree_search_strategy", "best_first"),
            vehicle_limit=getattr(config, "vehicle_limit", None),
            cleanup_frequency=getattr(config, "cleanup_frequency", 20),
            cleanup_threshold=getattr(config, "cleanup_threshold", -100.0),
            early_termination_gap=getattr(config, "early_termination_gap", 1e-3),
            multiple_waste_types=getattr(config, "multiple_waste_types", False),
            allow_heuristic_ryan_foster=getattr(config, "allow_heuristic_ryan_foster", False),
        )
