"""
LKH-3 (Lin-Kernighan-Helsgaun 3) configuration for Hydra.

Reference:
    Helsgaun, K. (2017). An extension of the LKH-TSP solver for constrained
    traveling salesman and vehicle routing problems.

Attributes:
    LKH3Config: Configuration for the LKH-3 policy.

Example:
    >>> from configs.policies.lkh3 import LKH3Config
    >>> config = LKH3Config()
    >>> config.time_limit
    60.0
    >>> config.max_trials
    1000
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LKH3Config:
    """Configuration for the LKH-3 policy.

    Attributes:
        max_trials: Maximum number of local-search iterations per run.
        runs: Number of independent LKH runs (multi-start).
        popmusic_subpath_size: Sub-path size for POPMUSIC candidate generation.
        popmusic_trials: Number of POPMUSIC decomposition runs.
        max_k_opt: Maximum k for k-opt moves (2–5).
        use_ip_merging: If True, use IP-based tour merging; else greedy.
        time_limit: Wall-clock time limit in seconds.
        seed: Random seed for reproducibility.
        dynamic_topology_discovery: If True, solver performs dynamic topology discovery.
        sa_max_trials: Maximum number of trials for Simulated Annealing.
        deep_plateau_limit: Number of iterations without improvement before triggering deep search.
        native_prize_collecting: If True, solver uses native prize collecting.
        packing: Packing algorithm config list.
        route_improvement: Route improvement operation config list.
    """

    runs: int = 10
    max_trials: int = 1000
    popmusic_subpath_size: int = 50
    popmusic_trials: int = 50
    popmusic_max_candidates: int = 5
    max_k_opt: int = 5
    use_ip_merging: bool = True
    max_pool_size: int = 5
    subgradient_iterations: int = 50  # Held-Karp ascent iterations for alpha-measure
    profit_aware_operators: bool = False
    alns_iterations: int = 100
    plateau_limit: int = 10
    deep_plateau_limit: int = 30
    sa_max_trials: int = 10000
    perturb_operator_weights: List[float] = field(default_factory=lambda: [0.6, 0.4])
    time_limit: float = 60.0
    dynamic_topology_discovery: bool = False
    native_prize_collecting: bool = False
    seed: Optional[int] = None

    packing: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)


    @classmethod
    def from_config(cls, config: Any) -> "LKH3Config":
        """Create LKH3Config from a configuration object or dictionary.

        Args:
            config: Configuration object or dict with LKH-3 parameter attributes.

        Returns:
            "LKH3Config": Populated parameter dataclass.
        """
        from dataclasses import fields
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            runs=getattr(config, "runs", 10),
            time_limit=getattr(config, "time_limit", 60.0),
            max_trials=getattr(config, "max_trials", 1000),
            popmusic_subpath_size=getattr(config, "popmusic_subpath_size", 50),
            popmusic_trials=getattr(config, "popmusic_trials", 50),
            popmusic_max_candidates=getattr(config, "popmusic_max_candidates", 5),
            max_k_opt=getattr(config, "max_k_opt", 5),
            use_ip_merging=getattr(config, "use_ip_merging", True),
            max_pool_size=getattr(config, "max_pool_size", 5),
            subgradient_iterations=getattr(config, "subgradient_iterations", 50),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            alns_iterations=getattr(config, "alns_iterations", 100),
            plateau_limit=getattr(config, "plateau_limit", 10),
            deep_plateau_limit=getattr(config, "deep_plateau_limit", 30),
            sa_max_trials=getattr(config, "sa_max_trials", 10000),
            perturb_operator_weights=getattr(config, "perturb_operator_weights", [0.6, 0.4]),
            seed=getattr(config, "seed", 42),
            dynamic_topology_discovery=getattr(config, "dynamic_topology_discovery", False),
            native_prize_collecting=getattr(config, "native_prize_collecting", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Params to a dictionary.

        Returns:
            Dict[str, Any]: Mapping of parameter names to their values.
        """
        from dataclasses import fields
        return {f.name: getattr(self, f.name) for f in fields(self)}
