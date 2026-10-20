"""
POPMUSIC (Partial Optimization Metaheuristic Under Special Intensification Conditions) configuration.

Attributes:
    POPMUSICConfig: Configuration for the POPMUSIC matheuristic.

Example:
    >>> from configs.policies.popmusic import POPMUSICConfig
    >>> config = POPMUSICConfig()
    >>> config.subproblem_size
    3
    >>> config.max_iterations
    None
    >>> config.base_solver
    'alns'
    >>> config.cluster_solver
    'fast_tsp'
    >>> config.initial_solver
    'nearest_neighbor'
    >>> config.seed
    None
    >>> config.profit_aware_operators
    False
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from logic.src.configs.policies.other.packing_dispatch import PackingConfig

from .alns import ALNSConfig
from .hgs import HGSConfig
from .other.route_improvement import RouteImprovingConfig
from .tsp import TSPConfig


@dataclass
class POPMUSICSubSolverConfig:
    """Nested configuration for sub-solvers within POPMUSIC.

    Attributes:
        tsp: Configuration for TSP solver.
        alns: Configuration for ALNS solver.
        hgs: Configuration for HGS solver.
    """

    tsp: Optional[TSPConfig] = field(default_factory=TSPConfig)
    alns: Optional[ALNSConfig] = field(default_factory=ALNSConfig)
    hgs: Optional[HGSConfig] = field(default_factory=HGSConfig)


@dataclass
class POPMUSICConfig:
    """Configuration for POPMUSIC matheuristic.
    Based on Taillard & Voss (2002).

    Attributes:
        subproblem_size: Number of neighboring routes to optimize (R).
        max_iterations: Maximum number of sub-problem attempts without improvement.
        base_solver: The solver configuration used for subproblem optimization.
        base_solver_config: Detailed configuration for the subproblem solver.
        cluster_solver: The solver configuration used for clustering.
        cluster_solver_config: Detailed configuration for the clustering solver.
        initial_solver: Solver used to generate the initial solution (e.g., 'nearest_neighbor').
        seed: Random seed for reproducibility.
        packing: List of packing algorithm config files.
        route_improvement: List of route improvement operations to apply.
    """

    # Core POPMUSIC parameters
    subproblem_size: int = 3
    k_prox: int = 10  # proximity network radius (paper §2.1)
    seed_strategy: str = "lifo"  # "lifo" | "fifo" | "random"
    max_iterations: Optional[int] = None  # None = run until U=empty

    # Solver orchestration
    base_solver: str = "alns"
    base_solver_config: Optional[POPMUSICSubSolverConfig] = field(default_factory=POPMUSICSubSolverConfig)
    cluster_solver: str = "fast_tsp"
    cluster_solver_config: Optional[POPMUSICSubSolverConfig] = field(default_factory=POPMUSICSubSolverConfig)
    initial_solver: str = "pmedian"
    seed: Optional[int] = None

    # Infrastructure
    profit_aware_operators: bool = False
    packing: Optional[List[PackingConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None

    @classmethod
    def from_config(cls, config: Any) -> "POPMUSICConfig":
        """Create POPMUSICConfig from a configuration object or dictionary.

        Args:
            config (Any): Configuration object.

        Returns:
            "POPMUSICConfig": Instance of POPMUSICConfig.
        """
        from dataclasses import fields

        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            subproblem_size=getattr(config, "subproblem_size", 3),
            max_iterations=getattr(config, "max_iterations", None),
            base_solver=getattr(config, "base_solver", "fast_tsp"),
            base_solver_config=getattr(config, "base_solver_config", None),
            cluster_solver=getattr(config, "cluster_solver", "fast_tsp"),
            cluster_solver_config=getattr(config, "cluster_solver_config", None),
            initial_solver=getattr(config, "initial_solver", "pmedian"),
            seed=getattr(config, "seed", 42),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            k_prox=getattr(config, "k_prox", 10),
            seed_strategy=getattr(config, "seed_strategy", "lifo"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert POPMUSICConfig to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing POPMUSICConfig.
        """
        from dataclasses import fields

        return {f.name: getattr(self, f.name) for f in fields(self)}
