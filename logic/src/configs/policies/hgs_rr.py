"""
HGS-RR (Hybrid Genetic Search with Ruin-and-Recreate) configuration.

Attributes:
    HGSRRConfig: Configuration for the HGS-RR policy.

Example:
    >>> from configs.policies.hgs_rr import HGSRRConfig
    >>> config = HGSRRConfig()
    >>> config.time_limit
    60.0
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .other.packing import PackingConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class HGSRRConfig:
    """Configuration for Hybrid Genetic Search with Ruin-and-Recreate (HGS-RR) policy.

    Attributes:
    restart_timer: Maximum wall-clock seconds for optimization, before restarting the algorithm (0 = unlimited)
        time_limit: Maximum time in seconds for the solver.
        seed: Random seed for reproducibility.
        population_size: Size of the genetic population.
        elite_size: Number of elite individuals to preserve.
        mutation_rate: Probability of applying ruin-recreate mutation.
        crossover_rate: Probability of crossover operation.
        n_iterations_no_improvement: Max iterations without improvement.
        no_improvement_threshold: Number of generations without improvement to trigger diversity maintenance.
        survivor_threshold: Threshold for survivor selection.
        neighbor_list_size: Number of nearest neighbors to consider.
        max_vehicles: Maximum number of vehicles (0 for unlimited).

        # Ruin-and-Recreate specific parameters
        min_removal_pct: Minimum percentage of nodes to remove (0.0-1.0).
        max_removal_pct: Maximum percentage of nodes to remove (0.0-1.0).
        noise_factor: Noise factor for randomized insertion.
        reaction_factor: Rate of operator weight updates (0.0-1.0).
        decay_parameter: Exponential decay rate for operator weights (0.0-1.0).

        # Operator selection
        destroy_operators: List of destroy operator names.
        repair_operators: List of repair operator names.
        operator_decay_rate: Exponential decay rate for operator weights.

        # Adaptive scoring
        score_sigma_1: Score for new global best solution.
        score_sigma_2: Score for solution improving current.
        score_sigma_3: Score for solution accepted but not improving.

        packing: List of packing algorithm config files.
        route_improvement: List of route improvement operations to apply.
    """

    restart_timer: float = 0.0
    time_limit: float = 60.0
    seed: Optional[int] = None
    population_size: int = 50
    elite_size: int = 10
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    n_iterations_no_improvement: int = 20000
    no_improvement_threshold: int = 20
    survivor_threshold: float = 2.0
    neighbor_list_size: int = 10
    max_vehicles: int = 0
    profit_aware_operators: bool = False

    # Ruin-and-Recreate parameters
    min_removal_pct: float = 0.1
    max_removal_pct: float = 0.4
    noise_factor: float = 0.015
    reaction_factor: float = 0.1
    decay_parameter: float = 0.95

    # Operator management
    destroy_operators: List[str] = field(
        default_factory=lambda: [
            "random_removal",
            "worst_removal",
            "cluster_removal",
            "shaw_removal",
            "string_removal",
        ]
    )
    repair_operators: List[str] = field(
        default_factory=lambda: [
            "greedy_insertion",
            "regret_2_insertion",
            "regret_k_insertion",
            "greedy_insertion_with_blinks",
        ]
    )
    operator_decay_rate: float = 0.95

    # Adaptive scoring
    score_sigma_1: float = 33.0
    score_sigma_2: float = 9.0
    score_sigma_3: float = 3.0

    packing: Optional[List[PackingConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None


    @classmethod
    def from_config(cls, config: Any) -> "HGSRRConfig":
        """Create parameters from a configuration object.

        Args:
            config: Configuration object (Hydra config or dataclass).

        Returns:
            HGSRRConfig: Initialized parameters.
        """
        return cls(
            time_limit=getattr(config, "time_limit", 10.0),
            population_size=getattr(config, "population_size", 50),
            elite_size=getattr(config, "elite_size", 10),
            mutation_rate=getattr(config, "mutation_rate", 0.3),
            n_iterations_no_improvement=getattr(config, "n_iterations_no_improvement", 20000),
            no_improvement_threshold=getattr(config, "no_improvement_threshold", 20),
            survivor_threshold=getattr(config, "survivor_threshold", 2.0),
            max_vehicles=getattr(config, "max_vehicles", 0),
            crossover_rate=getattr(config, "crossover_rate", 0.7),
            neighbor_list_size=getattr(config, "neighbor_list_size", 10),
            min_removal_pct=getattr(config, "min_removal_pct", 0.1),
            max_removal_pct=getattr(config, "max_removal_pct", 0.4),
            noise_factor=getattr(config, "noise_factor", 0.015),
            reaction_factor=getattr(config, "reaction_factor", 0.1),
            decay_parameter=getattr(config, "decay_parameter", 0.95),
            destroy_operators=getattr(
                config,
                "destroy_operators",
                ["random_removal", "worst_removal", "cluster_removal", "shaw_removal", "string_removal"],
            ),
            repair_operators=getattr(
                config,
                "repair_operators",
                ["greedy_insertion", "regret_2_insertion", "regret_k_insertion", "greedy_insertion_with_blinks"],
            ),
            operator_decay_rate=getattr(config, "operator_decay_rate", 0.95),
            score_sigma_1=getattr(config, "score_sigma_1", 33.0),
            score_sigma_2=getattr(config, "score_sigma_2", 9.0),
            score_sigma_3=getattr(config, "score_sigma_3", 3.0),
            seed=getattr(config, "seed", None),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
