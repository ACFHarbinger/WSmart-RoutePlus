"""
Parameters for Hybrid Genetic Search with Ruin-and-Recreate (HGS-RR).

This module defines the configuration parameters for the HGS-RR algorithm,
extending standard HGS with adaptive destroy/repair operator management.

Attributes:
    HGSRRParams: Configuration parameters for the HGS-RR algorithm.

Example:
    >>> params = HGSRRParams(time_limit=60.0)
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class HGSRRParams:
    """
    Configuration parameters for HGS-RR algorithm.

    Attributes:
        restart_timer: Maximum wall-clock seconds before algorithm restart (0 = unlimited).
        time_limit: Maximum execution time in seconds.
        population_size: Minimum population size (for each subpopulation).
        elite_size: Number of elite individuals preserved.
        mutation_rate: Probability of applying ruin-recreate mutation.
        n_iterations_no_improvement: Max iterations without improvement.
        no_improvement_threshold: Threshold for diversity/stopping.
        survivor_threshold: Population size multiplier for survivor selection.
        max_vehicles: Maximum number of vehicles (0 = unlimited).
        crossover_rate: Probability of crossover operation.
        neighbor_list_size: Granular search parameter for local search.
        min_removal_pct: Minimum percentage of nodes to remove (0.0-1.0).
        max_removal_pct: Maximum percentage of nodes to remove (0.0-1.0).
        noise_factor: Noise factor for randomized best insertion.
        reaction_factor: Rate of operator weight updates.
        decay_parameter: Decay rate for operator scores.
        destroy_operators: List of destroy operator names.
        repair_operators: List of repair operator names.
        operator_decay_rate: Exponential decay rate for operator weights.
        score_sigma_1: Score for new global best solution.
        score_sigma_2: Score for solution improving current.
        score_sigma_3: Score for solution accepted but not improving.
        seed: Random seed for reproducibility.
        vrpp: Whether to use VRP with Profits mode.
        profit_aware_operators: Whether to use profit-aware operators.
    """

    # Core HGS parameters
    restart_timer: float = 0.0
    time_limit: float = 10.0
    population_size: int = 50
    elite_size: int = 10
    mutation_rate: float = 0.3
    n_iterations_no_improvement: int = 20000
    no_improvement_threshold: int = 20  # Threshold for diversity/stopping
    survivor_threshold: float = 2.0
    max_vehicles: int = 0
    crossover_rate: float = 0.7
    neighbor_list_size: int = 10

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
    score_sigma_1: float = 33.0  # New global best
    score_sigma_2: float = 9.0  # Improvement
    score_sigma_3: float = 3.0  # Accepted

    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    @classmethod
    def from_config(cls, config: Any) -> "HGSRRParams":
        """Create parameters from a configuration object.

        Args:
            config: Configuration object (Hydra config or dataclass).

        Returns:
            HGSRRParams: Initialized parameters.
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
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
