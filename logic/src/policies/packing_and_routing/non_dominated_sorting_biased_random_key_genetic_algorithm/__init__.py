"""NDS-BRKGA sub-package.

Exposes the public API of the Non-Dominated Sorting Biased Random-Key
Genetic Algorithm (NDS-BRKGA) core components.

Attributes:
    NDSBRKGAPolicy (Type[NDSBRKGAPolicy]): Main policy for joint packing and routing.
    NDSBRKGAConfig (Type[NDSBRKGAConfig]): Hyperparameter configuration dataclass.
    Chromosome (Type[Chromosome]): Representation of a solution candidate.
    Population (Type[Population]): Manager for solution candidates and evolution.
    biased_crossover (Callable): Biased uniform crossover operator.
    fast_non_dominated_sort (Callable): NSGA-II sorting algorithm.
    crowding_distance (Callable): Crowd distance assignment for diversity.
    select_elite_nsga2 (Callable): Pareto-based elite selection.
    compute_overflow_risk (Callable): Risk calculation utility.
    evaluate_chromosome (Callable): Single candidate objective evaluator.
    evaluate_population (Callable): Batch population evaluator.
    compute_adaptive_thresholds (Callable): Dynamic selection threshold calculator.

Example:
    >>> from logic.src.policies.packing_and_routing.non_dominated_sorting_biased_random_key_genetic_algorithm import NDSBRKGAPolicy
    >>> policy = NDSBRKGAPolicy()
"""

from .chromosome import Chromosome, compute_adaptive_thresholds
from .crossover import biased_crossover
from .nsga2 import crowding_distance, fast_non_dominated_sort, select_elite_nsga2
from .objectives import compute_overflow_risk, evaluate_chromosome, evaluate_population
from logic.src.configs.policies.nds_brkga import NDSBRKGAConfig
from .policy_nds_brkga import NDSBRKGAPolicy
from .population import Population

__all__ = [
    "Chromosome",
    "compute_adaptive_thresholds",
    "biased_crossover",
    "fast_non_dominated_sort",
    "crowding_distance",
    "select_elite_nsga2",
    "compute_overflow_risk",
    "evaluate_chromosome",
    "evaluate_population",
    "NDSBRKGAConfig",
    "Population",
    "NDSBRKGAPolicy",
]
