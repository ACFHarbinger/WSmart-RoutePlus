"""Expert-policy configurations for imitation learning.

These dataclasses are the training-time expert settings. Hydra targets use
``logic.src.configs.learning.experts.<Class>``. The class name is the canonical
name: there is no short alias.

Attributes:
    ACOConfig: Ant colony optimisation.
    VectorALNSConfig: Adaptive large neighbourhood search.
    VectorHGSConfig: Hybrid genetic search.
    VectorHGSALNSConfig: Hybrid genetic search with ALNS education.
    VectorHVPLConfig: Hybrid volleyball premier league.
    VectorILSConfig: Iterated local search.
    ProfitableNNConfig: Profitable nearest neighbour.
    RLSConfig: Random local search.

Example:
    config = VectorALNSConfig(time_limit=30.0, max_iterations=5000)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Union


@dataclass
class ACOConfig:
    """Configuration for Ant Colony Optimization expert policy (HH-ACO, KS-ACO).

    Attributes:
        n_ants: Number of ants in the colony.
        alpha: Pheromone importance factor.
        beta: Heuristic (distance) importance factor.
        rho: Pheromone evaporation rate.
        tau_0: Initial pheromone level.
        tau_min: Minimum pheromone bound.
        tau_max: Maximum pheromone bound.
        max_iterations: Maximum number of ACO iterations.
        time_limit: Maximum time in seconds for the solver.
        q0: Probability of exploitation vs exploration (ACS-style).
        k_sparse: Number of nearest neighbors to consider (for k-sparse ACO).
        sequence_length: Length of operator sequence for hyper-heuristic ACO.
        local_search: Whether to apply local search after construction.
        elitist_weight: Weight for elitist pheromone update.
        operators: List of local search operators to use.
        engine: Solver engine to use.
        packing: List of packing algorithm config files.
        route_improvement: List of route improvement operations to apply.
    """

    n_ants: int = 20
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.1
    tau_0: float = 1.0
    tau_min: float = 0.01
    tau_max: float = 10.0
    max_iterations: int = 50
    time_limit: float = 30.0
    q0: float = 0.9
    k_sparse: int = 15
    sequence_length: int = 5
    local_search: bool = True
    elitist_weight: float = 1.0
    operators: List[str] = field(default_factory=lambda: ["swap", "2opt_intra", "relocate", "swap_star", "perturb"])


@dataclass
class VectorALNSConfig:
    """Configuration for Adaptive Large Neighborhood Search (ALNS) expert policy.

    Attributes:
        time_limit: Maximum time in seconds for the solver.
        max_iterations: Maximum number of ALNS iterations.
        start_temp: Initial temperature for simulated annealing acceptance.
        cooling_rate: Rate at which temperature decreases each iteration.
        reaction_factor: Weight update reaction factor for operator scores.
        min_removal: Minimum number of nodes to remove per destroy operation.
        max_removal_pct: Maximum percentage of nodes to remove per destroy operation.
        max_vehicles: Maximum number of vehicles (0 for unlimited).
        engine: Solver engine to use ('custom', 'alns').
        packing: List of packing algorithm config files.
        route_improvement: List of route improvement operations to apply.
    """

    time_limit: float = 60.0
    max_iterations: int = 5000
    start_temp: float = 100.0
    cooling_rate: float = 0.995
    reaction_factor: float = 0.1
    min_removal: int = 1
    max_removal_pct: float = 0.3
    max_vehicles: int = 0


@dataclass
class VectorHGSConfig:
    """Configuration for Hybrid Genetic Search (HGS) expert policy.
    Based on Vidal et al. (2022) - "Hybrid genetic search for the CVRP".

    Attributes:
        time_limit: Maximum time in seconds for the solver (0.0 = unlimited).
        mu: Minimum population size per subpopulation.
        lambda_param: Generation size (number before survivor selection).
        nb_elite: Number of elite individuals to preserve.
        nb_close: Number of close individuals for diversity measurement.
        nb_granular: Granular search parameter for local search.
        target_feasible: Target proportion of feasible solutions.
        n_iterations_no_improvement: Max iterations without improvement.
        mutation_rate: Probability of applying local search to offspring.
        repair_probability: Probability of repairing infeasible offspring.
        crossover_rate: Probability of applying crossover.
        local_search_iterations: Number of local search iterations.
        population_size: Number of individuals in the genetic population.
        n_generations: Number of generations to evolve.
        elite_size: Number of elite individuals to preserve.
        max_iterations: Local-search iterations inside each generation.
        max_vehicles: Maximum number of vehicles (0 for unlimited).
        initial_penalty_capacity: Initial penalty for capacity violations.
        penalty_increase: Multiplier for increasing penalty.
        penalty_decrease: Multiplier for decreasing penalty.
    """

    # Core HGS parameters (Vidal 2022)
    time_limit: float = 0.0
    mu: int = 25
    lambda_param: int = 40
    nb_elite: int = 4
    nb_close: int = 5
    nb_granular: int = 20
    target_feasible: float = 0.2
    n_iterations_no_improvement: int = 20000

    # Genetic operators
    mutation_rate: float = 1.0
    repair_probability: float = 0.5
    crossover_rate: float = 1.0

    # Diversity management

    # Local search
    local_search_iterations: int = 100
    population_size: int = 50
    n_generations: int = 50
    elite_size: int = 10
    max_iterations: int = 50
    max_vehicles: int = 0

    # Penalty management
    initial_penalty_capacity: float = 1.0
    penalty_increase: float = 1.2
    penalty_decrease: float = 0.85


@dataclass
class VectorHGSALNSConfig:
    """Configuration for Hybrid Genetic Search with ALNS Education (HGS-ALNS) expert policy.

    Attributes:
        time_limit: Maximum time in seconds for the solver.
        mu: Size of the genetic population.
        nb_elite: Number of elite individuals to preserve.
        mutation_rate: Probability of mutation.
        alns_education_iterations: Number of ALNS iterations for education phase.
        n_generations: Number of generations to evolve.
        population_size: Number of individuals in the genetic population.
        elite_size: Number of elite individuals educated with ALNS.
        alns_start_temp: Initial temperature for the ALNS education phase.
        alns_cooling_rate: Cooling rate for the ALNS education phase.
        max_vehicles: Maximum number of vehicles (0 for unlimited).
        engine: Solver engine to use.
        packing: List of packing algorithm config files.
        route_improvement: List of route improvement operations to apply.
    """

    time_limit: float = 60.0
    mu: int = 50
    nb_elite: int = 10
    mutation_rate: float = 0.2
    alns_education_iterations: int = 50
    n_generations: int = 100
    population_size: int = 20
    elite_size: int = 5
    alns_start_temp: float = 0.5
    alns_cooling_rate: float = 0.9995
    max_vehicles: int = 0
    crossover_rate: float = 0.8
    engine: str = "custom"


@dataclass
class VectorHVPLConfig:
    """Configuration for Hybrid Volleyball Premier League (HVPL) expert policy.

    Attributes:
        time_limit: Maximum time in seconds for the solver.
        max_iterations: Maximum number of league iterations.
        n_teams: Number of teams (solutions) per instance in the league.
        sub_rate: Rate of substitution for weak teams each iteration.
        aco_iterations: Number of ACO iterations for initialization/substitution.
        alns_iterations: Destroy-repair cycles in each coaching session.
    """

    time_limit: float = 60.0
    max_iterations: int = 50
    n_teams: int = 10
    sub_rate: float = 0.2
    aco_iterations: int = 1
    alns_iterations: int = 100


@dataclass
class VectorILSConfig:
    """Configuration for Iterated Local Search (ILS) expert policy.

    Attributes:
        n_restarts: Number of ILS restarts (perturbation cycles).
        ls_iterations: Iterations for local search within each phase.
        perturbation_strength: Fraction of tour to perturb.
        ls_operator: Local search operator name or dict of {name: prob}.
        perturbation_type: Perturbation method name or dict of {mode: prob}.
        time_limit: Maximum time in seconds for the solver.
        op_probs: Operator selection probabilities.
        perturb_probs: Perturbation method probabilities.
        packing: List of packing algorithm config files.
        route_improvement: List of route improvement operations to apply.
    """

    n_restarts: int = 5
    ls_iterations: int = 50
    perturbation_strength: float = 0.2
    ls_operator: Union[str, Dict[str, float]] = "two_opt"
    perturbation_type: Union[str, Dict[str, float]] = "double_bridge"
    time_limit: float = 30.0
    op_probs: Dict[str, float] = field(
        default_factory=lambda: {
            "two_opt": 0.25,
            "swap": 0.15,
            "relocate": 0.15,
            "two_opt_star": 0.2,
            "swap_star": 0.15,
            "three_opt": 0.1,
        }
    )
    perturb_probs: Dict[str, float] = field(
        default_factory=lambda: {
            "double_bridge": 0.5,
            "shuffle": 0.3,
            "random_swap": 0.2,
        }
    )


@dataclass
class ProfitableNNConfig:
    """Configuration for the profitable nearest-neighbour expert.

    Attributes:
        two_opt_passes: Maximum number of full 2-opt sweeps per tour.
    """

    two_opt_passes: int = 5


@dataclass
class RLSConfig:
    """Configuration for Random Local Search (RLS) policy.

    Attributes:
        n_iterations: Number of search iterations (operator applications).
        op_probs: Dictionary mapping operator names to selection probabilities.
                 Supported keys: 'two_opt', 'swap', 'relocate', 'two_opt_star', 'swap_star', 'three_opt'.
                 Normalized internally if they don't sum to 1.
        time_limit: Maximum time in seconds for the solver.
        packing: List of packing algorithm config files.
        route_improvement: List of route improvement operations to apply.
    """

    n_iterations: int = 100
    op_probs: Dict[str, float] = field(
        default_factory=lambda: {
            "two_opt": 0.25,
            "swap": 0.15,
            "relocate": 0.15,
            "two_opt_star": 0.2,
            "swap_star": 0.15,
            "three_opt": 0.1,
        }
    )
    time_limit: float = 30.0

