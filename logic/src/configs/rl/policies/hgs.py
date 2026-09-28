"""
HGS (Hybrid Genetic Search) configuration for expert policy training.

Attributes:
    HGSConfig: Configuration for HGS algorithm.

Example:
    hgs_config = HGSConfig(
        time_limit=60.0,
        mu=50,
        nb_elite=10,
        mutation_rate=0.2,
        alns_education_iterations=50,
        n_generations=100,
        max_vehicles=0,
        engine="custom",
        mandatory_selection=[],
        route_improvement=[],
    )
"""

from dataclasses import dataclass


@dataclass
class HGSConfig:
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
    max_vehicles: int = 0

    # Penalty management
    initial_penalty_capacity: float = 1.0
    penalty_increase: float = 1.2
    penalty_decrease: float = 0.85
