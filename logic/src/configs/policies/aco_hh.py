"""
HH-ACO (Hyper-Heuristic Ant Colony Optimization) configuration.

Attributes:
    HyperHeuristicACOConfig: Attributes for HH-ACO configuration.

Example:
    >>> from configs.policies.aco_hh import HyperHeuristicACOConfig
    >>> config = HyperHeuristicACOConfig()
    >>> config.n_ants
    20
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class HyperHeuristicACOConfig:
    """Configuration for Hyper-Heuristic Ant Colony Optimization (HH-ACO).

    Attributes:
        n_ants: Number of ants in the colony.
        alpha: Pheromone importance factor.
        beta: Heuristic (distance) importance factor.
        rho: Pheromone evaporation rate.
        tau_0: Initial pheromone level.
        Q: Pheromone floor constant added to every improving-journey deposit.
        lambda_val: Base for the visibility exponential when use_dynamic_lambda=False.
        use_dynamic_lambda: If True (default), replace λ^I with exp(I / Z) where Z is
            a running EMA of |improvement|. This achieves scale invariance across
            problem sizes and is the recommended setting for the VRPP. Set to False for
            strict paper fidelity (λ = 1.0001 as base).
        max_iterations: Maximum number of ACO iterations.
        elitism_ratio: Fraction of the swarm teleported to the global best.
        time_limit: Maximum time in seconds for the solver.
        stagnation_limit: Number of iterations without improvement before pheromone update.
        sequence_length: Length of operator sequence for hyper-heuristic ACO.
        local_search: Whether to apply local search after construction.
        local_search_iterations: Number of local search iterations.
        elitist_weight: Weight for elitist pheromone update.
        operators: List of local search operators to use.
        mandatory_selection: List of mandatory strategy config files.
        route_improvement: List of route improvement operations to apply.
    """

    n_ants: int = 20
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.1
    tau_0: float = 1.0
    Q: float = 1.0
    lambda_val: float = 1.0001
    use_dynamic_lambda: bool = True
    max_iterations: int = 50
    elitism_ratio: float = 0.5
    time_limit: float = 30.0
    seed: Optional[int] = None
    stagnation_limit: int = 10
    sequence_length: Optional[int] = None
    local_search: bool = True
    local_search_iterations: int = 500
    elitist_weight: float = 1.0
    operators: Optional[List[str]] = None
    time_weighted_visibility: bool = False
    vrpp: bool = True
    profit_aware_operators: bool = False
    mandatory_selection: Optional[List[MandatorySelectionConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
