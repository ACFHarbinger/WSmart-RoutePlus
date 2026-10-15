"""
RL-HVPL (Reinforcement Learning Hybrid Volleyball Premier League) configuration for Hydra.

Attributes:
    RLHVPLConfig: Configuration for the RL-HVPL policy.

Example:
    >>> from configs.policies.rl_hvpl import RLHVPLConfig
    >>> config = RLHVPLConfig()
    >>> config.n_teams
    10
    >>> config.time_limit
    60.0
    >>> config.max_iterations
    100
    >>> config.profit_aware_operators
    False
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from logic.src.configs.policies.aco_ks import KSparseACOConfig
from logic.src.configs.policies.alns import ALNSConfig
from logic.src.configs.policies.other import RLConfig


@dataclass
class RLHVPLConfig:
    """
    Configuration for the Reinforcement Learning Hybrid Volleyball Premier League policy.

    Combines VPL population dynamics, ACO initialization with Q-Learning,
    and ALNS local search with SARSA for adaptive operator selection.

    Attributes:
        n_teams: Number of teams in the VPL population.
        time_limit: Time limit for the algorithm in seconds.
        max_iterations: Maximum number of iterations.
        elite_coaching_iterations: Maximum number of iterations for elite coaching.
        regular_coaching_iterations: Maximum number of iterations for regular coaching.
        elite_size: Size of the elite set.
        sub_rate: Rate of subproblem solving.
        pheromone_update_strategy: Strategy for updating pheromones.
        profit_weight: Weight for profit in pheromone updates.
        aco: Configuration for ACO initialization.
        alns: Configuration for ALNS local search.
        rl_config: Configuration for the RL policy.
        seed: Random seed.
        profit_aware_operators: Whether to use profit-aware operators.
        packing: List of mandatory nodes.
        route_improvement: List of route improvements.
    """

    # General parameters
    n_teams: int = 10
    time_limit: float = 60.0
    max_iterations: int = 100
    elite_coaching_iterations: int = 300
    regular_coaching_iterations: int = 100
    elite_size: int = 3
    sub_rate: float = 0.2
    pheromone_update_strategy: str = "profit"
    profit_weight: float = 1.0

    # Nested component configs
    aco: KSparseACOConfig = field(default_factory=KSparseACOConfig)
    alns: ALNSConfig = field(default_factory=ALNSConfig)

    # RL Configuration (Centralized)
    rl_config: RLConfig = field(default_factory=RLConfig)

    # Common policy fields
    seed: Optional[int] = None
    profit_aware_operators: bool = False
    packing: List[str] = field(default_factory=list)
    route_improvement: List[Any] = field(default_factory=list)
    aco_params: KSparseACOConfig = field(
        default_factory=lambda: KSparseACOConfig(
            n_ants=10,
            k_sparse=10,
            alpha=1.0,
            beta=2.0,
            rho=0.1,
            scale=5.0,
            tau_0=None,
            tau_min=0.001,
            tau_max=10.0,
            max_iterations=1,  # Single construction per call
            time_limit=10.0,
            local_search=True,
            local_search_iterations=50,
            elitist_weight=1.0,
        )
    )
    alns_params: ALNSConfig = field(
        default_factory=lambda: ALNSConfig(
            max_iterations=200,
            start_temp=100.0,
            cooling_rate=0.97,
            reaction_factor=0.5,
            max_removal_pct=0.3,
            time_limit=20.0,
        )
    )
