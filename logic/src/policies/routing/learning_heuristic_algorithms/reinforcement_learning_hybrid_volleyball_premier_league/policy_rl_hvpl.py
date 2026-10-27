"""
RL-HVPL Policy Adapter.

Adapts the Reinforcement Learning Hybrid Volleyball Premier League logic
to the agnostic policy interface.

Attributes:
    RLHVPLPolicy: RLHVPL policy class.

Examples:
    >>> rl_hvpl_policy = RLHVPLPolicy()
    >>> routes, profit, cost = rl_hvpl_policy.run(sub_dist_matrix, sub_wastes, capacity, revenue, cost_unit, values)
    >>> print(routes)
    [[1, 2, 3], [4, 5, 6]]
    >>> print(profit)
    1000.0
    >>> print(cost)
    100.0
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np

from logic.src.configs.policies.acceptance_criteria import AcceptanceConfig, BoltzmannAcceptanceConfig
from logic.src.configs.policies.helpers.learning.reinforcement_learning import RLConfig
from logic.src.configs.policies.routing.learning_heuristic_algorithms.rl_hvpl import RLHVPLConfig
from logic.src.configs.policies.routing.meta_heuristics.aco_ks import KSparseACOConfig
from logic.src.configs.policies.routing.meta_heuristics.alns import ALNSConfig
from logic.src.enums import PolicyTag
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry
from logic.src.policies.routing.learning_heuristic_algorithms.reinforcement_learning_hybrid_volleyball_premier_league.rl_hvpl import (
    RLHVPLSolver,
)
from logic.src.utils.registry import ALL


@ALL.register(
    PolicyTag.REINFORCEMENT_LEARNING,
    PolicyTag.META_HEURISTIC,
    PolicyTag.POPULATION_BASED,
    PolicyTag.ROUTING,
    PolicyTag.PROFIT_AWARE,
)
@RoutingAlgorithmRegistry.register("rl_hvpl")
class RLHVPLPolicy(BaseRoutingPolicy):
    """
    RL-HVPL policy class.

    Visits pre-selected 'mandatory' bins using the population-based RL-HVPL metaheuristic.

    Attributes:
        config: RLHVPLConfig dataclass, raw dict from YAML, or None.
    """

    def __init__(self, config: Optional[Union[RLHVPLConfig, Dict[str, Any]]] = None):
        """Initialize RL-HVPL policy with optional config.

        Args:
            config: RLHVPLConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return config class for RL-HVPL.

        Returns:
            Optional[Type]: Config class for RL-HVPL.
        """
        return RLHVPLConfig

    def _get_config_key(self) -> str:
        """Return config key for RL-HVPL.

        Returns:
            str: Config key for RL-HVPL.
        """
        return "rl_hvpl"

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """
        Execute the Reinforcement Learning Hybrid Volleyball Premier League (RL-HVPL)
        solver logic.

        RL-HVPL is a population-based metaheuristic inspired by the management of
        a volleyball league. It maintains a population of solutions ("teams")
        that evolve through "matches" (operator applications from ACO or ALNS).
        A reinforcement learning agent manages the transformation of teams,
        evaluating their relative strengths and applying coaching or
        substitutions to the worst-performing solution to drive global
        convergence towards higher profits.

        Args:
            sub_dist_matrix: Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes: Mapping of local node indices to their
                current bin inventory levels.
            capacity: Maximum vehicle collection capacity.
            revenue: Revenue obtained per kilogram of waste collected.
            cost_unit: Monetary cost incurred per kilometer traveled.
            values: Merged configuration dictionary containing
                HVPL parameters, nested ACO parameters, and ALNS settings.
            mandatory_nodes: Local indices of bins that MUST be
                collected in this period.
            kwargs: Additional context, including:
                - search_context (Optional[SearchContext]): Context for tracking
                  recursive solver statistics.
                - multi_day_context (Optional[MultiDayContext]): Context for
                  inter-day state propagation.

        Returns:
            Tuple[List[List[int]], float, float]: A 3-tuple containing:
                - routes: Optimized collection routes for the current day.
                - profit: Total calculated net profit (Total Revenue - Total Cost).
                - cost: Total travel cost calculated by the solver.
        """
        seed = values.get("seed", 42)
        profit_aware_operators = values.get("profit_aware_operators", False)

        # Extract sub-params for ACO and ALNS
        aco_params = KSparseACOConfig(
            n_ants=values.get("aco_n_ants", 10),
            k_sparse=values.get("aco_k_sparse", 10),
            alpha=values.get("aco_alpha", 1.0),
            beta=values.get("aco_beta", 2.0),
            rho=values.get("aco_rho", 0.1),
            scale=values.get("aco_scale", 5.0),
            tau_0=values.get("aco_tau_0"),
            tau_min=values.get("aco_tau_min", 0.001),
            tau_max=values.get("aco_tau_max", 10.0),
            max_iterations=values.get("aco_iterations", 1),
            time_limit=values.get("aco_time_limit", 60.0),
            local_search=values.get("aco_local_search", True),
            local_search_iterations=values.get("aco_local_search_iterations", 50),
            elitist_weight=values.get("aco_elitist_weight", 1.0),
            profit_aware_operators=profit_aware_operators,
            seed=seed,
        )

        alns_params = cast(Any, ALNSConfig)(
            max_iterations=values.get("alns_iterations", 200),
            acceptance_criterion=AcceptanceConfig(
                method="bmc",
                params=BoltzmannAcceptanceConfig(
                    initial_temp=values.get("alns_start_temp", 100.0),
                    alpha=values.get("alns_cooling_rate", 0.97),
                    start_temp_control=0.05,
                ),
            ),
            reaction_factor=values.get("alns_reaction_factor", 0.5),
            max_removal_pct=values.get("alns_max_removal_pct", 0.3),
            time_limit=values.get("alns_time_limit", 60.0),
            profit_aware_operators=profit_aware_operators,
            seed=seed,
        )

        # RLConfig extraction (assuming it comes nested if mapped cleanly, but
        # handle flattened kwargs or a sub-dict correctly)
        rl_config = values.get("rl_config")
        if rl_config is None:
            rl_config = RLConfig()
        elif isinstance(rl_config, dict):
            # Try to build RLConfig from dict if passed organically
            rl_config = RLConfig(**rl_config)

        params = RLHVPLConfig(
            n_teams=values.get("n_teams", 10),
            max_iterations=values.get("max_iterations", 100),
            sub_rate=values.get("sub_rate", 0.2),
            time_limit=values.get("time_limit", 60.0),
            aco_params=aco_params,
            alns_params=alns_params,
            rl_config=rl_config,
            pheromone_update_strategy=values.get("pheromone_update_strategy", "profit"),
            profit_weight=values.get("profit_weight", 1.0),
            elite_coaching_iterations=values.get("elite_coaching_iterations", 300),
            regular_coaching_iterations=values.get("regular_coaching_iterations", 100),
            elite_size=values.get("elite_size", 3),
            seed=seed,
            profit_aware_operators=profit_aware_operators,
        )

        solver = RLHVPLSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        routes, profit, solver_cost = solver.solve()
        return routes, profit, solver_cost
