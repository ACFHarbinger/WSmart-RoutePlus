"""
HVPL Policy Adapter.

Adapts the Hybrid Volleyball Premier League (HVPL) logic to the agnostic interface.

Reference:
    Sun, S., Ma, L., Liu, Y., & Wang, L. (2023). "Volleyball premier league
    algorithm with ACO and ALNS for simultaneous pickup–delivery location
    routing problem."

Attributes:
    HVPLPolicy: Policy adapter for the HVPL metaheuristic.

Example:
    >>> policy = HVPLPolicy()
    >>> routes, profit, cost = policy(obs)
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.hvpl import HVPLConfig
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry
from logic.src.policies.routing.hyper_heuristics.ant_colony_optimization_hyper_heuristic.params import (
    OPERATOR_NAMES,
    HyperACOParams,
)
from logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search.params import ALNSParams
from logic.src.policies.routing.meta_heuristics.hybrid_volleyball_premier_league.params import HVPLParams
from logic.src.policies.routing.meta_heuristics.hybrid_volleyball_premier_league.solver import HVPLSolver


@RoutingAlgorithmRegistry.register("hvpl")
class HVPLPolicy(BaseRoutingPolicy):
    """
    HVPL policy class.

    Visits pre-selected 'mandatory' bins using the population-based HVPL metaheuristic.

    Attributes:
        config: Configuration parameters for the policy.
    """

    def __init__(self, config: Optional[Union[HVPLConfig, Dict[str, Any]]] = None):
        """Initialize HVPL policy with optional config.

        Args:
            config: HVPLConfig dataclass, raw dict from YAML, or None.

        Returns:
            None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the configuration class for HVPL.

        Args:
            None.

        Returns:
            Optional[Type]: The HVPLConfig class.
        """
        return HVPLConfig

    def _get_config_key(self) -> str:
        """Return config key for HVPL.

        Args:
            None.

        Returns:
            str: "hvpl".
        """
        return "hvpl"

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
        """Execute the Hybrid Volleyball Premier League (HVPL) solver logic.

        HVPL is a population-based metaheuristic inspired by the competition and
        substitution dynamics in a volleyball league. In this implementation:
        - Initialization: ACO-driven population formation.
        - Competition Phase: Teams (solutions) participate in matches where better
          performing team "brightness" or tactical strength influences others.
        - Substitution & Coaching: Mechanisms for replacing weak components of a
          team with stronger tactics derived from ACO or ALNS successes.

        Args:
            sub_dist_matrix: Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes: Mapping of local node indices to their
                current bin inventory levels.
            capacity: Maximum vehicle collection capacity.
            revenue: Revenue obtained per kilogram of waste collected.
            cost_unit: Monetary cost incurred per kilometer traveled.
            values: Merged configuration dictionary containing
                HVPL parameters and nested ACO/ALNS configs.
            mandatory_nodes: Local indices of bins that MUST be
                collected in this period.
            kwargs: Additional context, including:
                - search_context (Optional[SearchContext]): Context for tracking
                  recursive solver statistics.
                - multi_day_context (Optional[MultiDayContext]): Context for
                  inter-day state propagation.

        Returns:
            Tuple[List[List[int]], float, float]: A 3-tuple containing:
                - routes: Optimized collection routes (list-of-lists, local indices).
                - profit: Total calculated net profit (Total Revenue - Total Cost).
                - cost: Total travel cost calculated by the solver.
        """
        seed = values.get("seed", 42)
        vrpp = values.get("vrpp", True)
        profit_aware_operators = values.get("profit_aware_operators", False)

        # Get ACO config (nested or flattened)
        aco_config = values.get("aco", {})
        if isinstance(aco_config, dict):
            aco_params = HyperACOParams(
                n_ants=aco_config.get("n_ants", 10),
                alpha=aco_config.get("alpha", 1.0),
                beta=aco_config.get("beta", 2.0),
                rho=aco_config.get("rho", 0.5),
                eta_decay=aco_config.get("eta_decay", 0.5),
                tau_0=aco_config.get("tau_0", 1.0),
                Q=aco_config.get("Q", 1.0),
                lambda_val=aco_config.get("lambda_val", 1.0001),
                use_dynamic_lambda=aco_config.get("use_dynamic_lambda", True),
                max_iterations=aco_config.get("max_iterations", 50),
                elitism_ratio=aco_config.get("elitism_ratio", 1.0),
                time_limit=aco_config.get("time_limit", 30.0),
                stagnation_limit=aco_config.get("stagnation_limit", 10),
                operators=aco_config.get("operators", OPERATOR_NAMES.copy()),
                seed=seed,
                vrpp=vrpp,
                profit_aware_operators=profit_aware_operators,
            )
        else:
            # Fallback to flattened parameters for backward compatibility
            aco_params = HyperACOParams(
                n_ants=values.get("n_ants", 10),
                alpha=values.get("alpha", 1.0),
                beta=values.get("beta", 2.0),
                rho=values.get("rho", 0.5),
                eta_decay=values.get("eta_decay", 0.5),
                tau_0=values.get("tau_0", 1.0),
                Q=values.get("Q", 1.0),
                lambda_val=values.get("lambda_val", 1.0001),
                use_dynamic_lambda=values.get("use_dynamic_lambda", True),
                max_iterations=values.get("max_iterations", 50),
                elitism_ratio=values.get("elitism_ratio", 1.0),
                time_limit=values.get("time_limit", 30.0),
                stagnation_limit=values.get("stagnation_limit", 10),
                operators=values.get("operators", OPERATOR_NAMES.copy()),
                seed=seed,
                vrpp=vrpp,
                profit_aware_operators=profit_aware_operators,
            )

        # Get ALNS config (nested or flattened)
        alns_config = values.get("alns", {})
        if isinstance(alns_config, dict):
            alns_params = ALNSParams(
                max_iterations=alns_config.get("max_iterations", 500),
                start_temp=alns_config.get("start_temp", 100.0),
                cooling_rate=alns_config.get("cooling_rate", 0.95),
                reaction_factor=alns_config.get("reaction_factor", 0.1),
                min_removal=alns_config.get("min_removal", 1),
                time_limit=alns_config.get("time_limit", 60.0),
                start_temp_control=alns_config.get("start_temp_control", 100.0),
                xi=alns_config.get("xi", 1.0),
                segment_size=alns_config.get("segment_size", 10),
                seed=seed,
                vrpp=vrpp,
                profit_aware_operators=profit_aware_operators,
            )
        else:
            # Fallback to flattened parameters for backward compatibility
            alns_params = ALNSParams(
                max_iterations=values.get("alns_iterations", 500),
                start_temp=values.get("alns_start_temp", 100.0),
                cooling_rate=values.get("alns_cooling_rate", 0.95),
                reaction_factor=values.get("alns_reaction_factor", 0.1),
                min_removal=values.get("alns_min_removal", 1),
                start_temp_control=values.get("alns_start_temp_control", 100.0),
                xi=values.get("alns_xi", 1.0),
                segment_size=values.get("alns_segment_size", 10),
                time_limit=values.get("alns_time_limit", 60.0),
                seed=seed,
                vrpp=vrpp,
                profit_aware_operators=profit_aware_operators,
            )

        params = HVPLParams(
            n_teams=values.get("n_teams", 30),
            max_iterations=values.get("max_iterations", 100),
            substitution_rate=values.get("substitution_rate", 0.2),
            crossover_rate=values.get("crossover_rate", 0.8),
            mutation_rate=values.get("mutation_rate", 0.1),
            elite_size=values.get("elite_size", 3),
            aco_init_iterations=values.get("aco_init_iterations", 50),
            time_limit=values.get("time_limit", 300.0),
            aco_params=aco_params,
            alns_params=alns_params,
            seed=seed,
            vrpp=vrpp,
            profit_aware_operators=profit_aware_operators,
        )

        solver = HVPLSolver(
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
