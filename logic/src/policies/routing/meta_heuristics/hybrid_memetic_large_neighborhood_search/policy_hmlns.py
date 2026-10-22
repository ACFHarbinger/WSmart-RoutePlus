"""
Hybrid Memetic Large Neighborhood Search (HMLNS) Policy Adapter.

Adapts the rigorous Hybrid Memetic Large Neighborhood Search (replaces HVPL).

Attributes:
    HybridMemeticLargeNeighborhoodSearchPolicy: Policy adapter for the HMLNS solver.

Example:
    >>> policy = HybridMemeticLargeNeighborhoodSearchPolicy(config)
    >>> routes, profit, cost = policy.run(state)
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import HybridMemeticLargeNeighborhoodSearchConfig
from logic.src.configs.policies.routing.meta_heuristics.alns import ALNSConfig
from logic.src.configs.policies.acceptance_criteria import AcceptanceConfig, BoltzmannAcceptanceConfig
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry

from .params import HybridMemeticLargeNeighborhoodSearchParams, MACOParams
from .solver import HybridMemeticLargeNeighborhoodSearchSolver


@RoutingAlgorithmRegistry.register("hmlns")
class HybridMemeticLargeNeighborhoodSearchPolicy(BaseRoutingPolicy):
    """
    Hybrid Memetic Large Neighborhood Search policy class.

    Multi-phase hybrid solver (MACO + GA + ALNS). Replaces HVPL.

    Attributes:
        config: Policy configuration.
    """

    def __init__(self, config: Optional[Union[HybridMemeticLargeNeighborhoodSearchConfig, Dict[str, Any]]] = None):
        """Initialize HMLNS policy with optional config.

        Args:
            config: HybridMemeticLargeNeighborhoodSearchConfig dataclass, raw dict from YAML, or None.

        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the configuration class for this policy.

        Args:
            None.

        Returns:
            Optional[Type]: HybridMemeticLargeNeighborhoodSearchConfig class.
        """
        return HybridMemeticLargeNeighborhoodSearchConfig

    def _get_config_key(self) -> str:
        """Return config key.

        Args:
            None.

        Returns:
            str: "hmlns".
        """
        return "hmlns"

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
        Execute the Hybrid Memetic Large Neighborhood Search (HMLNS) solver logic.

        HMLNS is a multi-phase hybrid metaheuristic that coordinates several
        optimization paradigms:
        - MACO Initialization: Generates a high-quality initial population.
        - Genetic Algorithm (GA) Core: Evolves the population through
          selection, crossover, and mutation.
        - ALNS Education: Refines offspring and elite solutions using adaptive
          local search operators.
        Replaces the earlier HVPL strategy with a more robust memetic framework
        governing the transitions between construction, evolution, and
        intensification phases.

        Args:
            sub_dist_matrix: Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes: Mapping of local node indices to their
                current bin inventory levels.
            capacity: Maximum vehicle collection capacity.
            revenue: Revenue obtained per kilogram of waste collected.
            cost_unit: Monetary cost incurred per kilometer traveled.
            values: Merged configuration dictionary containing
                HMLNS parameters and nested MACO/ALNS configs.
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
        profit_aware_operators = values.get("profit_aware_operators", False)

        # Get MACO config
        maco_config = values.get("maco", values.get("aco", {})) or {}
        aco_params = MACOParams(
            n_ants=maco_config.get("n_ants", 20),
            alpha=maco_config.get("alpha", 1.0),
            beta=maco_config.get("beta", 2.0),
            rho=maco_config.get("rho", 0.1),
            tau_0=maco_config.get("tau_0", 1.0),
            tau_min=maco_config.get("tau_min", 0.001),
            tau_max=maco_config.get("tau_max", 10.0),
            max_iterations=maco_config.get("max_iterations", 1),
            time_limit=maco_config.get("time_limit", 60.0),
            local_search=maco_config.get("local_search", False),
            local_search_iterations=maco_config.get("local_search_iterations", 0),
            elitist_weight=maco_config.get("elitist_weight", 1.0),
            seed=seed,
            profit_aware_operators=profit_aware_operators,
        )

        # Get ALNS config
        alns_config = values.get("alns") or {}
        alns_params = ALNSConfig(
            max_iterations=alns_config.get("max_iterations", 500),
            acceptance_criterion=AcceptanceConfig(
                method="bmc",
                params=BoltzmannAcceptanceConfig(
                    initial_temp=alns_config.get("start_temp", 100.0),
                    alpha=alns_config.get("cooling_rate", 0.95),
                    start_temp_control=0.05,
                ),
            ),
            reaction_factor=alns_config.get("reaction_factor", 0.1),
            max_removal_pct=alns_config.get("max_removal_pct", 0.2),
            segment_size=alns_config.get("segment_size", 10),
            time_limit=alns_config.get("time_limit", 60.0),
            seed=seed,
            profit_aware_operators=profit_aware_operators,
        )

        params = HybridMemeticLargeNeighborhoodSearchParams(
            population_size=values.get("population_size", 30),
            max_generations=values.get("max_generations", 50),
            substitution_rate=values.get("substitution_rate", 0.2),
            crossover_rate=values.get("crossover_rate", 0.8),
            mutation_rate=values.get("mutation_rate", 0.1),
            elitism_count=values.get("elitism_count", 3),
            aco_init_iterations=values.get("aco_init_iterations", 50),
            time_limit=values.get("time_limit", 300.0),
            aco_params=aco_params,
            alns_params=alns_params,
            seed=seed,
            profit_aware_operators=profit_aware_operators,
        )

        solver = HybridMemeticLargeNeighborhoodSearchSolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )

        routes, profit, cost = solver.solve()
        return routes, profit, cost
