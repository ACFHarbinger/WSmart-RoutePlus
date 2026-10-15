"""
Memetic Algorithm with Tolerance-based Selection (MA-TS) Policy Adapter.

Attributes:
    MemeticAlgorithmToleranceBasedSelectionPolicy: Policy class for MATBS.

Example:
    >>> policy = MemeticAlgorithmToleranceBasedSelectionPolicy()
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.ma_ts import MemeticAlgorithmToleranceBasedSelectionConfig
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry

from logic.src.configs.policies.ma_ts import MemeticAlgorithmToleranceBasedSelectionConfig
from .solver import MemeticAlgorithmToleranceBasedSelectionSolver


@RoutingAlgorithmRegistry.register("ma_ts")
class MemeticAlgorithmToleranceBasedSelectionPolicy(BaseRoutingPolicy):
    """
    Memetic Algorithm Tolerance-based Selection policy class.

    Attributes:
        solver: Internal solver instance.
        params: Algorithm parameters.
    """

    def __init__(self, config: Optional[Union[MemeticAlgorithmToleranceBasedSelectionConfig, Dict[str, Any]]] = None):
        """Initializes the Memetic Algorithm Tolerance-Based Selection policy.

        Args:
            config: Optional configuration source.

        Returns:
            None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Returns the configuration class for MATBS.

        Args:
            None.

        Returns:
            Optional[Type]: The MemeticAlgorithmToleranceBasedSelectionConfig class.
        """
        return MemeticAlgorithmToleranceBasedSelectionConfig

    def _get_config_key(self) -> str:
        """Returns the configuration key for the MATBS policy.

        Args:
            None.

        Returns:
            str: The registry key 'ma_ts'.
        """
        return "ma_ts"

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
        """Execute the Memetic Algorithm with Tolerance-Based Selection (MA-TS) solver logic.

        MA-TS is a memetic algorithm variant that employs a "tolerance"
        threshold during the survivor selection phase. Instead of strictly
        choosing the absolute best individuals, it allows for the selection
        of solutions that fall within a specified performance tolerance of the
        global optimum. This "fuzzy" selection pressure helps maintain
        structural diversity in the population for longer periods, preventing
        stagnation in massive local optima and improving the algorithm's
        ability to escape from locally deceptive regions of the profit surface.

        Args:
            sub_dist_matrix: Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes: Mapping of local node indices to their
                current bin inventory levels.
            capacity: Maximum vehicle collection capacity.
            revenue: Revenue obtained per kilogram of waste collected.
            cost_unit: Monetary cost incurred per kilometer traveled.
            values: Merged configuration dictionary containing
                MA-TS parameters (population_size, tolerance_pct, recombination_rate).
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
        params = MemeticAlgorithmToleranceBasedSelectionConfig(
            population_size=values.get("population_size", 10),
            max_iterations=values.get("max_iterations", 100),
            tolerance_pct=values.get("tolerance_pct", 0.05),
            recombination_rate=values.get("recombination_rate", 0.6),
            perturbation_strength=values.get("perturbation_strength", 2),
            n_removal=values.get("n_removal", 1),
            local_search_iterations=values.get("local_search_iterations", 500),
            time_limit=values.get("time_limit", 60.0),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )

        solver = MemeticAlgorithmToleranceBasedSelectionSolver(
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
