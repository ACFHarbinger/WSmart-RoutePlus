"""MA (Memetic Algorithm) Policy Adapter.

Attributes:
    MAPolicy: Policy class for Memetic Algorithm.

Example:
    >>> from logic.src.configs.policies.ma import MAConfig
    >>> config = MAConfig(pop_size=50)
    >>> policy = MAPolicy(config)
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.ma import MAConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry

from logic.src.configs.policies.ma import MAConfig
from .solver import MASolver


@GlobalRegistry.register(
    PolicyTag.META_HEURISTIC,
    PolicyTag.POPULATION_BASED,
    PolicyTag.EVOLUTIONARY_ALGORITHM,
    PolicyTag.MEMETIC_SEARCH,
    PolicyTag.ROUTING,
    PolicyTag.PROFIT_AWARE,
)
@RoutingAlgorithmRegistry.register("ma")
class MAPolicy(BaseRoutingPolicy):
    """Memetic Algorithm (MA) Policy.

    Attributes:
        config: Configuration for the policy.
    """

    def __init__(self, config: Optional[Union[MAConfig, Dict[str, Any]]] = None):
        """Initialize the MAPolicy adapter.

        Args:
            config: Optional configuration source.

        Returns:
            None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Hydra-compliant configuration schema.

        Returns:
            MAConfig class.
        """
        return MAConfig

    def _get_config_key(self) -> str:
        """Identifier for this policy.

        Returns:
            The key 'ma'.
        """
        return "ma"

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
        """Execute the Memetic Algorithm (MA) solver logic.

        MA combines evolutionary global search with local search refinement.

        Args:
            sub_dist_matrix: Symmetric distance matrix.
            sub_wastes: Mapping of local node indices to waste levels.
            capacity: Maximum vehicle collection capacity.
            revenue: Revenue per kilogram of waste.
            cost_unit: Monetary cost per kilometer.
            values: Merged configuration dictionary.
            mandatory_nodes: Local indices of bins that MUST be collected.
            kwargs: Additional context.

        Returns:
            Tuple of (routes, profit, cost).
        """
        # 1. Parameter Extraction & Mapping (See params.py for conceptual mapping)
        params = MAConfig(
            pop_size=int(values.get("pop_size", 30)),
            max_generations=int(values.get("max_generations", 100)),
            crossover_rate=float(values.get("crossover_rate", 0.8)),
            mutation_rate=float(values.get("mutation_rate", 0.1)),
            local_search_rate=float(values.get("local_search_rate", 1.0)),
            tournament_size=int(values.get("tournament_size", 3)),
            n_removal=int(values.get("n_removal", 2)),
            time_limit=float(values.get("time_limit", 60.0)),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )

        # 2. Solver Initialization
        solver = MASolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )

        # 3. Evolutionary Optimization (Moscato FIG 3.1)
        # Returns the best individual and its fitness (profit).
        best_routes, best_reward = solver.solve()

        # 4. Result Formatting & Supplemental Metrics
        # The simulator Expects (Routes, NetProfit, AbsoluteDistanceCost).
        total_cost = solver._cost(best_routes) * cost_unit

        return best_routes, best_reward, total_cost
