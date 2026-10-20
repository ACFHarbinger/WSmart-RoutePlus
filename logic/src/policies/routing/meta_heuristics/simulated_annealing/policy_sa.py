"""Simulated Annealing Policy Adapter.

Attributes:
    SAPolicy: Policy class for Simulated Annealing.

Example:
    >>> config = {"initial_temperature": 100.0, "cooling_rate": 0.99}
    >>> policy = SAPolicy(config)
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.sa import SAConfig
from logic.src.enums import PolicyTag
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry
from logic.src.utils.registry import ALL

from .solver import SASolver


@ALL.register(
    PolicyTag.META_HEURISTIC,
    PolicyTag.TRAJECTORY_BASED,
    PolicyTag.LOCAL_SEARCH,
    PolicyTag.ROUTING,
    PolicyTag.PROFIT_AWARE,
)
@RoutingAlgorithmRegistry.register("sa")
class SAPolicy(BaseRoutingPolicy):
    """Simulated Annealing (SA) Policy.

    Attributes:
        config: Configuration for the policy.
    """

    def __init__(self, config: Optional[Union[Dict[str, Any], Any]] = None):
        """Initializes the SA policy.

        Args:
            config: Configuration source.

        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Returns the configuration class for SA.

        Returns:
            None (dictionary based).
        """
        return SAConfig

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
        """Execute the Simulated Annealing (SA) solver logic.

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
        # 1. Parse parameters
        params = self._effective_config(values)

        # 2. Instantiate core solver
        solver = SASolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )

        # 3. Optimize and return
        best_routes, best_profit, best_cost = solver.solve(initial_solution=None)

        return best_routes, best_profit, best_cost
