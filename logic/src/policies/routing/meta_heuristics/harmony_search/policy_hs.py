"""HS Policy Adapter.

Attributes:
    HSPolicy: Policy class for Harmony Search.

Example:
    >>> from logic.src.configs.policies.hs import HSConfig
    >>> config = HSConfig(hm_size=20)
    >>> policy = HSPolicy(config)
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.hs import HSConfig
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry

from .solver import HSSolver


@RoutingAlgorithmRegistry.register("hs")
class HSPolicy(BaseRoutingPolicy):
    """Harmony Search (HS) Policy - Music-Inspired Global Optimization.

    Attributes:
        config: Configuration for the policy.
    """

    def __init__(self, config: Optional[Union[HSConfig, Dict[str, Any]]] = None):
        """Initializes the Harmony Search policy.

        Args:
            config: Optional configuration dictionary.

        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Returns the config class.

        Returns:
            The HSConfig class.
        """
        return HSConfig

    def _get_config_key(self) -> str:
        """Returns the config key.

        Returns:
            The key 'hs'.
        """
        return "hs"

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
        """Execute the Harmony Search (HS) metaheuristic solver logic.

        HS is a music-inspired metaheuristic based on searching for harmony.

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
        params = HSConfig(
            hm_size=int(values.get("hm_size", 10)),
            HMCR=float(values.get("HMCR", 0.9)),
            PAR=float(values.get("PAR", 0.3)),
            BW=float(values.get("BW", 0.05)),
            max_iterations=int(values.get("max_iterations", 500)),
            local_search_iterations=int(values.get("local_search_iterations", 500)),
            time_limit=float(values.get("time_limit", 60.0)),
            seed=values.get("seed", 42),
            profit_aware_operators=values.get("profit_aware_operators", False),
        )

        solver = HSSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        routes, profit, cost = solver.solve()
        return routes, profit, cost
