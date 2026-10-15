"""FA Policy Adapter.

Adapts the Discrete Firefly Algorithm (FA) solver to the agnostic
BaseRoutingPolicy interface.

Attributes:
    FAPolicy: Policy class for Discrete Firefly Algorithm.

Example:
    >>> from logic.src.configs.policies.fa import FAConfig
    >>> config = FAConfig(pop_size=20)
    >>> policy = FAPolicy(config)
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.fa import FAConfig
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry

from .solver import FASolver


@RoutingAlgorithmRegistry.register("fa")
class FAPolicy(BaseRoutingPolicy):
    """FA policy class.

    Attributes:
        config: Configuration for the policy.
    """

    def __init__(self, config: Optional[Union[FAConfig, Dict[str, Any]]] = None):
        """Initializes the FA policy.

        Args:
            config: Configuration source for the Discrete Firefly Algorithm.

        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Returns the configuration class for FA.

        Returns:
            The FAConfig class.
        """
        return FAConfig

    def _get_config_key(self) -> str:
        """Returns the configuration key for the FA policy.

        Returns:
            The registry key 'fa'.
        """
        return "fa"

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
        """Execute the Discrete Firefly Algorithm (FA) solver logic.

        FA is a nature-inspired metaheuristic based on the behavior of fireflies.

        Args:
            sub_dist_matrix: Symmetric distance matrix.
            sub_wastes: Mapping of local node indices to waste levels.
            capacity: Maximum vehicle collection capacity.
            revenue: Revenue obtained per kilogram of waste.
            cost_unit: Monetary cost incurred per kilometer.
            values: Merged configuration dictionary.
            mandatory_nodes: Local indices of bins that MUST be collected.
            kwargs: Additional context.

        Returns:
            Tuple of (routes, profit, cost).
        """
        params = FAConfig(
            pop_size=int(values.get("pop_size", 20)),
            beta0=float(values.get("beta0", 1.0)),
            gamma=float(values.get("gamma", 0.1)),
            alpha_profit=float(values.get("alpha_profit", 0.5)),
            beta_will=float(values.get("beta_will", 0.3)),
            gamma_cost=float(values.get("gamma_cost", 0.2)),
            alpha_rnd=float(values.get("alpha_rnd", 0.2)),
            n_removal=int(values.get("n_removal", 3)),
            max_iterations=int(values.get("max_iterations", 100)),
            local_search_iterations=int(values.get("local_search_iterations", 500)),
            time_limit=float(values.get("time_limit", 60.0)),
            seed=values.get("seed", 42),
            profit_aware_operators=values.get("profit_aware_operators", False),
        )

        solver = FASolver(
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
