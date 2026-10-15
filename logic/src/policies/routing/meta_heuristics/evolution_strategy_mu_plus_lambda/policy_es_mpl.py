r"""(μ+λ) Evolution Strategy Policy Adapter.

Adapts the rigorous (μ+λ)-ES implementation into the overarching policy registry.

Attributes:
    MuPlusLambdaESPolicy: Policy class for (μ+λ)-ES.

Example:
    >>> policy = MuPlusLambdaESPolicy()
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import MuPlusLambdaESConfig
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry

from logic.src.configs.policies.es_mpl import MuPlusLambdaESConfig
from .solver import MuPlusLambdaESSolver


@RoutingAlgorithmRegistry.register("es_mpl")
class MuPlusLambdaESPolicy(BaseRoutingPolicy):
    """(μ+λ) Evolution Strategy policy class.

    Attributes:
        config: Configuration for the policy.
    """

    def __init__(self, config: Optional[Union[MuPlusLambdaESConfig, Dict[str, Any]]] = None):
        """Initializes the (μ+λ)-ES policy with optional configuration.

        Args:
            config: Configuration object or dictionary.

        Returns:
            None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the configuration class for this policy.

        Returns:
            MuPlusLambdaESConfig class.
        """
        return MuPlusLambdaESConfig

    def _get_config_key(self) -> str:
        """Returns the configuration key for the (μ+λ)-ES policy.

        Returns:
            The string key "es_mpl".
        """
        return "es_mpl"

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
        """Execute the (mu + lambda) Evolution Strategy (ES) solver logic.

        (mu + lambda) ES is an elitist evolutionary algorithm:
        - mu: The number of parent individuals.
        - lambda: The number of offspring generated.
        - Selection: Selected from the union of mu parents and lambda offspring.

        Args:
            sub_dist_matrix: Symmetric distance matrix.
            sub_wastes: Mapping of local node indices to waste quantities.
            capacity: Maximum vehicle collection capacity.
            revenue: Revenue obtained per kilogram of waste.
            cost_unit: Monetary cost incurred per kilometer.
            values: Merged configuration dictionary.
            mandatory_nodes: Local indices of bins that MUST be collected.
            kwargs: Additional context.

        Returns:
            Tuple of (routes, profit, cost).
        """
        params = MuPlusLambdaESConfig(
            mu=values.get("mu", 10),
            lambda_=values.get("lambda_", 5),
            n_removal=values.get("n_removal", 3),
            max_iterations=values.get("max_iterations", 500),
            local_search_iterations=values.get("local_search_iterations", 100),
            time_limit=values.get("time_limit", 60.0),
            seed=values.get("seed", 42),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
        )

        solver = MuPlusLambdaESSolver(
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
