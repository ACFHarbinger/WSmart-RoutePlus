r"""(μ,λ) Evolution Strategy Policy Adapter.

Adapts the rigorous (μ,λ)-ES implementation into the overarching policy registry.

Attributes:
    MuCommaLambdaESPolicy: Policy class for (μ,λ)-ES.

Example:
    >>> policy = MuCommaLambdaESPolicy()
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import MuCommaLambdaESConfig
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry

from .params import MuCommaLambdaESParams
from .solver import MuCommaLambdaESSolver


@RoutingAlgorithmRegistry.register("es_mcl")
class MuCommaLambdaESPolicy(BaseRoutingPolicy):
    """(μ,λ) Evolution Strategy policy class.

    Executes a strict generational evolutionary algorithm with truncation selection.

    Attributes:
        config: Configuration for the policy.
    """

    def __init__(self, config: Optional[Union[MuCommaLambdaESConfig, Dict[str, Any]]] = None):
        """Initialize (μ,λ)-ES policy with optional config.

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
            MuCommaLambdaESConfig class.
        """
        return MuCommaLambdaESConfig

    def _get_config_key(self) -> str:
        """Return the configuration key for this policy.

        Returns:
            The string key "es_mcl".
        """
        return "es_mcl"

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
        """Execute the (mu, lambda) Evolution Strategy (ES) solver logic.

        (mu, lambda)-ES is a rigorous generational evolutionary algorithm:
        - mu: The number of parents selected to produce the next generation.
        - lambda: The number of offspring generated from the parents.
        - Selection: Only the lambda offspring are considered for the next
          generation (the parents are discarded), using truncation selection
          (best mu out of lambda).

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
        # Map configuration dictionary to strict ES parameters
        params = MuCommaLambdaESParams(
            mu=values.get("mu", 15),
            lambda_=values.get("lambda_", 100),
            n_removal=values.get("n_removal", 3),
            max_iterations=values.get("max_iterations", 500),
            local_search_iterations=values.get("local_search_iterations", 100),
            time_limit=values.get("time_limit", 60.0),
            seed=values.get("seed", 42),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
        )

        solver = MuCommaLambdaESSolver(
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
