"""GLS (Guided Local Search) Policy Adapter.

Attributes:
    GLSPolicy: Policy class for Guided Local Search.

Example:
    >>> from logic.src.configs.policies.gls import GLSConfig
    >>> config = GLSConfig(lambda_param=1.0)
    >>> policy = GLSPolicy(config)
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.gls import GLSConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry

from logic.src.configs.policies.gls import GLSConfig
from .solver import GLSSolver


@GlobalRegistry.register(
    PolicyTag.META_HEURISTIC,
    PolicyTag.TRAJECTORY_BASED,
    PolicyTag.LOCAL_SEARCH,
    PolicyTag.ROUTING,
    PolicyTag.PROFIT_AWARE,
)
@RoutingAlgorithmRegistry.register("gls")
class GLSPolicy(BaseRoutingPolicy):
    """Guided Large Neighborhood Search (G-LNS) policy class.

    Attributes:
        config: Configuration for the policy.
    """

    def __init__(self, config: Optional[Union[GLSConfig, Dict[str, Any]]] = None):
        """Initializes the GLS policy.

        Args:
            config: Configuration source for the Guided Local Search.

        Returns:
            None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Returns the configuration class for GLS.

        Returns:
            The GLSConfig class.
        """
        return GLSConfig

    def _get_config_key(self) -> str:
        """Returns the configuration key for the GLS policy.

        Returns:
            The registry key 'gls'.
        """
        return "gls"

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
        """Execute the Guided Local Search (GLS) metaheuristic solver logic.

        GLS is a metaheuristic that augments the objective function with penalties.

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
        params = GLSConfig(
            lambda_param=float(values.get("lambda_param", 1.0)),
            alpha_param=float(values.get("alpha_param", 0.3)),
            penalty_cycles=int(values.get("penalty_cycles", 1000)),
            n_removal=int(values.get("n_removal", 2)),
            n_llh=int(values.get("n_llh", 6)),
            inner_iterations=int(values.get("inner_iterations", 100)),
            fls_coupling_prob=float(values.get("fls_coupling_prob", 0.8)),
            time_limit=float(values.get("time_limit", 60.0)),
            seed=values.get("seed", 42),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
        )

        solver = GLSSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        return solver.solve()
