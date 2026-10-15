"""ILS (Iterated Local Search) Policy Adapter.

Attributes:
    ILSPolicy: Policy class for Iterated Local Search.

Example:
    >>> from logic.src.configs.policies.ils import ILSConfig
    >>> config = ILSConfig(n_restarts=50)
    >>> policy = ILSPolicy(config)
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.ils import ILSConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry

from logic.src.configs.policies.ils import ILSConfig
from .solver import ILSSolver


@GlobalRegistry.register(
    PolicyTag.META_HEURISTIC,
    PolicyTag.TRAJECTORY_BASED,
    PolicyTag.LOCAL_SEARCH,
    PolicyTag.ROUTING,
    PolicyTag.PROFIT_AWARE,
)
@RoutingAlgorithmRegistry.register("ils")
class ILSPolicy(BaseRoutingPolicy):
    """Iterated Local Search policy class.

    Attributes:
        config: Configuration for the policy.
    """

    def __init__(self, config: Optional[Union[ILSConfig, Dict[str, Any]]] = None):
        """Initializes the ILS policy.

        Args:
            config: Optional configuration dictionary.

        Returns:
            None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Returns the config class.

        Returns:
            The ILSConfig class.
        """
        return ILSConfig

    def _get_config_key(self) -> str:
        """Returns the config key.

        Returns:
            The key 'ils'.
        """
        return "ils"

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
        """Execute the Iterated Local Search (ILS) metaheuristic solver logic.

        ILS iteratively applies a local search to a perturbed solution.

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
        params = ILSConfig(
            n_restarts=int(values.get("n_restarts", 30)),
            inner_iterations=int(values.get("inner_iterations", 20)),
            n_removal=int(values.get("n_removal", 2)),
            n_llh=int(values.get("n_llh", 5)),
            perturbation_strength=float(values.get("perturbation_strength", 0.15)),
            time_limit=float(values.get("time_limit", 60.0)),
            seed=int(values.get("seed", 42)),
            profit_aware_operators=bool(values.get("profit_aware_operators", False)),
        )

        solver = ILSSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        return solver.solve()
