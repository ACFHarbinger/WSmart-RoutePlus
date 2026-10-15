"""FILO Policy Adapter.

Adapts the Fast Iterative Localized Optimization (FILO) logic to the agnostic interface.

Attributes:
    FILOPolicy: Policy class for FILO.

Example:
    >>> policy = FILOPolicy()
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import FILOConfig
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry

from .filo import FILOSolver
from logic.src.configs.policies.filo import FILOConfig


@RoutingAlgorithmRegistry.register("filo")
class FILOPolicy(BaseRoutingPolicy):
    """FILO policy class.

    Attributes:
        config: Configuration for the policy.
    """

    def __init__(self, config: Optional[Union[FILOConfig, Dict[str, Any]]] = None):
        """Initialize FILO policy with optional config.

        Args:
            config: FILOConfig dataclass, raw dict, or None.

        Returns:
            None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the configuration class for this policy.

        Returns:
            FILOConfig class.
        """
        return FILOConfig

    def _get_config_key(self) -> str:
        """Returns the configuration key for the FILO policy.

        Returns:
            The registry key 'filo'.
        """
        return "filo"

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
        """Execute the Fast Iterative Localized Optimization (FILO) solver logic.

        FILO is a high-performance metaheuristic designed for large-scale VRPs.

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
        # Build parameters from config
        params = FILOConfig(
            time_limit=float(values.get("time_limit", 60.0)),
            max_iterations=int(values.get("max_iterations", 100)),
            initial_temperature_factor=float(values.get("initial_temperature_factor", 10.0)),
            final_temperature_factor=float(values.get("final_temperature_factor", 100.0)),
            shaking_lb_factor=float(values.get("shaking_lb_factor", 0.5)),
            shaking_ub_factor=float(values.get("shaking_ub_factor", 2.0)),
            shaking_lb_intensity=float(values.get("shaking_lb_intensity", 0.1)),
            shaking_ub_intensity=float(values.get("shaking_ub_intensity", 0.5)),
            delta_gamma=float(values.get("delta_gamma", 0.1)),
            gamma_base=float(values.get("gamma_base", 1.0)),
            gamma_lambda=float(values.get("gamma_lambda", 2.0)),
            omega_base_multiplier=float(values.get("omega_base_multiplier", 1.0)),
            local_search_iterations=int(values.get("local_search_iterations", 500)),
            n_cw=int(values.get("n_cw", 100)),
            svc_size=int(values.get("svc_size", 50)),
            seed=values.get("seed", 42),
            profit_aware_operators=values.get("profit_aware_operators", False),
        )

        solver = FILOSolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )

        routes, best_profit, best_cost = solver.solve()

        return routes, best_profit, best_cost
