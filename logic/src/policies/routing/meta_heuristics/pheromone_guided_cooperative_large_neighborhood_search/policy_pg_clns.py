"""
PG-CLNS Policy Adapter.

Adapts the Pheromone-Guided Cooperative Large Neighborhood Search (PG-CLNS) logic to the agnostic interface.

Attributes:
    PGCLNSPolicy: Policy adapter for the PG-CLNS metaheuristic.

Example:
    >>> policy = PGCLNSPolicy()
    >>> routes, profit, cost = policy(obs)
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.pg_clns import PGCLNSConfig
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry

from .pg_clns import PGCLNSSolver


@RoutingAlgorithmRegistry.register("pg_clns")
class PGCLNSPolicy(BaseRoutingPolicy):
    """
    PG-CLNS policy class.

    Visits pre-selected 'mandatory' bins using the population-based PG-CLNS metaheuristic.

    Attributes:
        config: Configuration parameters for the policy.
    """

    def __init__(self, config: Optional[Union[PGCLNSConfig, Dict[str, Any]]] = None):
        """Initialize PG-CLNS policy with optional config.

        Args:
            config: PGCLNSConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the configuration class for PG-CLNS.

        Returns:
            Optional[Type]: The PGCLNSConfig class.
        """
        return PGCLNSConfig

    def _get_config_key(self) -> str:
        """Return config key for PG-CLNS.

        Returns:
            str: "pg_clns".
        """
        return "pg_clns"

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
        """Execute the Pheromone-Guided Cooperative Large Neighborhood Search (PG-CLNS) solver logic.

        PG-CLNS is a hybrid metaheuristic that combines the constructive power of
        Ant Colony Optimization (ACO) with the local improvement capabilities of
        Large Neighborhood Search (LNS) within a cooperative population-based framework.

        Args:
            sub_dist_matrix: Symmetric distance matrix for the current sub-problem nodes.
            sub_wastes: Mapping of local node indices to their current bin inventory levels.
            capacity: Maximum vehicle collection capacity.
            revenue: Revenue obtained per kilogram of waste collected.
            cost_unit: Monetary cost incurred per kilometer traveled.
            values: Merged configuration dictionary containing PG-CLNS parameters.
            mandatory_nodes: Local indices of bins that MUST be collected in this period.
            kwargs: Additional context.

        Returns:
            Tuple[List[List[int]], float, float]: Optimized routes, total profit, and total cost.
        """
        seed = values.get("seed", 42)
        cfg = self._effective_config(values)

        solver = PGCLNSSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            cfg,
            mandatory_nodes,
            seed=seed,
        )

        routes, profit, solver_cost = solver.solve()
        return routes, profit, solver_cost
