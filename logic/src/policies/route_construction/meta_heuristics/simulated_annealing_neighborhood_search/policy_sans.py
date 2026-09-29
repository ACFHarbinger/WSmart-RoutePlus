"""
SANS Policy Adapter (Simulated Annealing Neighborhood Search).

Uses Simulated Annealing for route optimization.
Supports two engines:
  - 'new': Improved SA with initial solution and iterative refinement
  - 'og': Original look-ahead algorithm for collection (LAC)

Attributes:
    SANSPolicy: Policy class for the SANS approach.

Example:
    >>> from logic.src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.policy_sans import SANSPolicy
    >>> sans_policy = SANSPolicy()
    >>> sans_policy.execute()
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import SANSConfig
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.dispatcher import (
    execute_new,
    execute_og,
)

from .params import SANSParams


@RouteConstructorRegistry.register("sans")
@RouteConstructorRegistry.register("lac")  # Backward compatibility alias
class SANSPolicy(BaseRoutingPolicy):
    """
    Simulated Annealing Neighborhood Search policy class.

    Uses SA optimization with custom initialization and mandatory enforcement.
    Supports two engines via the 'engine' parameter:
      - 'new': Improved simulated annealing with initial solution computation
      - 'og': Original look-ahead collection (LAC) algorithm

    Attributes:
        None
    """

    def __init__(self, config: Optional[Union[SANSConfig, Dict[str, Any]]] = None):
        """Initialize SANS policy with optional config.

        Args:
            config: SANSConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Returns the configuration class for the SANS policy.

        Returns:
            Optional[Type]: The configuration class for the SANS policy.
        """
        return SANSConfig

    def _get_config_key(self) -> str:
        """Returns the configuration key for the SANS policy.

        Returns:
            str: The registry key 'sans'.
        """
        return "sans"

    def _create_subset_problem(
        self,
        mandatory: List[int],
        distance_matrix: Any,
        bins: Any,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Dict[int, float], List[int], List[int]]:
        """Create subset problem for SANS.

        SANS historically operates on the full problem, not a restricted subset.
        Override to always use use_all_bins=True to preserve this behavior.

        Args:
            mandatory: List of mandatory bin indices.
            distance_matrix: Full distance matrix.
            bins: Bins object.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (sub_dist_matrix, sub_wastes, subset_indices, local_mandatory).
        """
        # Force use_all_bins=True to preserve SANS's historical full-bin behavior
        # Remove any existing use_all_bins from kwargs to avoid duplicate argument
        kwargs.pop("use_all_bins", None)
        return super()._create_subset_problem(
            mandatory=mandatory,
            distance_matrix=distance_matrix,
            bins=bins,
            use_all_bins=True,
            **kwargs,
        )

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
        """Run the SANS solver.

        Implements the SANS-specific solving logic, dispatching to either
        execute_new or execute_og based on the engine parameter.

        Since _create_subset_problem() forces use_all_bins=True, the subset
        problem is always the full problem, and local indices equal global indices.

        Args:
            sub_dist_matrix: Distance matrix (full problem due to use_all_bins=True).
            sub_wastes: Dictionary of wastes (full problem).
            capacity: Vehicle capacity.
            revenue: Revenue per unit.
            cost_unit: Cost per unit of distance.
            values: Dictionary of merged config values.
            mandatory_nodes: List of mandatory node indices (global == local).
            **kwargs: Additional arguments including bins, coords, distance_matrix, etc.

        Returns:
            Tuple[List[List[int]], float, float]:
                Tuple of (routes, profit, cost).
        """
        # Determine engine and parameters
        params = SANSParams.from_config(self._config or kwargs.get("config", {}).get("sans", {}))

        # Extract original data from kwargs
        bins = kwargs.get("bins")
        distance_matrix = kwargs.get("distance_matrix")

        if bins is None or distance_matrix is None:
            # Fallback: return empty solution if required data is missing
            return [[]], 0.0, 0.0

        # Since use_all_bins=True, mandatory_nodes are already global IDs
        # and local indices equal global indices

        # Prepare kwargs for dispatcher functions
        # Preserve caller data and tracking/context fields used by dispatchers.
        dispatcher_kwargs = dict(kwargs)
        dispatcher_kwargs["mandatory"] = mandatory_nodes
        dispatcher_kwargs.setdefault("seed", params.seed)

        # Call the appropriate engine
        if params.engine == "og":
            tour, cost, profit, _, _ = execute_og(self, params=params, **dispatcher_kwargs)
        else:
            tour, cost, profit, _, _ = execute_new(self, params=params, **dispatcher_kwargs)

        # Convert tour (flat list) to routes (list of lists)
        # Tour format: [0, node1, node2, ..., 0, node3, node4, ..., 0]
        # No index mapping needed since local == global
        routes = []
        current_route = []
        for node in tour:
            if node == 0:
                if current_route:
                    routes.append(current_route)
                    current_route = []
            else:
                current_route.append(node)
        if current_route:
            routes.append(current_route)

        # If no routes were created, return empty
        if not routes:
            routes = [[]]

        return routes, profit, cost
