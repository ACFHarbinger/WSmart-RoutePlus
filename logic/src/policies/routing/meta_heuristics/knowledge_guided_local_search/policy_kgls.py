"""
KGLS Policy Adapter.

Adapts the Knowledge-Guided Local Search (KGLS) to the global routing interface.

Attributes:
    KGLSPolicy: Policy adapter for the KGLS metaheuristic.

Example:
    >>> policy = KGLSPolicy()
    >>> routes, profit, cost = policy(obs)
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import KGLSConfig
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry

from .kgls import KGLSSolver
from .params import KGLSParams


@RoutingAlgorithmRegistry.register("kgls")
class KGLSPolicy(BaseRoutingPolicy):
    """
    KGLS policy class.

    Explores CVRP solutions by penalizing structurally sub-optimal connections.

    Attributes:
        config: Configuration parameters for the policy.
    """

    def __init__(self, config: Optional[Union[KGLSConfig, Dict[str, Any]]] = None):
        """Initialize KGLS policy with optional config.

        Args:
            config: KGLSConfig dataclass, raw dict from YAML, or None.

        Returns:
            None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the configuration class for KGLS.

        Args:
            None.

        Returns:
            Optional[Type]: The KGLSConfig class.
        """
        return KGLSConfig

    def _get_config_key(self) -> str:
        """Return config key for KGLS.

        Args:
            None.

        Returns:
            str: "kgls".
        """
        return "kgls"

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
        """Execute the Knowledge-Guided Local Search (KGLS) metaheuristic solver logic.

        KGLS is an advanced variant of GLS that incorporates domain-specific
        structural "knowledge" to guide the search. It penalizes features that
        are known to be sub-optimal based on geometric or historical problem
        properties (e.g., edge widths, relative lengths). The search uses a
        multi-move local search engine (relocate, swap, 2-opt, cross-exchange)
        and adjusts penalties across cycles to explore different feasible regions
        while prioritizing structural integrity.

        Args:
            sub_dist_matrix: Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes: Mapping of local node indices to their
                current bin inventory levels.
            capacity: Maximum vehicle collection capacity.
            revenue: Revenue obtained per kilogram of waste collected.
            cost_unit: Monetary cost incurred per kilometer traveled.
            values: Merged configuration dictionary containing
                KGLS parameters (num_perturbations, neighborhood_size, moves).
            mandatory_nodes: Local indices of bins that MUST be
                collected in this period.
            kwargs: Additional context, including:
                - data_nodes (Optional[Dict]): Raw nodes data for coordinate mapping.
                - search_context (Optional[SearchContext]): Context for tracking
                  recursive solver statistics.
                - multi_day_context (Optional[MultiDayContext]): Context for
                  inter-day state propagation.

        Returns:
            Tuple[List[List[int]], float, float]: A 3-tuple containing:
                - routes: Optimized collection routes (list-of-lists, local indices).
                - profit: Total calculated net profit (Total Revenue - Total Cost).
                - cost: Total travel cost calculated by the solver.
        """
        params = KGLSParams(
            time_limit=float(values.get("time_limit", 60.0)),
            num_perturbations=int(values.get("num_perturbations", 3)),
            neighborhood_size=int(values.get("neighborhood_size", 20)),
            moves=values.get("moves", ["relocate", "swap", "two_opt", "cross_exchange"]),
            penalization_cycle=values.get("penalization_cycle", ["width", "length", "width_length"]),
            local_search_iterations=int(values.get("local_search_iterations", 100)),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )

        # We need the locations for the Cost Evaluator to compute edge widths
        loc_data = kwargs.get("data_nodes")
        n_nodes = sub_dist_matrix.shape[0]
        locations = np.zeros((n_nodes, 2))
        if loc_data is not None:
            # Re-map 1...N onto 1...len(sub_dist_matrix) maintaining the continuous graph
            locs = loc_data.get("locs", [])
            for i, mapped_n in enumerate(mandatory_nodes + list(set(sub_wastes.keys()) - set(mandatory_nodes))):
                if mapped_n < len(locs):
                    # + 1 offset because 0 is depot
                    locations[i + 1] = locs[mapped_n]
            # Map depot
            locations[0] = loc_data.get("depot", [0.0, 0.0])

        solver = KGLSSolver(
            dist_matrix=sub_dist_matrix,
            locations=locations,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )

        routes, best_profit, best_cost = solver.solve()

        return routes, best_profit, best_cost
