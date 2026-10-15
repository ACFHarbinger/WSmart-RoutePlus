"""Regret-K Insertion Route Improver.

Attributes:
    RegretKInsertionRouteImprover: Main class for regret-based insertion.

Example:
    >>> from logic.src.policies.route_improvement.regret_k_insertion import RegretKInsertionRouteImprover
    >>> improver = RegretKInsertionRouteImprover(config=cfg)
    >>> refined_tour, metrics = improver.process(tour, distance_matrix=dm, k=3)
"""

from typing import Any, List, Tuple

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.search_context import ImprovementMetrics
from logic.src.interfaces.route_improvement import IRouteImprovement
from logic.src.policies.helpers.operators.recreate_repair.regret import (
    regret_k_insertion,
    regret_k_profit_insertion,
)

from .base import RouteImproverRegistry
from .common.helpers import (
    assemble_tour,
    resolve_mandatory_nodes,
    split_tour,
    to_numpy,
)


@GlobalRegistry.register(
    PolicyTag.IMPROVEMENT,
    PolicyTag.HEURISTIC,
)
@RouteImproverRegistry.register("regret_k_insertion")
class RegretKInsertionRouteImprover(IRouteImprovement):
    """Regret-K insertion route improver.

    Calculates regret values to prioritize bin insertions that have
    few good alternative positions across the current routes.

    Attributes:
        config (Dict[str, Any]): Internal configuration state.

    Example:
        >>> improver = RegretKInsertionRouteImprover(config=cfg)
        >>> refined_tour, metrics = improver.process(tour, regret_k=3)
    """

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:
        """Apply Regret-K insertion to reconcile omitted bins.

        Args:
            tour: Initial tour sequence.
            kwargs: Context containing:
                distance_matrix: Distance lookup.
                regret_k: Regret depth (default 2).
                regret_noise: Random noise factor for insertion (default 0.0).
                cost_per_km: Distance cost coefficient.
                revenue_kg: Waste revenue coefficient.
                wastes: Bin waste mass mapping.
                capacity: Maximum vehicle capacity.
                n_bins: Total number of bins available.
                mandatory_nodes: Required nodes.

        Returns:
            Tuple[List[int], ImprovementMetrics]: Refined tour and metrics.
        """
        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        if distance_matrix is None or not tour:
            return tour, {"algorithm": "RegretKInsertionRouteImprover"}

        # Parameters
        k = kwargs.get("regret_k", self.config.get("regret_k", 2))
        noise = kwargs.get("regret_noise", self.config.get("regret_noise", 0.0))
        cost_per_km = kwargs.get("cost_per_km", self.config.get("cost_per_km", 0.0))
        revenue_kg = kwargs.get("revenue_kg", self.config.get("revenue_kg", 0.0))

        # Mandatory nodes resolution
        mandatory_nodes = resolve_mandatory_nodes(kwargs, self.config)

        # Problem parameters
        wastes = kwargs.get("wastes", self.config.get("wastes", {}))
        capacity = kwargs.get("capacity", self.config.get("capacity", float("inf")))

        # Convert distance matrix to numpy
        dm = to_numpy(distance_matrix)

        # Identify unselected bins
        n_bins = kwargs.get("n_bins", dm.shape[0] - 1)
        selected = set(tour) - {0}
        unselected = sorted(list(set(range(1, n_bins + 1)) - selected))

        try:
            routes = split_tour(tour) or [[]]

            if revenue_kg > 0 or cost_per_km > 0:
                refined_routes = regret_k_profit_insertion(
                    routes=routes,
                    removed_nodes=unselected,
                    dist_matrix=dm,
                    wastes=wastes,
                    capacity=capacity,
                    R=revenue_kg,
                    C=cost_per_km,
                    k=k,
                    mandatory_nodes=mandatory_nodes,
                    expand_pool=True,
                    noise=noise,
                )
            else:
                refined_routes = regret_k_insertion(
                    routes=routes,
                    removed_nodes=unselected,
                    dist_matrix=dm,
                    wastes=wastes,
                    capacity=capacity,
                    k=k,
                    mandatory_nodes=mandatory_nodes,
                    expand_pool=True,
                    noise=noise,
                )

            return assemble_tour(refined_routes), {"algorithm": "RegretKInsertionRouteImprover"}

        except Exception:
            return tour, {"algorithm": "RegretKInsertionRouteImprover"}
