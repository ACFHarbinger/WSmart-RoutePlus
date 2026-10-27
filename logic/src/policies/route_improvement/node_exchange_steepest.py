"""Steepest Node Exchange Route Improver.

Delegates to operators.improvement_descent.node_exchange_steepest (or its
profit variant when revenue/cost are configured) to perform pairwise
node swaps until a local minimum is reached.

Attributes:
    NodeExchangeSteepestRouteImprover: Pairwise node exchange improver class.

Example:
    >>> improver = NodeExchangeSteepestRouteImprover()
    >>> best_tour, metrics = improver.process(tour, distance_matrix=dm)
"""

from typing import Any, Dict, List, Tuple, cast

from logic.src.enums import PolicyTag
from logic.src.interfaces.context.search_context import ImprovementMetrics
from logic.src.interfaces.route_improvement import IRouteImprovement
from logic.src.policies.helpers.operators.improvement_descent import (
    node_exchange_steepest,
    node_exchange_steepest_profit,
)
from logic.src.utils.registry import ALL

from .base import RouteImproverRegistry
from .common.helpers import assemble_tour, split_tour, to_numpy


@ALL.register(
    PolicyTag.IMPROVEMENT,
    PolicyTag.HEURISTIC,
    PolicyTag.LOCAL_SEARCH,
)
@RouteImproverRegistry.register("node_exchange_steepest")
class NodeExchangeSteepestRouteImprover(IRouteImprovement):
    """Steepest-descent node exchange route improver.

    Performs inter-route and intra-route node swaps to improve the tour.

    Attributes:
        config (Dict[str, Any]): Configuration parameters.

    Example:
        >>> improver = NodeExchangeSteepestRouteImprover()
        >>> tour, metrics = improver.process(tour, distance_matrix=dm, max_iter=100)
    """

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:
        """Apply steepest node exchange improvement to the tour.

        Args:
            tour: The initial tour sequence.
            kwargs: Search context, including:
                - distance_matrix (np.ndarray): The distance matrix.
                - wastes (Dict[int, float]): Bin waste demands.
                - capacity (float): Vehicle capacity.
                - cost_per_km (float): Distance cost.
                - revenue_kg (float): Waste revenue.
                - max_iter (int): Maximum number of iterations.

        Returns:
            Tuple[List[int], ImprovementMetrics]: Refined tour and performance metrics.
        """
        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        if distance_matrix is None or not tour:
            return tour, {"algorithm": "NodeExchangeSteepestRouteImprover"}

        wastes = cast(Dict[int, float], kwargs.get("wastes", self.config.get("wastes", {})))
        capacity = cast(float, kwargs.get("capacity", self.config.get("capacity", float("inf"))))
        cost_per_km = cast(float, kwargs.get("cost_per_km", self.config.get("cost_per_km", 0.0)))
        revenue_kg = cast(float, kwargs.get("revenue_kg", self.config.get("revenue_kg", 0.0)))
        max_iter = cast(int, kwargs.get("max_iter", self.config.get("max_iter", 500)))

        dm = to_numpy(distance_matrix)

        try:
            routes = split_tour(tour)
            if not routes:
                return tour, {"algorithm": "NodeExchangeSteepestRouteImprover"}

            if revenue_kg > 0 or cost_per_km > 0:
                refined = node_exchange_steepest_profit(
                    routes=routes,
                    dist_matrix=dm,
                    wastes=wastes,
                    capacity=capacity,
                    R=revenue_kg,
                    C=cost_per_km,
                    max_iter=max_iter,
                )
            else:
                refined = node_exchange_steepest(
                    routes=routes,
                    dist_matrix=dm,
                    wastes=wastes,
                    capacity=capacity,
                    max_iter=max_iter,
                )

            return assemble_tour(refined), {"algorithm": "NodeExchangeSteepestRouteImprover"}

        except Exception:
            return tour, {"algorithm": "NodeExchangeSteepestRouteImprover"}
