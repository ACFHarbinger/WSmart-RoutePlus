"""Or-opt Route Improver.

This module implements the Or-opt local search operator, which relocates chains
of consecutive nodes to different positions within or between routes.

Attributes:
    OrOptRouteImprover: Route improvement class using Or-opt.

Example:
    >>> improver = OrOptRouteImprover()
    >>> best_tour, metrics = improver.process(tour, distance_matrix=dm)
"""

from typing import Any, List, Tuple

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.search_context import ImprovementMetrics
from logic.src.interfaces.route_improvement import IRouteImprovement

from .base import RouteImproverRegistry
from .common.helpers import assemble_tour, split_tour, to_numpy


@GlobalRegistry.register(
    PolicyTag.IMPROVEMENT,
    PolicyTag.HEURISTIC,
    PolicyTag.LOCAL_SEARCH,
)
@RouteImproverRegistry.register("or_opt")
class OrOptRouteImprover(IRouteImprovement):
    """Or-opt route improver that relocates chains of nodes.

    Wraps SweepLocalSearch.or_opt.

    Attributes:
        config (Dict[str, Any]): Configuration parameters.

    Example:
        >>> improver = OrOptRouteImprover()
        >>> tour, metrics = improver.process(tour, distance_matrix=dm, chain_len=2)
    """

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:
        """Apply Or-opt improvement to the tour.

        Args:
            tour: Initial tour sequence.
            kwargs: Search context, including:
                - distance_matrix (np.ndarray): The distance matrix.
                - chain_len (int): Maximum length of the node chain to relocate.
                - iterations (int): Maximum number of iterations.
                - wastes (Dict[int, float]): Bin waste demands.
                - capacity (float): Vehicle capacity.
                - R (float): Revenue per kg.
                - C (float): Cost per km.
                - seed (int): Random seed.

        Returns:
            Tuple[List[int], ImprovementMetrics]: Refined tour and performance metrics.
        """
        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        if distance_matrix is None or not tour:
            return tour, {"algorithm": "OrOptRouteImprover"}

        # Parameters
        chain_len = kwargs.get("chain_len", self.config.get("chain_len", 2))
        iterations = kwargs.get("iterations", kwargs.get("max_iterations", self.config.get("iterations", 500)))
        seed = kwargs.get("seed", self.config.get("seed", 42))

        # Problem parameters
        wastes = kwargs.get("wastes", self.config.get("wastes", {}))
        capacity = kwargs.get("capacity", self.config.get("capacity", float("inf")))
        R = kwargs.get("R", self.config.get("R", 1.0))
        C = kwargs.get("C", self.config.get("C", 1.0))

        # Convert distance matrix to numpy
        dm = to_numpy(distance_matrix)
        if len(tour) < 3:
            return tour, {"algorithm": "OrOptRouteImprover"}

        try:
            from logic.src.policies.helpers.local_search.local_search_sweeps import SweepLocalSearch

            routes = split_tour(tour)
            if not routes:
                return tour, {"algorithm": "OrOptRouteImprover"}

            ls = SweepLocalSearch(
                dist_matrix=dm,
                wastes=wastes,
                capacity=capacity,
                R=R,
                C=C,
                improvement_threshold=1e-6,
                seed=seed,
            )
            ls.set_routes(routes)

            for _ in range(iterations):
                if not ls.or_opt(chain_len=chain_len):
                    break

            return assemble_tour(ls.get_routes()), {"algorithm": "OrOptRouteImprover"}

        except Exception:
            # Fallback to original tour on any unexpected error
            return tour, {"algorithm": "OrOptRouteImprover"}
