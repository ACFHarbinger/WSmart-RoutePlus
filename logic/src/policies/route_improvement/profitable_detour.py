"""Profitable Detour Route Improver.

This module implements a detour-based improvement heuristic that identifies
unselected bins that can be profitably inserted into existing routes without
exceeding a specified detour budget or vehicle capacity.

Attributes:
    ProfitableDetourRouteImprover: Route improvement class using profitable detours.

Example:
    >>> improver = ProfitableDetourRouteImprover()
    >>> best_tour, metrics = improver.process(tour, distance_matrix=dm, detour_epsilon=0.1)
"""

from typing import Any, Dict, List, Tuple, cast

from logic.src.enums import PolicyTag
from logic.src.interfaces.context.search_context import ImprovementMetrics
from logic.src.interfaces.route_improvement import IRouteImprovement
from logic.src.utils.registry import ALL

from .base import RouteImproverRegistry
from .common.helpers import assemble_tour, route_load, split_tour, to_numpy


@ALL.register(
    PolicyTag.IMPROVEMENT,
    PolicyTag.HEURISTIC,
    PolicyTag.PROFIT_AWARE,
)
@RouteImproverRegistry.register("profitable_detour")
class ProfitableDetourRouteImprover(IRouteImprovement):
    """Profitable detour route improver.

    Inserts bins that lie within a certain detour budget from existing tour edges.

    Attributes:
        config (Dict[str, Any]): Configuration parameters.

    Example:
        >>> improver = ProfitableDetourRouteImprover()
        >>> tour, metrics = improver.process(tour, distance_matrix=dm)
    """

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:  # noqa: C901
        """Apply Profitable Detour augmentation to the tour.

        Args:
            tour: Initial tour sequence (list of bin IDs including depot 0s).
            kwargs: Search context, including:
                - distance_matrix (np.ndarray): The distance matrix.
                - detour_epsilon (float): Maximum proportion of detour distance relative to original edge.
                - cost_per_km (float): Distance cost.
                - revenue_kg (float): Waste revenue.
                - wastes (Dict[int, float]): Bin waste demands.
                - capacity (float): Vehicle capacity.
                - n_bins (int): Total number of bins available.

        Returns:
            Tuple[List[int], ImprovementMetrics]: Refined and potentially expanded tour.

        Complexity:
            O(Routes * Edges * |unselected|^2) in the worst case where many bins
            can be inserted between a single edge pair.
        """
        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        if distance_matrix is None or not tour:
            return tour, {"algorithm": "ProfitableDetourRouteImprover"}

        # Parameters
        epsilon = cast(float, kwargs.get("detour_epsilon", self.config.get("detour_epsilon", 0.2)))
        cost_per_km = cast(float, kwargs.get("cost_per_km", self.config.get("cost_per_km", 0.0)))
        revenue_kg = cast(float, kwargs.get("revenue_kg", self.config.get("revenue_kg", 0.0)))

        # Problem parameters
        wastes = cast(Dict[int, float], kwargs.get("wastes", self.config.get("wastes", {})))
        capacity = cast(float, kwargs.get("capacity", self.config.get("capacity", float("inf"))))

        # Convert distance matrix to numpy
        dm = to_numpy(distance_matrix)

        # Identify unselected bins
        n_bins = cast(int, kwargs.get("n_bins", dm.shape[0] - 1))
        selected = set(tour) - {0}
        unselected = set(range(1, n_bins + 1)) - selected

        try:
            routes = split_tour(tour)
            if not routes:
                return tour, {"algorithm": "ProfitableDetourRouteImprover"}

            current_routes = [r[:] for r in routes]
            current_loads = [route_load(r, wastes) for r in current_routes]

            for r_idx in range(len(current_routes)):
                route = current_routes[r_idx]
                if not route:
                    continue

                # Nodes including depot: [0, n0, n1, ..., nk, 0]
                full_route = [0] + route + [0]

                # We iterate over edges in the route. Note that inserting a node
                # changes subsequent edges. We process originally existing edges.
                # However, the brief says "For each edge (u, v) in the current tour".
                # I'll iterate through the route nodes.

                i = 0
                while i < len(full_route) - 1:
                    u, v = full_route[i], full_route[i + 1]
                    base_dist = float(dm[u, v])

                    candidates = []
                    for b in unselected:
                        detour = float(dm[u, b]) + float(dm[b, v]) - base_dist
                        if detour <= epsilon * base_dist:
                            gain = revenue_kg * wastes.get(b, 0.0) - cost_per_km * detour
                            if gain > 0:
                                candidates.append((b, detour, gain))

                    # Sort candidates by gain descending
                    candidates.sort(key=lambda x: x[2], reverse=True)

                    inserted_any = False
                    # Insert profitable ones
                    for b, _detour, _gain in candidates:
                        if current_loads[r_idx] + wastes.get(b, 0.0) <= capacity:
                            # Insert after u (which is at index i in full_route)
                            current_routes[r_idx].insert(i, b)
                            current_loads[r_idx] += wastes.get(b, 0.0)
                            unselected.remove(b)
                            # Update full_route locally
                            full_route.insert(i + 1, b)
                            inserted_any = True
                            break  # Re-evaluate the new edges (u, b) and (b, v)

                    if not inserted_any:
                        i += 1

            return assemble_tour(current_routes), {"algorithm": "ProfitableDetourRouteImprover"}

        except Exception:
            return tour, {"algorithm": "ProfitableDetourRouteImprover"}
