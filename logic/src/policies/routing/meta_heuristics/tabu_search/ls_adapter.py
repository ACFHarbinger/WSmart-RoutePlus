"""Local-search adapter used by the tabu search's low-level heuristics.

Attributes:
    TSLSAdapter: Exposes a ``TSSolver`` state to the shared local-search operators.

Example:
    >>> TSLSAdapter(solver)
"""

from typing import List, Set


class TSLSAdapter:
    """Adapter for HGS-style operators to work with TSSolver.

    Attributes:
        routes: Routing solution.
        d: Distance matrix.
        waste: Waste mapping.
        Q: Capacity.
        C: Cost unit.
        params: Operator parameters.
        _load_cache: Cache for route loads.

    Example:
        >>> adapter = TSLSAdapter(routes, dist, waste, cap, cost)
    """

    def __init__(self, routes, dist_matrix, wastes, capacity, cost_unit):
        """Initializes the Tabu Search Local Search adapter.

        Args:
            routes: Initial routes.
            dist_matrix: Symmetric distance matrix.
            wastes: Mapping of bin IDs to waste quantities.
            capacity: Maximum vehicle collection capacity.
            cost_unit: Cost per km traveled.

        """
        self.routes = routes
        self.d = dist_matrix
        self.waste = wastes
        self.Q = capacity
        self.C = cost_unit
        self.params = type("Params", (), {"use_relocate_chain": False})()
        self._load_cache = {}

    def _get_load_cached(self, r_idx: int) -> float:
        """Get the load of a route using a cache.

        Args:
            r_idx: Route index.

        Returns:
            Load of the route.
        """
        if r_idx not in self._load_cache:
            self._load_cache[r_idx] = sum(self.waste.get(n, 0) for n in self.routes[r_idx])
        return self._load_cache[r_idx]

    def _update_map(self, affected_indices: Set[int]):
        """Update the load cache.

        Args:
            affected_indices: Indices of routes to update.

        """
        for r_idx in affected_indices:
            if r_idx in self._load_cache:
                del self._load_cache[r_idx]

    def _calc_load_fresh(self, r: List[int]) -> float:
        """Calculate the load of a route.

        Args:
            r: Route.

        Returns:
            float: Load of the route.
        """
        return sum(self.waste.get(n, 0) for n in r)
