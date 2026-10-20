"""Move-operator context for the ACO hyper-heuristic.

:class:`HyperOperatorContext` wraps a set of routes with the node map, route loads,
neighbour lists and cached top insertions that the local-search move operators expect,
so the hyper-heuristic's wrapper operators can call them as standalone moves. Moved
unchanged from ``hyper_operators.py``, which re-exports it.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np


class HyperOperatorContext:
    """
    Context object that provides the interface expected by move operators.
    Wraps routes and provides helper methods for move evaluation.

    Attributes:
        routes: Current routes.
        d: Distance matrix.
        waste: Node wastes.
        Q: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        mandatory_nodes: Set of mandatory node indices.
        profit_aware_operators: Whether to use profit-aware heuristics.
        route_loads: Cached loads for each route.
        node_map: Mapping of node to its route and position.
        neighbors: Pre-computed neighbors for each node.
        rng: Random number generator.
        top_insertions: Cache for O(1) SWAP* insertion cost evaluation.
    """

    def __init__(
        self,
        routes: List[List[int]],
        dist_matrix: np.ndarray,
        waste: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        mandatory_nodes: Optional[List[int]] = None,
        rng: Optional[random.Random] = None,
        profit_aware_operators: bool = False,
        use_cache: bool = False,
    ):
        """
        Initialize HyperOperatorContext.

        Args:
            routes: Current routes.
            dist_matrix: Distance matrix.
            waste: Node wastes.
            capacity: Vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            mandatory_nodes: List of mandatory node indices.
            rng: Random number generator.
            profit_aware_operators: Whether to use profit-aware heuristics.
            use_cache: Whether to use the cache for operator applications.
        """
        self.routes = routes
        self.d = dist_matrix
        self.waste = waste
        self.Q = capacity
        self.R = R
        self.C = C
        self.mandatory_nodes = set(mandatory_nodes) if mandatory_nodes else set()
        self.profit_aware_operators = profit_aware_operators
        self.route_loads: List[float] = []
        self.node_map: Dict[int, Tuple[int, int]] = {}
        self.neighbors: Dict[int, List[int]] = {}
        self.rng = rng or random.Random()
        self.use_cache = use_cache

        # Top-3 Insertion Cache for O(1) SWAP* evaluation (Vidal 2022)
        # Maps: node_id -> route_idx -> [(cost_delta, position), ...]
        self.top_insertions: Dict[int, Dict[int, List[Tuple[float, int]]]] = {}

        self._build_structures()

    def _build_structures(self) -> None:
        """Build node map, route loads, and neighbor lists.

        """
        self.node_map.clear()
        self.route_loads = []

        for ri, route in enumerate(self.routes):
            load = sum(self.waste.get(n, 0) for n in route)
            self.route_loads.append(load)
            for pi, node in enumerate(route):
                self.node_map[node] = (ri, pi)

        # Build neighbor lists (k=10 nearest)
        n_nodes = len(self.d)
        for i in range(1, n_nodes):
            row = self.d[i]
            order = np.argsort(row)
            cands = []
            for c in order:
                if c not in (i, 0):
                    cands.append(c)
                    if len(cands) >= 10:
                        break
            self.neighbors[i] = cands

        # Compute Top-3 Insertion Cache for SWAP* operator ONLY if requested
        if self.use_cache:
            self._compute_top_insertions()

    def _calc_load_fresh(self, r: List[int]) -> float:
        """Calculate load of a route.

        Args:
            r: Route.

        Returns:
            float: Load of the route.
        """
        return sum(self.waste.get(x, 0) for x in r)

    def _get_load_cached(self, ri: int) -> float:
        """Get cached load of a route.

        Args:
            ri: Route index.

        Returns:
            float: Cached load of the route.
        """
        return self.route_loads[ri]

    def get_dist(self, i: int, j: int) -> float:
        """Abstraction to fetch distance from matrix or calculate on-demand.

        Args:
            i: First node index.
            j: Second node index.

        Returns:
            float: Distance between node i and node j.
        """
        return float(self.d[i, j])

    def _update_map(self, affected_indices: set):
        """Update node map and route loads.

        Args:
            affected_indices: Set of affected route indices.
        """
        for ri in sorted(affected_indices):
            if ri < len(self.routes):
                for pi, node in enumerate(self.routes[ri]):
                    self.node_map[node] = (ri, pi)
                self.route_loads[ri] = self._calc_load_fresh(self.routes[ri])
                # Update top insertions for affected route ONLY if cache is active
                if self.use_cache:
                    self._compute_top_insertions(route_idx=ri)

    def _compute_top_insertions(self, route_idx: Optional[int] = None):
        """
        Compute or update the Top-3 Insertion Cache for O(1) SWAP* evaluation.

        Following Vidal et al. (2022), this cache stores the 3 best insertion positions
        for each node into each route, enabling constant-time insertion cost evaluation.

        Args:
            route_idx: If provided, update cache only for this route. If None, initialize for all routes.
        """
        routes_to_update = [route_idx] if route_idx is not None else range(len(self.routes))

        for r_idx in routes_to_update:
            route = self.routes[r_idx]
            if not route:
                continue

            # For each node that could potentially be inserted into this route
            for node in self.neighbors:
                # Skip nodes already in this route
                if node in route:
                    continue

                # Initialize cache structure if needed
                if node not in self.top_insertions:
                    self.top_insertions[node] = {}

                # Calculate insertion costs for all positions
                insertion_costs = []
                for pos in range(len(route) + 1):
                    prev = route[pos - 1] if pos > 0 else 0
                    nxt = route[pos] if pos < len(route) else 0
                    delta = self.get_dist(prev, node) + self.get_dist(node, nxt) - self.get_dist(prev, nxt)
                    insertion_costs.append((delta, pos))

                # Keep top 3 (lowest cost) insertions
                insertion_costs.sort(key=lambda x: x[0])
                self.top_insertions[node][r_idx] = insertion_costs[:3]
