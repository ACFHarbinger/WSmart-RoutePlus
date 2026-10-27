"""
FILO Local Search module.

Implements the localized HRVND engine with SVC-bounded move evaluation
and O(1) scaling via Selective Vertex Caching (SVC).

All operator gain computations are self-contained. Moves are generated
strictly from the SVC kernel using precomputed neighbor lists.

**O(1) State Tracking:**
- ``node_to_route_pos[u]`` → ``(route_idx, pos_idx)``  (inverse mapping)
- ``route_loads[ri]``      → current load of route ri   (incremental)

Both are maintained incrementally on every move application,
eliminating all O(N) scans from the inner evaluation loops.

The search uses Randomized Variable Neighborhood Descent (RVND):
Tier-1 operators are shuffled, each evaluated to local optimality.
If any operator improves, the tier restarts. If Tier-1 stagnates,
Tier-2 (recursive ejection chains, depth up to n_EC=3) is invoked.

Gamma sparsification probabilistically filters neighbor evaluations.

Attributes:
    FILOLocalSearch: Implementation of the FILO Local Search algorithm.

Example:
    >>> from logic.src.policies.helpers.local_search import FILOLocalSearch

References:
    Accorsi & Vigo (2021) Section 4.4, 4.5, and 4.6.

Note:
    Tier-1 operators, move gains and K_NEIGHBORS are located in
    :mod:`.local_search_filo_operators`.
"""

import random
from typing import Any, Dict, List, Optional, Set, Tuple, cast

import numpy as np

from logic.src.policies.helpers.local_search.local_search_base import LocalSearch
from logic.src.policies.helpers.local_search.local_search_filo_operators import (
    K_NEIGHBORS as K_NEIGHBORS,
)
from logic.src.policies.helpers.local_search.local_search_filo_operators import (
    FILOOperatorsMixin as FILOOperatorsMixin,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EJCH_MAX_DEPTH: int = 3  # Maximum ejection chain recursion depth

__all__ = ["FILOLocalSearch", "FILOOperatorsMixin", "K_NEIGHBORS", "EJCH_MAX_DEPTH"]


class FILOLocalSearch(LocalSearch, FILOOperatorsMixin):
    """Localized HRVND with O(1) state tracking, RVND, and gamma sparsification.

    Attributes:
        node_to_route_pos (Dict[int, Tuple[int, int]]): mapping each node
            to its ``(route_idx, position_idx)``.  Replaces O(N) ``_find_node``.
        _route_loads (List[float]): storing the current load of each route.
            Replaces O(N) ``sum(waste...)`` calls.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize FILO Local Search.

        Args:
            args: Variable length argument list.
            kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self._node_gamma: Dict[int, float] = {}
        self.node_to_route_pos: Dict[int, Tuple[int, int]] = {}
        self._route_loads: List[float] = []

    # ------------------------------------------------------------------
    # O(1) State Management
    # ------------------------------------------------------------------

    def _rebuild_index(self) -> None:
        """Rebuild ``node_to_route_pos`` and ``_route_loads`` from scratch.

        Called once at the start of ``optimize()``.  Cost: O(total nodes).
        """
        self.node_to_route_pos.clear()
        self._route_loads = []
        for ri, route in enumerate(self.routes):
            load = 0.0
            for pi, node in enumerate(route):
                self.node_to_route_pos[node] = (ri, pi)
                load += self.waste.get(node, 0.0)
            self._route_loads.append(load)

    def _rebuild_route_positions(self, ri: int) -> None:
        """Rebuild position indices for a single route.  Cost: O(|route|).

        Args:
            ri: Index of the route to rebuild.
        """
        for pi, node in enumerate(self.routes[ri]):
            self.node_to_route_pos[node] = (ri, pi)

    def _find_node(self, u: int) -> Tuple[int, int]:
        """O(1) lookup for node position via inverse mapping.

        Args:
            u: Node ID to find.

        Returns:
            Tuple[int, int]: (route_idx, position_idx) of the node, or (-1, -1) if not found.
        """
        return self.node_to_route_pos.get(u, (-1, -1))

    # ------------------------------------------------------------------
    # Tier-2: Recursive Ejection Chain (depth up to n_EC)
    # ------------------------------------------------------------------

    def _try_ejection_chain(self, active_nodes: Set[int]) -> bool:  # type: ignore[override]  # FILO takes active nodes, not a route index
        """Attempt a recursive ejection chain from an SVC vertex.

        Args:
            active_nodes: Set of active node IDs to search.

        Returns:
            bool: True if a move was made, False otherwise.
        """
        for u in active_nodes:
            ri, pi = self._find_node(u)
            if ri == -1:
                continue
            if self._recursive_eject(u, ri, pi, depth=EJCH_MAX_DEPTH, visited=set()):
                return True
        return False

    def _recursive_eject(
        self,
        u: int,
        ri: int,
        pi: int,
        depth: int,
        visited: Set[int],
    ) -> bool:
        """Recursively eject node *u* from route *ri*, displacing others.

        Args:
            u: Node ID to eject.
            ri: Route index of the node.
            pi: Position index of the node.
            depth: Remaining depth of the recursion.
            visited: Set of visited nodes to prevent cycles.

        Returns:
            bool: True if a move was made, False otherwise.
        """
        if depth <= 0:
            return False
        visited = visited | {u}
        w_u = self.waste.get(u, 0.0)

        # Compute removal savings for u
        route_i = self.routes[ri]
        prev_i = route_i[pi - 1] if pi > 0 else 0
        nxt_i = route_i[pi + 1] if pi < len(route_i) - 1 else 0
        d = cast(np.ndarray, self.d)  # matrix is present when FILO indexes distances
        removal_saving = d[prev_i, u] + d[u, nxt_i] - d[prev_i, nxt_i]

        # Phase 1: Try direct insertion (improving)
        best_cost, best_ins = float("inf"), None
        for rj in range(len(self.routes)):
            if rj == ri:
                continue
            if self._route_loads[rj] + w_u > self.Q:
                continue
            route_j = self.routes[rj]
            for pj in range(len(route_j) + 1):
                prev = route_j[pj - 1] if pj > 0 else 0
                nxt = route_j[pj] if pj < len(route_j) else 0
                cost = d[prev, u] + d[u, nxt] - d[prev, nxt]
                if cost < best_cost:
                    best_cost, best_ins = cost, (rj, pj)

        if best_ins and (best_cost - removal_saving) * self.C < -1e-6:
            rj, pj = best_ins
            self.routes[ri].pop(pi)
            self._route_loads[ri] -= w_u
            self.routes[rj].insert(pj, u)
            self._route_loads[rj] += w_u
            self._rebuild_route_positions(ri)
            self._rebuild_route_positions(rj)
            return True

        # Phase 2: Eject victim v, place u at v's position, recursively place v
        for v in self.neighbors[u][:K_NEIGHBORS]:
            if v in visited:
                continue
            v_ri, v_pi = self._find_node(v)
            if v_ri == -1 or v_ri == ri:
                continue
            w_v = self.waste.get(v, 0.0)
            if self._route_loads[v_ri] - w_v + w_u > self.Q:
                continue

            prev_v = self.routes[v_ri][v_pi - 1] if v_pi > 0 else 0
            nxt_v = self.routes[v_ri][v_pi + 1] if v_pi < len(self.routes[v_ri]) - 1 else 0
            old_v_cost = d[prev_v, v] + d[v, nxt_v]
            new_u_cost = d[prev_v, u] + d[u, nxt_v]
            local_delta = (new_u_cost - old_v_cost - removal_saving) * self.C

            if local_delta < 1e-6:
                # Speculatively apply
                self.routes[ri].pop(pi)
                self._route_loads[ri] -= w_u
                self.routes[v_ri][v_pi] = u
                self._route_loads[v_ri] += w_u - w_v
                self.node_to_route_pos[u] = (v_ri, v_pi)
                self._rebuild_route_positions(ri)

                if self._place_displaced(v, depth - 1, visited):
                    return True

                # Rollback
                self.routes[v_ri][v_pi] = v
                self._route_loads[v_ri] += w_v - w_u
                self.routes[ri].insert(pi, u)
                self._route_loads[ri] += w_u
                self.node_to_route_pos[v] = (v_ri, v_pi)
                self._rebuild_route_positions(ri)

        return False

    def _place_displaced(self, v: int, depth: int, visited: Set[int]) -> bool:
        """Try to place displaced node *v* via direct insertion or cascade.

        Args:
            v: Node ID to place.
            depth: Remaining depth of the recursion.
            visited: Set of visited nodes to prevent cycles.

        Returns:
            bool: True if a move was made, False otherwise.
        """
        w_v = self.waste.get(v, 0.0)
        d = cast(np.ndarray, self.d)  # matrix is present when FILO indexes distances

        # Phase A: Direct feasible insertion
        best_cost, best_ins = float("inf"), None
        for rj in range(len(self.routes)):
            if self._route_loads[rj] + w_v > self.Q:
                continue
            route_j = self.routes[rj]
            for pj in range(len(route_j) + 1):
                prev = route_j[pj - 1] if pj > 0 else 0
                nxt = route_j[pj] if pj < len(route_j) else 0
                cost = d[prev, v] + d[v, nxt] - d[prev, nxt]
                if cost < best_cost:
                    best_cost, best_ins = cost, (rj, pj)

        if best_ins:
            rj, pj = best_ins
            self.routes[rj].insert(pj, v)
            self._route_loads[rj] += w_v
            self._rebuild_route_positions(rj)
            return True

        # Phase B: Cascade
        if depth <= 0:
            return False

        for w in self.neighbors.get(v, [])[:K_NEIGHBORS]:
            if w in visited:
                continue
            w_ri, w_pi = self._find_node(w)
            if w_ri == -1:
                continue
            w_w = self.waste.get(w, 0.0)
            if self._route_loads[w_ri] - w_w + w_v > self.Q:
                continue

            # Speculatively eject w, insert v at w's position
            self.routes[w_ri][w_pi] = v
            self._route_loads[w_ri] += w_v - w_w
            self.node_to_route_pos[v] = (w_ri, w_pi)

            if self._place_displaced(w, depth - 1, visited | {v}):
                return True

            # Rollback
            self.routes[w_ri][w_pi] = w
            self._route_loads[w_ri] += w_w - w_v
            self.node_to_route_pos[w] = (w_ri, w_pi)

        return False

    # ------------------------------------------------------------------
    # Main optimization loop: RVND
    # ------------------------------------------------------------------

    def optimize(
        self,
        solution: List[List[int]],
        active_nodes: Optional[Set[int]] = None,
        node_gamma: Optional[List[float]] = None,
    ) -> List[List[int]]:
        """Execute RVND bounded by SVC with gamma sparsification.

        Args:
            solution: Current set of routes.
            active_nodes: Vertices in the Selective Vertex Cache.
            node_gamma: Per-node gamma values for probabilistic neighbor filtering.

        Returns:
            Improved set of routes.
        """
        self.routes = [r[:] for r in solution]
        if not active_nodes:
            return self.routes

        # Build O(1) state tracking structures
        self._rebuild_index()

        # Store gamma values
        self._node_gamma = {}
        if node_gamma:
            for i, g in enumerate(node_gamma):
                self._node_gamma[i] = g

        max_iters = getattr(self.params, "local_search_iterations", 500)
        iterations = 0

        tier1_ops = [
            ("relocate", self._try_relocate),
            ("swap", self._try_swap),
            ("2opt", self._try_2opt),
            ("tails", self._try_tails),
            ("split", self._try_split),
        ]

        improved_global = True
        while improved_global and iterations < max_iters:
            improved_global = False
            random.shuffle(tier1_ops)

            for _op_name, op_fn in tier1_ops:
                while iterations < max_iters:
                    if op_fn(active_nodes):
                        improved_global = True
                        iterations += 1
                    else:
                        break

                if improved_global:
                    break

            # Tier-2: Recursive Ejection Chain (only if Tier-1 stagnates)
            if not improved_global and iterations < max_iters and self._try_ejection_chain(active_nodes):
                improved_global = True
                iterations += 1

        return [r for r in self.routes if r]
