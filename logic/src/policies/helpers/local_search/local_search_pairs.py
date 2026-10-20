"""Neighbour-pair evaluation and route-structure bookkeeping of the local search.

:class:`LocalSearchPairMixin` holds the
:class:`~logic.src.policies.helpers.local_search.local_search_base.LocalSearch` methods that
prune and dispatch a node pair (polar sectors and their overlap, the inter-route and
intra-route operator cascades) and that keep the node map and cached route loads current.
Moved unchanged from ``local_search_base.py``.
"""

from __future__ import annotations

import math
from typing import Optional, Set, Tuple


class LocalSearchPairMixin:
    """Pair evaluation and route bookkeeping methods of :class:`LocalSearch` (moved unchanged from ``local_search_base.py``)."""

    def _polar_sector(self, route_idx: int) -> Optional[Tuple[float, float]]:
        """
        Return the (min_angle, max_angle) polar sector of a route in radians,
        measured from depot (node 0). Returns None for empty routes.
        Uses caching to avoid redundant calculations.

        Args:
            route_idx: Index of the route in `self.routes`.

        Returns:
            Optional[Tuple[float, float]]: (min_angle, max_angle) in radians,
                or None if route is empty.
        """
        if route_idx in self.route_sectors:
            return self.route_sectors[route_idx]

        route = self.routes[route_idx]
        if not route:
            self.route_sectors[route_idx] = None
            return None

        x_coords = self.x_coords
        y_coords = self.y_coords
        if x_coords is None or y_coords is None:
            res = (-math.pi, math.pi)
            self.route_sectors[route_idx] = res
            return res

        angles = []
        for node in route:
            angle = math.atan2(y_coords[node] - y_coords[0], x_coords[node] - x_coords[0])
            angles.append(angle)

        res = (min(angles), max(angles))
        self.route_sectors[route_idx] = res
        return res

    def _sectors_overlap(self, s1: Optional[Tuple[float, float]], s2: Optional[Tuple[float, float]]) -> bool:
        """Return True if two polar sectors overlap, handling the ±π wraparound.

        Args:
            s1: First polar sector.
            s2: Second polar sector.

        Returns:
            bool: True if the sectors overlap (including circular wraparound).
        """
        if s1 is None or s2 is None:
            return True
        # Standard interval overlap
        if s1[0] <= s2[1] and s2[0] <= s1[1]:
            return True
        # Wraparound: shift one sector by 2π and re-test
        shifted = (s1[0] + 2 * math.pi, s1[1] + 2 * math.pi)
        if shifted[0] <= s2[1] and s2[0] <= shifted[1]:
            return True
        shifted = (s2[0] + 2 * math.pi, s2[1] + 2 * math.pi)
        return s1[0] <= shifted[1] and shifted[0] <= s1[1]

    def _process_pair(self, u: int, v: int) -> bool:
        """
        Attempt all applicable move operators on the (u, v) pair.
        Returns True if any improving move was applied.

        Args:
            u: First node ID.
            v: Second node ID.

        Returns:
            bool: True if any move was successfully applied.
        """
        u_loc = self.node_map.get(u)
        v_loc = self.node_map.get(v)

        if not u_loc and not v_loc:
            return False

        # --- NEW VRPP LOGIC: Try Drop Operator ---
        if self._should_try_operator("drop"):
            # Try dropping u if it's currently routed
            if u_loc and self._move_drop(u, u_loc[0], u_loc[1]):
                return True

            # Re-fetch v_loc in case dropping u shifted indices
            v_loc = self.node_map.get(v)
            # Try dropping v if it's currently routed
            if v_loc and self._move_drop(v, v_loc[0], v_loc[1]):
                return True
        # -----------------------------------------

        # Rest of your original logic remains exactly the same
        if not u_loc and v_loc:
            r_v, p_v = v_loc
            return self._should_try_operator("unrouted_insert") and self._move_unrouted_insert(u, r_v, p_v)

        if u_loc and not v_loc:
            r_u, p_u = u_loc
            return self._should_try_operator("unrouted_insert") and self._move_unrouted_insert(v, r_u, p_u)

        r_u, p_u = u_loc  # type: ignore[misc]
        r_v, p_v = v_loc  # type: ignore[misc]

        if r_u != r_v:
            return self._process_inter_route(u, v, r_u, p_u, r_v, p_v)
        else:
            return self._process_intra_route(u, v, r_u, p_u, r_v, p_v)

    def _process_inter_route(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """Process moves between different routes.

        Args:
            u: Node from the first route.
            v: Node from the second route.
            r_u: Index of the first route.
            p_u: Position of node `u` in `r_u`.
            r_v: Index of the second route.
            p_v: Position of node `v` in `r_v`.

        Returns:
            bool: True if an improving inter-route move was applied.
        """
        if (
            self._should_try_operator("relocate") or self._should_try_operator("inter_relocate")
        ) and self._move_relocate(u, v, r_u, p_u, r_v, p_v):
            return True

        if (self._should_try_operator("swap") or self._should_try_operator("inter_swap")) and self._move_swap(
            u, v, r_u, p_u, r_v, p_v
        ):
            return True

        if self._should_try_operator("inter_2opt_star") and self._move_2opt_star(u, v, r_u, p_u, r_v, p_v):
            return True
        if self._should_try_operator("cross") and self._move_cross(u, v, r_u, p_u, r_v, p_v):
            return True
        if self._should_try_operator("shift_2_0") and self._move_shift_2_0(r_u, p_u, r_v, p_v):
            return True
        if self._should_try_operator("swap_2_1") and self._move_swap_2_1(r_u, p_u, r_v, p_v):
            return True
        if self._should_try_operator("swap_2_2") and self._move_swap_2_2(r_u, p_u, r_v, p_v):
            return True

        return (
            self._target_neighborhood == "inter_swap_star"
            and self._sectors_overlap(self._polar_sector(r_u), self._polar_sector(r_v))
            and self._move_swap_star(u, v, r_u, p_u, r_v, p_v)
        )

    def _process_intra_route(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """Process moves within the same route.

        Args:
            u: First node ID.
            v: Second node ID.
            r_u: Index of the route containing both nodes.
            p_u: Position of node `u`.
            r_v: Position of node `v`.
            p_v: Alias for `p_v` (redundant but kept for API consistency).

        Returns:
            bool: True if an improving intra-route move was applied.
        """
        if (
            self._should_try_operator("relocate") or self._should_try_operator("intra_relocate")
        ) and self._move_relocate(u, v, r_u, p_u, r_v, p_v):
            return True

        if (self._should_try_operator("swap") or self._should_try_operator("intra_swap")) and self._move_swap(
            u, v, r_u, p_u, r_v, p_v
        ):
            return True

        if self._should_try_operator("intra_2opt") and self._move_2opt_intra(u, v, r_u, p_u, r_v, p_v):
            return True
        if self._should_try_operator("intra_or_opt") and self._move_or_opt(r_u, p_u, self.random.choice([1, 2, 3])):
            return True
        return (
            self._should_try_operator("intra_3opt")
            and getattr(self.params, "use_3opt", False)
            and self._move_3opt_intra(u, v, r_u, p_u, r_v, p_v, self.random)
        )

    def _update_map(self, affected_indices: Set[int]):
        """
        Update node map and route loads after route modifications.
        Also updates Top-3 Insertion Cache for affected routes (Vidal 2022).

        Args:
            affected_indices: Set of route indices that were modified.
        """
        # 1. Clean up stale mappings for these routes first.
        # If we don't do this, nodes that were removed/dropped from these routes
        # will still appear as "routed" in node_map at their old (now invalid) positions.
        for node, loc in list(self.node_map.items()):
            if loc[0] in affected_indices:
                del self.node_map[node]

        # 2. Rebuild node_map for affected routes, so the routed set is current
        # before _compute_top_insertions uses it for O(1) skip checks.
        for ri in sorted(affected_indices):
            for pi, node in enumerate(self.routes[ri]):
                self.node_map[node] = (ri, pi)
            self.route_loads[ri] = self._calc_load_fresh(self.routes[ri])
            self.route_sectors.pop(ri, None)  # invalidate stale sector cache

        # Only recompute insertion cache if unvisited nodes actually exist.
        # After a move that only relocates routed nodes (relocate, swap, 2-opt),
        # every neighbor is in node_map and the cache update is pure waste.
        if any(n not in self.node_map for n in self.neighbors):
            for ri in sorted(affected_indices):
                self._compute_top_insertions(route_idx=ri)

    def _get_load_cached(self, ri: int) -> float:
        """Get cached total load of a route.

        Args:
            ri: Route index.

        Returns:
            float: Cached total load of the route.
        """
        return self.route_loads[ri]
