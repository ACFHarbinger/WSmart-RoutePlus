"""Atomic move wrappers of the local search.

:class:`LocalSearchMovesMixin` holds the
:class:`~logic.src.policies.helpers.local_search.local_search_base.LocalSearch` methods that
wrap each intra-route and inter-route move operator (relocate, swap, SWAP*, 2-opt, 2-opt*,
3-opt, Or-opt, CROSS, shift and swap variants, cross exchange, lambda interchange, cyclic
transfer, exchange and ejection chains, relocate chain, three-permutation) and the
unrouted-insert and drop moves. Moved unchanged from ``local_search_base.py``.
"""

from __future__ import annotations

import random

from logic.src.policies.helpers.operators.inter_route_local_search import (
    move_2opt_star,
    move_cross,
    move_swap_star,
    shift_2_0,
    swap_2_1,
    swap_2_2,
)
from logic.src.policies.helpers.operators.inter_route_local_search.ejection_chain import ejection_chain
from logic.src.policies.helpers.operators.intra_route_local_search import (
    move_2opt_intra,
    move_3opt_intra,
    move_or_opt,
    move_relocate,
    move_swap,
)


class LocalSearchMovesMixin:
    """Move-operator wrappers of :class:`LocalSearch` (moved unchanged from ``local_search_base.py``)."""

    def _move_relocate(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """Relocate a node to a new position.

        Args:
            u: Node ID to relocate.
            v: Target neighbor node ID.
            r_u: Current route index of `u`.
            p_u: Current position of `u`.
            r_v: Target route index.
            p_v: Target position (usually after `v`).

        Returns:
            bool: True if the relocate move was accepted.
        """
        return move_relocate(self, u, v, r_u, p_u, r_v, p_v)

    def _move_swap(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """Swap two nodes.

        Args:
            u: First node ID.
            v: Second node ID.
            r_u: Route index of `u`.
            p_u: Position of `u`.
            r_v: Route index of `v`.
            p_v: Position of `v`.

        Returns:
            bool: True if the swap move was accepted.
        """
        return move_swap(self, u, v, r_u, p_u, r_v, p_v)

    def _move_swap_star(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """Apply SWAP* operator (best insertion swap).

        Args:
            u: First node ID.
            v: Second node ID.
            r_u: Route index of `u`.
            p_u: Position of `u`.
            r_v: Route index of `v`.
            p_v: Position of `v`.

        Returns:
            bool: True if the SWAP* move was accepted.
        """
        return move_swap_star(self, u, v, r_u, p_u, r_v, p_v)

    def _move_3opt_intra(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int, rng: random.Random) -> bool:
        """Apply intra-route 3-opt move.

        Args:
            u: First node ID.
            v: Second node ID.
            r_u: Route index.
            p_u: First position.
            r_v: Route index (same as `r_u`).
            p_v: Second position.
            rng: Random number generator for tie-breaking.

        Returns:
            bool: True if the 3-opt move was accepted.
        """
        return move_3opt_intra(self, u, v, r_u, p_u, r_v, p_v, rng)

    def _move_2opt_star(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """Apply inter-route 2-opt* move.

        Args:
            u: First node ID.
            v: Second node ID.
            r_u: First route index.
            p_u: First position.
            r_v: Second route index.
            p_v: Second position.

        Returns:
            bool: True if the 2-opt* move was accepted.
        """
        return move_2opt_star(self, u, v, r_u, p_u, r_v, p_v)

    def _move_2opt_intra(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """Apply intra-route 2-opt move.

        Args:
            u: First node ID.
            v: Second node ID.
            r_u: Route index.
            p_u: First position.
            r_v: Route index (same as `r_u`).
            p_v: Second position.

        Returns:
            bool: True if the intra-route 2-opt move was accepted.
        """
        return move_2opt_intra(self, u, v, r_u, p_u, r_v, p_v)

    def _move_or_opt(self, r_idx: int, pos: int, chain_len: int) -> bool:
        """Apply Or-opt move.

        Args:
            r_idx: Route index.
            pos: Starting position.
            chain_len: Length of the chain to relocate.

        Returns:
            bool: True if the Or-opt move was accepted.
        """
        return move_or_opt(self, r_idx, pos, chain_len)

    def _move_cross(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """Apply Cross move.

        Args:
            u: First node ID.
            v: Second node ID.
            r_u: First route index.
            p_u: First position.
            r_v: Second route index.
            p_v: Second position.

        Returns:
            bool: True if the Cross move was accepted.
        """
        return move_cross(self, u, v, r_u, p_u, r_v, p_v)

    def _move_shift_2_0(self, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """Apply Shift(2,0) move (relocate two nodes).

        Args:
            r_u: Source route index.
            p_u: Starting position of the 2-node chain.
            r_v: Target route index.
            p_v: Target insertion position.

        Returns:
            bool: True if the Shift(2,0) move was accepted.
        """
        return shift_2_0(self, r_u, p_u, r_v, p_v)

    def _move_swap_2_1(self, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """Apply Swap(2,1) move (swap 2 nodes with 1 node).

        Args:
            r_u: First route index.
            p_u: Starting position of the 2-node chain.
            r_v: Second route index.
            p_v: Position of the single node.

        Returns:
            bool: True if the Swap(2,1) move was accepted.
        """
        return swap_2_1(self, r_u, p_u, r_v, p_v)

    def _move_swap_2_2(self, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """Apply Swap(2,2) move (swap two pairs of nodes).

        Args:
            r_u: First route index.
            p_u: Starting position of the first 2-node chain.
            r_v: Second route index.
            p_v: Starting position of the second 2-node chain.

        Returns:
            bool: True if the Swap(2,2) move was accepted.
        """
        return swap_2_2(self, r_u, p_u, r_v, p_v)

    def _try_ejection_chain(self, r_u: int) -> bool:
        """Docstring.

        Args:
            r_u: Starting route index for the ejection chain.

        Returns:
            bool: True if the Ejection Chain move was accepted.
        """
        return ejection_chain(self, r_u)

    def _move_unrouted_insert(self, node: int, route_idx: int, _hint_pos: int) -> bool:
        """
        Try inserting an unrouted node back into a route in O(1) time (VRPP operator).

        Evaluates inserting 'node' (currently unvisited) into 'route_idx' at the best position.
        Uses Top-3 Insertion Cache for O(1) position lookup instead of O(N) loop.
        Uses penalized search: considers both profit gain and capacity penalties.

        Args:
            node: Unvisited node to potentially insert.
            route_idx: Route to insert into.
            _hint_pos: Position hint (unused).

        Returns:
            bool: True if insertion improves penalized profit.
        """
        route = self.routes[route_idx]

        # 1. Base Case: Empty route
        if not route:
            cost_delta = 2 * self.get_dist(0, node) * self.C
            best_pos = 0
        else:
            # 2. O(1) Cache Query - No loops required!
            # Since 'node' is unrouted, all cached positions are valid (no removal offsets)
            if node in self.top_insertions and route_idx in self.top_insertions[node]:
                # Always use first cached position (guaranteed valid for unrouted nodes)
                best_delta_raw, best_pos = self.top_insertions[node][route_idx][0]
                cost_delta = best_delta_raw * self.C
            else:
                # Cache miss: compute insertion cost directly
                best_pos, best_delta = 0, float("inf")
                for pos in range(len(route) + 1):
                    prev = route[pos - 1] if pos > 0 else 0
                    nxt = route[pos] if pos < len(route) else 0
                    delta = self.get_dist(prev, node) + self.get_dist(node, nxt) - self.get_dist(prev, nxt)
                    if delta < best_delta:
                        best_delta, best_pos = delta, pos
                cost_delta = best_delta * self.C

        profit_delta = self.waste.get(node, 0) * self.R
        load_delta = self.waste.get(node, 0)

        # 3. Penalties & Evaluation (Standard HGS logic)
        penalty_weight = getattr(self, "penalty_capacity", 1.0)
        current_load = self._get_load_cached(route_idx)
        new_load = current_load + load_delta

        old_penalty = penalty_weight * max(0.0, current_load - self.Q)
        new_penalty = penalty_weight * max(0.0, new_load - self.Q)
        penalty_delta = new_penalty - old_penalty

        # Total change in objective: profit gain - cost increase - penalty increase
        total_change = profit_delta - cost_delta - penalty_delta

        # Accept if improves penalized profit
        if total_change > 1e-4:
            if not route:
                self.routes[route_idx] = [node]
            else:
                self.routes[route_idx].insert(best_pos, node)
            self._update_map({route_idx})
            return True

        return False

    def _move_drop(self, node: int, route_idx: int, pos: int) -> bool:
        """
        Try dropping a routed node in O(1) time (VRPP operator).

        Evaluates removing 'node' from 'route_idx' at 'pos'.
        Uses penalized search: considers both profit loss and capacity penalty reduction.

        Args:
            node: Routed node to potentially drop.
            route_idx: Route to drop from.
            pos: Position of the node in the route.

        Returns:
            bool: True if dropping improves penalized profit.
        """
        route = self.routes[route_idx]

        # Validation
        if pos < 0 or pos >= len(route) or route[pos] != node:
            return False

        if getattr(self, "mandatory_nodes", None) and node in self.mandatory_nodes:
            return False

        prev = route[pos - 1] if pos > 0 else 0
        nxt = route[pos + 1] if pos < len(route) - 1 else 0

        # 1. Cost & Profit Deltas
        # Cost saving is negative delta (we save distance by bypassing the node)
        delta_dist = self.get_dist(prev, nxt) - (self.get_dist(prev, node) + self.get_dist(node, nxt))
        cost_delta = delta_dist * self.C  # Guaranteed to be <= 0 by triangle inequality

        # Profit loss is negative delta
        profit_delta = -self.waste.get(node, 0) * self.R
        load_delta = -self.waste.get(node, 0)

        # 2. Penalties (Standard HGS logic)
        penalty_weight = getattr(self, "penalty_capacity", 1.0)
        current_load = self._get_load_cached(route_idx)
        new_load = current_load + load_delta

        old_penalty = penalty_weight * max(0.0, current_load - self.Q)
        new_penalty = penalty_weight * max(0.0, new_load - self.Q)
        penalty_delta = new_penalty - old_penalty

        # Total change in objective: profit gain - cost increase - penalty increase
        total_change = profit_delta - cost_delta - penalty_delta

        # 3. Acceptance
        # Accept if dropping the node strictly improves the penalized objective
        if total_change > 1e-4:
            route.pop(pos)
            self._update_map({route_idx})
            return True

        return False
