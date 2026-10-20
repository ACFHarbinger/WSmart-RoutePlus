"""Tier-1 local search operators and move gains for FILO.

This module provides the move evaluation and operator mixin for the FILO local search algorithm.
"""

import random
from typing import List, Set

K_NEIGHBORS: int = 20  # Max neighbor fan-out per vertex


class FILOOperatorsMixin:
    """Tier-1 move evaluation and gain calculation mixin for FILOLocalSearch."""

    def _get_gain_relocate(self, u: int, ri: int, pi: int, rj: int, pj: int) -> float:
        """Cost delta for moving *u* from route *ri* to position *pj* in *rj*.

        Args:
            u: Node to move.
            ri: Current route index.
            pi: Current position index in route ri.
            rj: Target route index.
            pj: Target position index in route rj.

        Returns:
            float: Cost delta (gain) of the relocate move.
        """
        route_i, route_j = self.routes[ri], self.routes[rj]
        prev_i = route_i[pi - 1] if pi > 0 else 0
        nxt_i = route_i[pi + 1] if pi < len(route_i) - 1 else 0
        removal = self.d[prev_i, u] + self.d[u, nxt_i] - self.d[prev_i, nxt_i]

        prev_j = route_j[pj - 1] if pj > 0 else 0
        nxt_j = route_j[pj] if pj < len(route_j) else 0
        insertion = self.d[prev_j, u] + self.d[u, nxt_j] - self.d[prev_j, nxt_j]
        return (insertion - removal) * self.C

    def _get_gain_swap(self, u: int, ri: int, pi: int, v: int, rj: int, pj: int) -> float:
        """Cost delta for swapping nodes *u* and *v*.

        Args:
            u: First node to swap.
            ri: Route index of first node.
            pi: Position index of first node.
            v: Second node to swap.
            rj: Route index of second node.
            pj: Position index of second node.

        Returns:
            float: Cost delta (gain) of the swap move.
        """
        if ri == rj and abs(pi - pj) == 1:
            p1, p2 = min(pi, pj), max(pi, pj)
            r = self.routes[ri]
            n1, n2 = r[p1], r[p2]
            prev1 = r[p1 - 1] if p1 > 0 else 0
            nxt2 = r[p2 + 1] if p2 < len(r) - 1 else 0
            old = self.d[prev1, n1] + self.d[n1, n2] + self.d[n2, nxt2]
            new = self.d[prev1, n2] + self.d[n2, n1] + self.d[n1, nxt2]
            return (new - old) * self.C

        route_i, route_j = self.routes[ri], self.routes[rj]
        prev_u = route_i[pi - 1] if pi > 0 else 0
        nxt_u = route_i[pi + 1] if pi < len(route_i) - 1 else 0
        old_u = self.d[prev_u, u] + self.d[u, nxt_u]

        prev_v = route_j[pj - 1] if pj > 0 else 0
        nxt_v = route_j[pj + 1] if pj < len(route_j) - 1 else 0
        old_v = self.d[prev_v, v] + self.d[v, nxt_v]

        new_u_in_j = self.d[prev_v, u] + self.d[u, nxt_v]
        new_v_in_i = self.d[prev_u, v] + self.d[v, nxt_u]
        return (new_u_in_j + new_v_in_i - old_u - old_v) * self.C

    def _get_gain_2opt(self, ri: int, pi: int, pj: int) -> float:
        """Cost delta for reversing segment ``routes[ri][pi..pj]``.

        Args:
            ri: Route index.
            pi: Start position index of the segment.
            pj: End position index of the segment.

        Returns:
            float: Cost delta (gain) of the 2-opt move.
        """
        if pi >= pj:
            return 0.0
        route = self.routes[ri]
        prev_i = route[pi - 1] if pi > 0 else 0
        nxt_j = route[pj + 1] if pj < len(route) - 1 else 0
        old = self.d[prev_i, route[pi]] + self.d[route[pj], nxt_j]
        new = self.d[prev_i, route[pj]] + self.d[route[pi], nxt_j]
        return (new - old) * self.C

    def _get_gain_tails(self, ri: int, pi: int, rj: int, pj: int) -> float:
        """Cost delta for swapping tails ``routes[ri][pi:]`` <-> ``routes[rj][pj:]``.

        Args:
            ri: First route index.
            pi: Start position of tail in first route.
            rj: Second route index.
            pj: Start position of tail in second route.

        Returns:
            float: Cost delta (gain) of the tails swap.
        """
        route_i, route_j = self.routes[ri], self.routes[rj]
        u, v = route_i[pi], route_j[pj]
        prev_u = route_i[pi - 1] if pi > 0 else 0
        prev_v = route_j[pj - 1] if pj > 0 else 0
        old = self.d[prev_u, u] + self.d[prev_v, v]
        new = self.d[prev_u, v] + self.d[prev_v, u]
        return (new - old) * self.C

    def _get_gain_split(self, ri: int, pi: int) -> float:
        """Cost delta for splitting route *ri* at position *pi*.

        Args:
            ri: Route index.
            pi: Position index to split the route at.

        Returns:
            float: Cost delta (gain) of the split move.
        """
        route = self.routes[ri]
        u = route[pi]
        prev_u = route[pi - 1] if pi > 0 else 0
        old = self.d[prev_u, u]
        new = self.d[prev_u, 0] + self.d[0, u]
        return (new - old) * self.C

    # ------------------------------------------------------------------
    # Gamma-gated neighbor iteration
    # ------------------------------------------------------------------

    def _gamma_neighbors(self, u: int) -> List[int]:
        """Return neighbors of *u* filtered by gamma sparsification.

        Args:
            u: Node ID to get neighbors for.

        Returns:
            List[int]: Filtered list of neighbors.
        """
        gamma_u = self._node_gamma.get(u, 1.0)
        return [v for v in self.neighbors[u][:K_NEIGHBORS] if random.random() <= gamma_u]

    # ------------------------------------------------------------------
    # Tier-1: SVC-bounded RVND operators (O(1) lookups)
    # ------------------------------------------------------------------

    def _try_relocate(self, active_nodes: Set[int]) -> bool:
        """Exhaustively search for the best relocate move in the SVC kernel.

        Args:
            active_nodes: Set of active node IDs to search.

        Returns:
            bool: True if a move was made, False otherwise.
        """
        best_gain, best_move = 0.0, None
        for u in active_nodes:
            ri, pi = self._find_node(u)
            if ri == -1:
                continue
            w_u = self.waste.get(u, 0.0)
            for v in self._gamma_neighbors(u):
                v_ri, v_pi = self._find_node(v)
                if v_ri == -1:
                    continue
                target_pj = v_pi + 1
                if ri != v_ri and self._route_loads[v_ri] + w_u > self.Q:
                    continue
                g = self._get_gain_relocate(u, ri, pi, v_ri, target_pj)
                if g < best_gain - 1e-6:
                    best_gain = g
                    best_move = (u, ri, pi, v_ri, target_pj)
        if best_move:
            u, ri, pi, rj, pj = best_move
            w_u = self.waste.get(u, 0.0)
            self.routes[ri].pop(pi)
            self._route_loads[ri] -= w_u
            adj_pj = pj if ri != rj or pi >= pj else pj - 1
            self.routes[rj].insert(adj_pj, u)
            self._route_loads[rj] += w_u
            self._rebuild_route_positions(ri)
            self._rebuild_route_positions(rj)
            return True
        return False

    def _try_swap(self, active_nodes: Set[int]) -> bool:
        """Exhaustively search for the best swap move in the SVC kernel.

        Args:
            active_nodes: Set of active node IDs to search.

        Returns:
            bool: True if a move was made, False otherwise.
        """
        best_gain, best_move = 0.0, None
        for u in active_nodes:
            ri, pi = self._find_node(u)
            if ri == -1:
                continue
            w_u = self.waste.get(u, 0.0)
            load_i = self._route_loads[ri]
            for v in self._gamma_neighbors(u):
                if u == v:
                    continue
                v_ri, v_pi = self._find_node(v)
                if v_ri == -1:
                    continue
                w_v = self.waste.get(v, 0.0)
                if ri != v_ri:
                    load_j = self._route_loads[v_ri]
                    if load_i - w_u + w_v > self.Q or load_j - w_v + w_u > self.Q:
                        continue
                g = self._get_gain_swap(u, ri, pi, v, v_ri, v_pi)
                if g < best_gain - 1e-6:
                    best_gain = g
                    best_move = (u, ri, pi, v, v_ri, v_pi)
        if best_move:
            u, ri, pi, v, v_ri, v_pi = best_move
            w_u, w_v = self.waste.get(u, 0.0), self.waste.get(v, 0.0)
            self.routes[ri][pi], self.routes[v_ri][v_pi] = v, u
            if ri != v_ri:
                self._route_loads[ri] += w_v - w_u
                self._route_loads[v_ri] += w_u - w_v
            self.node_to_route_pos[v] = (ri, pi)
            self.node_to_route_pos[u] = (v_ri, v_pi)
            return True
        return False

    def _try_2opt(self, active_nodes: Set[int]) -> bool:
        """Exhaustively search for the best 2-opt move in the SVC kernel.

        Args:
            active_nodes: Set of active node IDs to search.

        Returns:
            bool: True if a move was made, False otherwise.
        """
        best_gain, best_move = 0.0, None
        for u in active_nodes:
            ri, pi = self._find_node(u)
            if ri == -1:
                continue
            for v in self._gamma_neighbors(u):
                v_ri, v_pi = self._find_node(v)
                if v_ri == -1 or ri != v_ri or abs(pi - v_pi) <= 1:
                    continue
                p1, p2 = min(pi, v_pi), max(pi, v_pi)
                g = self._get_gain_2opt(ri, p1, p2)
                if g < best_gain - 1e-6:
                    best_gain = g
                    best_move = (ri, p1, p2)
        if best_move:
            target_ri, p1, p2 = best_move
            self.routes[target_ri][p1 : p2 + 1] = reversed(self.routes[target_ri][p1 : p2 + 1])
            # Loads unchanged (intra-route); rebuild positions only
            self._rebuild_route_positions(target_ri)
            return True
        return False

    def _try_tails(self, active_nodes: Set[int]) -> bool:
        """Exhaustively search for the best tails (2-opt*) move in the SVC kernel.

        Args:
            active_nodes: Set of active node IDs to search.

        Returns:
            bool: True if a move was made, False otherwise.
        """
        best_gain, best_move = 0.0, None
        for u in active_nodes:
            ri, pi = self._find_node(u)
            if ri == -1:
                continue
            load_i = self._route_loads[ri]
            tail_i_w = sum(self.waste.get(x, 0.0) for x in self.routes[ri][pi:])
            for v in self._gamma_neighbors(u):
                v_ri, v_pi = self._find_node(v)
                if v_ri == -1 or ri == v_ri:
                    continue
                load_j = self._route_loads[v_ri]
                tail_j_w = sum(self.waste.get(x, 0.0) for x in self.routes[v_ri][v_pi:])
                if load_i - tail_i_w + tail_j_w > self.Q or load_j - tail_j_w + tail_i_w > self.Q:
                    continue
                g = self._get_gain_tails(ri, pi, v_ri, v_pi)
                if g < best_gain - 1e-6:
                    best_gain = g
                    best_move = (ri, pi, v_ri, v_pi, tail_i_w, tail_j_w)
        if best_move:
            ri, pi, rj, pj, tail_i_w, tail_j_w = best_move
            t_i, t_j = self.routes[ri][pi:], self.routes[rj][pj:]
            self.routes[ri] = self.routes[ri][:pi] + t_j
            self.routes[rj] = self.routes[rj][:pj] + t_i
            self._route_loads[ri] += tail_j_w - tail_i_w
            self._route_loads[rj] += tail_i_w - tail_j_w
            self._rebuild_route_positions(ri)
            self._rebuild_route_positions(rj)
            return True
        return False

    def _try_split(self, active_nodes: Set[int]) -> bool:
        """Exhaustively search for the best split move in the SVC kernel.

        Args:
            active_nodes: Set of active node IDs to search.

        Returns:
            bool: True if a move was made, False otherwise.
        """
        best_gain, best_move = 0.0, None
        for u in active_nodes:
            if random.random() > self._node_gamma.get(u, 1.0):
                continue
            ri, pi = self._find_node(u)
            if ri == -1:
                continue
            if len(self.routes[ri]) < 3 or pi == 0 or pi >= len(self.routes[ri]) - 1:
                continue
            g = self._get_gain_split(ri, pi)
            if g < best_gain - 1e-6:
                best_gain = g
                best_move = (ri, pi)
        if best_move:
            target_ri, target_pi = best_move
            new_route = self.routes[target_ri][target_pi:]
            self.routes[target_ri] = self.routes[target_ri][:target_pi]
            # Compute new route load from the split tail
            new_load = sum(self.waste.get(n, 0.0) for n in new_route)
            self._route_loads[target_ri] -= new_load
            self._route_loads.append(new_load)
            self.routes.append(new_route)
            self._rebuild_route_positions(target_ri)
            new_ri = len(self.routes) - 1
            self._rebuild_route_positions(new_ri)
            return True
        return False
