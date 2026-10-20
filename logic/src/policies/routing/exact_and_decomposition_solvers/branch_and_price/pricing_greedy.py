"""pricing greedy methods relocated from pricing_subproblem."""

from typing import Any, Dict, List, Optional, Set, Tuple


class PricingGreedyMixin:
    """Relocated pricing greedy methods."""

    def _evaluate_candidate_insertion(
        self,
        v: int,
        v_waste: float,
        route: List[int],
        dual_values: Dict[Any, float],
        forbidden_arcs: Set[Tuple[int, int]],
        req_successors: Dict[int, int],
        req_predecessors: Dict[int, int],
        rf_separate_pairs: Set[Tuple[int, int]],
        visited: Set[int],
        node_prizes: Optional[Dict[int, float]] = None,
    ) -> Tuple[Optional[int], float]:
        """Evaluate candidate v and find best insertion position and profit.

        Args:
            v: Candidate node to insert.
            v_waste: Waste volume of node v.
            route: Current partial route.
            dual_values: Node dual values.
            forbidden_arcs: Set of forbidden arcs.
            req_successors: Required successors map.
            req_predecessors: Required predecessors map.
            rf_separate_pairs: Ryan-Foster separation pairs.
            visited: Set of nodes already in the route.
            node_prizes: Optional fixed prizes per node.

        Returns:
            A tuple of (best_position, best_marginal_profit).
        """
        # Rule 4: Ryan-Foster separation.
        if self._violates_rf_separation(v, visited, rf_separate_pairs):
            return None, -float("inf")

        best_pos = None
        best_marginal = -float("inf")

        # ---- Marginal profit (best insertion position) ----------
        for pos in range(len(route) + 1):
            prev = self.depot if pos == 0 else route[pos - 1]
            nxt = self.depot if pos == len(route) else route[pos]

            # Rule 1: Forbidden arcs.
            if (prev, v) in forbidden_arcs or (v, nxt) in forbidden_arcs:
                continue

            # Rule 2: Required successors.
            if prev in req_successors and req_successors[prev] != v:
                continue
            if v in req_successors and req_successors[v] != nxt:
                continue

            # Rule 3: Required predecessors.
            if v in req_predecessors and req_predecessors[v] != prev:
                continue
            if nxt in req_predecessors and nxt != self.depot and req_predecessors[nxt] != v:
                continue

            detour = self.cost_matrix[prev, v] + self.cost_matrix[v, nxt] - self.cost_matrix[prev, nxt]

            revenue_component = node_prizes.get(v, 0.0) if node_prizes is not None else v_waste * self.R

            marginal = revenue_component - detour * self.C - dual_values.get(v, 0.0)

            if marginal > best_marginal:
                best_marginal = marginal
                best_pos = pos

        return best_pos, best_marginal

    def _greedy_routing(
        self,
        start_node: int,
        dual_values: Dict[Any, float],
        forbidden_arcs: Set[Tuple[int, int]],
        req_successors: Dict[int, int],
        req_predecessors: Dict[int, int],
        rf_separate_pairs: Set[Tuple[int, int]],
        rf_together_pairs: Set[Tuple[int, int]],
        capacity_cut_duals: Optional[Dict[Any, float]] = None,
        node_prizes: Optional[Dict[int, float]] = None,
    ) -> Tuple[List[int], float]:
        """Build a single route greedily from *start_node*.

        At each step the next insertion is chosen to maximise marginal profit
        (revenue + dual benefit − detour cost). All active branching
        constraints are checked before a candidate node is accepted.

        Constraint enforcement inside the construction loop:

        1. **Forbidden arc** ``(tail, v)``:
           Reject *v* if the arc from the current route tail to *v* is forbidden.

        2. **Required successor** of tail ``→ w``:
           If the current tail has a required outgoing arc to *w*, only *w*
           is a legal next node (any other *v* is rejected).

        3. **Required predecessor** of *v* ``← x``:
           Reject *v* if it requires a predecessor other than the current tail.

        4. **Ryan-Foster separate** ``(r, s)`` with ``together=False``:
           Reject *v* if adding *v* would place it in the same route as a
           node already present that is in a separation pair with *v*.

        After construction, together-pairs are checked at route-completion
        time.  Any route that contains exactly one member of a together-pair
        is discarded (reduced cost set to −∞ so it is filtered out).

        Args:
            start_node: First customer node to seed the route.
            dual_values: Node-coverage and vehicle-limit duals.
            forbidden_arcs: Set of (u, v) pairs that must not be traversed.
            req_successors: u → w meaning u must be immediately followed by w.
            req_predecessors: v → x meaning v must be immediately preceded by x.
            rf_separate_pairs: Sorted node pairs that must be in different routes.
            rf_together_pairs: Sorted node pairs that must be in the same route.
            capacity_cut_duals: Optional duals for capacity cuts.
            node_prizes: Optional fixed prizes per node.

        Returns:
            A tuple of (route_nodes, reduced_cost).
        """
        # ---- Validate start node itself ---------------------------------
        # The depot (0) is the implicit predecessor of start_node.
        if start_node in req_predecessors and req_predecessors[start_node] != self.depot:
            # start_node requires a non-depot predecessor; cannot start here.
            return [], -float("inf")

        route: List[int] = [start_node]
        current_load = self.wastes.get(start_node, 0.0)
        visited: Set[int] = {start_node}

        # ---- Greedy insertion -------------------------------------------
        while True:
            tail = route[-1]

            # If tail has a required successor that is unvisited, it is the
            # *only* valid next node; skip the full candidate scan.
            if tail in req_successors:
                forced = req_successors[tail]
                if forced not in visited and forced != self.depot:
                    forced_waste = self.wastes.get(forced, 0.0)
                    # Check that forced node's predecessor requirement is met and Ryan-Foster separation.
                    if (
                        current_load + forced_waste <= self.capacity
                        and (forced not in req_predecessors or req_predecessors[forced] == tail)
                        and not self._violates_rf_separation(forced, visited, rf_separate_pairs)
                    ):
                        route.append(forced)
                        visited.add(forced)
                        current_load += forced_waste
                        continue
                # Cannot satisfy required successor → stop extending.
                break

            # General case: score all unvisited candidates.
            best_node: Optional[Tuple[int, int]] = None  # (node, insertion_pos)
            best_profit: float = -float("inf")
            for v in range(1, self.n_nodes + 1):
                if v in visited:
                    continue

                v_waste = self.wastes.get(v, 0.0)
                if current_load + v_waste > self.capacity:
                    continue

                pos, marginal = self._evaluate_candidate_insertion(
                    v,
                    v_waste,
                    route,
                    dual_values,
                    forbidden_arcs,
                    req_successors,
                    req_predecessors,
                    rf_separate_pairs,
                    visited,
                    node_prizes,
                )

                if pos is not None and marginal > best_profit:
                    best_profit = marginal
                    best_node = (v, pos)

            if best_node is None or best_profit <= 0:
                break

            v, pos = best_node
            route.insert(pos, v)
            visited.add(v)
            current_load += self.wastes.get(v, 0.0)

        # ---- Together-pair completion check -----------------------------
        # Discard any route that contains exactly one member of a together-pair.
        for r, s in rf_together_pairs:
            r_in = r in visited
            s_in = s in visited
            if r_in != s_in:
                # Together constraint violated at route completion.
                return route, -float("inf")

        reduced_cost = self._compute_reduced_cost(route, dual_values, capacity_cut_duals, node_prizes)
        return route, reduced_cost

    @staticmethod
    def _violates_rf_separation(
        candidate: int,
        visited: Set[int],
        rf_separate_pairs: Set[Tuple[int, int]],
    ) -> bool:
        """Return True if adding *candidate* violates a Ryan-Foster separation constraint.

        A violation occurs when *candidate* is in a ``together=False`` pair
        with any node already present in ``visited``.

        Args:
            candidate: Node being evaluated for insertion.
            visited: Nodes already in the partial route.
            rf_separate_pairs: Sorted (min, max) pairs that must be separated.

        Returns:
            True if insertion would violate a separation constraint.
        """
        for already in visited:
            key = (min(candidate, already), max(candidate, already))
            if key in rf_separate_pairs:
                return True
        return False
