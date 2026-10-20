"""solver support methods relocated from solver."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Set, Tuple

import numpy as np

from logic.src.policies.helpers.solvers_and_matheuristics.common.route import Route

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class RCSPPSolverSupportMixin:
    """Relocated solver support methods."""

    def _heuristic_pricing(
        self,
        max_routes: int,
        node_duals: Dict[int, float],
        rcc_duals: Dict,
        forbidden_arcs: FrozenSet,
    ) -> List[Route]:
        """Fast heuristic column generator for large instances.

        Uses savings-based construction + greedy augmentation to find
        profitable routes quickly. Runs in O(n²) vs O(n × 2^k) for exact DP.

        Constructs routes by:
        1. Sorting nodes by reduced cost (revenue - dual) descending
        2. Greedily building routes using nearest-neighbor with capacity check
        3. Only returning routes with positive reduced cost

        Args:
            max_routes: Maximum number of routes to generate.
            node_duals: Dual values for nodes.
            rcc_duals: Dual values for route-covering constraints.
            forbidden_arcs: Set of forbidden arcs.

        Returns:
            List of routes with positive reduced cost.
        """
        if self.is_farkas:
            return []  # heuristic doesn't handle Farkas pricing

        # Reduced cost per node: R*waste - dual
        node_rc: Dict[int, float] = {}
        for n in range(1, self.n_nodes + 1):
            rev = self.node_prizes.get(n, 0.0) if self.node_prizes is not None else self.wastes.get(n, 0.0) * self.R
            node_rc[n] = rev - node_duals.get(n, 0.0)

        # Only consider nodes with positive reduced cost
        candidates = sorted(
            [n for n in range(1, self.n_nodes + 1) if node_rc[n] > 0],
            key=lambda n: node_rc[n],
            reverse=True,
        )
        if not candidates:
            return []

        routes: List[Route] = []
        seen_sets: set = set()

        # Strategy 1: Savings-based construction (Clarke-Wright)
        # For each pair of high-value nodes, check if combining into one route is profitable
        for seed in candidates[: min(20, len(candidates))]:
            route_nodes = [seed]
            load = self.wastes.get(seed, 0.0)
            route_rc = node_rc[seed] - self.cost_matrix[self.depot, seed] * self.C

            # Extend greedily: nearest neighbor with positive marginal RC
            current = seed
            visited = {seed}
            while True:
                best_next, best_gain = -1, 0.0
                for v in candidates:
                    if v in visited or (current, v) in forbidden_arcs:
                        continue
                    if load + self.wastes.get(v, 0.0) > self.capacity + 1e-6:
                        continue
                    # Marginal gain: RC of v minus detour cost
                    detour = (
                        self.cost_matrix[current, v]
                        + self.cost_matrix[v, self.depot]
                        - self.cost_matrix[current, self.depot]
                    )
                    gain = node_rc[v] - self.C * detour
                    if gain > best_gain:
                        best_gain, best_next = gain, v
                if best_next == -1:
                    break
                route_nodes.append(best_next)
                visited.add(best_next)
                load += self.wastes.get(best_next, 0.0)
                route_rc += best_gain
                current = best_next

            # Complete route: return to depot
            route_rc -= self.cost_matrix[current, self.depot] * self.C
            if not self.is_farkas and self.vehicle_dual > 1e-9:
                route_rc -= self.vehicle_dual

            # Only add if positive RC and not seen before
            key = frozenset(route_nodes)
            if route_rc > 1e-6 and key not in seen_sets:
                seen_sets.add(key)
                rt = self._compute_route_details(route_nodes)
                rt.reduced_cost = route_rc
                routes.append(rt)
                if len(routes) >= max_routes:
                    break

        return routes

    def _compute_completion_bounds(self) -> None:  # noqa: C901
        """Compute backward bounds used to aggressively prune DP states.

        Calculates the maximum possible reduced cost from each node to the depot,
        incorporating duals and penalties for cut violations.

        Returns:
            None. Updates self.bounds_to and self.bounds_from.
        """

        # Incorporate RCC duals into completion bounds for tighter pruning.
        # RCC duals are arc-crossing penalties with a clear per-edge interpretation.
        self.bounds_to = np.zeros(self.n_nodes + 1)
        self.bounds_from = np.zeros(self.n_nodes + 1)
        rcc_duals = getattr(self, "_rcc_duals_for_bounds", {})
        nodes = list(range(self.n_nodes + 1))
        for _ in range(self.n_nodes):
            changed = False
            for i in nodes:
                # To Depot
                for j in nodes:
                    if i == j:
                        continue

                    edge_cost = 0.0 if self.is_farkas else (self.cost_matrix[i, j] * self.C)
                    if j == self.depot and not self.is_farkas and self.vehicle_dual > 1e-9:
                        edge_cost += self.vehicle_dual
                    if self.is_farkas:
                        node_rev = 0.0
                    elif self.node_prizes is not None:
                        node_rev = self.node_prizes.get(j, 0.0)
                    else:
                        node_rev = self.wastes.get(j, 0.0) * self.R
                    node_dual = self.dual_values.get(j, 0.0)

                    # Add RCC dual contributions: for each cut set S containing j,
                    # subtract the dual if crossing the cut boundary.
                    rcc_penalty = sum(mu for S, mu in rcc_duals.items() if j in S and i not in S)

                    if self.is_farkas:
                        val = (node_dual + rcc_penalty) + self.bounds_to[j]
                    else:
                        val = (node_rev - edge_cost - node_dual - rcc_penalty) + self.bounds_to[j]

                    if val > self.bounds_to[i]:
                        self.bounds_to[i] = val
                        changed = True

                # From Depot
                for j in nodes:
                    if i == j:
                        continue

                    edge_cost = 0.0 if self.is_farkas else (self.cost_matrix[j, i] * self.C)
                    if self.is_farkas:
                        node_rev = 0.0
                    elif self.node_prizes is not None:
                        node_rev = self.node_prizes.get(i, 0.0)
                    else:
                        node_rev = self.wastes.get(i, 0.0) * self.R
                    node_dual = self.dual_values.get(i, 0.0)

                    rcc_penalty = sum(mu for S, mu in rcc_duals.items() if i in S and j not in S)

                    if self.is_farkas:
                        val = (node_dual + rcc_penalty) + self.bounds_from[j]
                    else:
                        val = (node_rev - edge_cost - node_dual - rcc_penalty) + self.bounds_from[j]

                    if val > self.bounds_from[i]:
                        self.bounds_from[i] = val
                        changed = True
            if not changed:
                break

    def _preprocess_constraints(
        self, constraints: List[Any]
    ) -> Tuple[FrozenSet[Tuple[int, int]], Dict[int, int], Dict[int, int], Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """Convert generic branching constraints into efficient lookup structures.

        Args:
            constraints: List of active branching constraints from the B&B tree.

        Returns:
            A tuple containing:
                - forbidden_arcs: Set of (u, v) pairs that cannot be used.
                - req_succ: Mapping of u to its required successor v.
                - req_pred: Mapping of v to its required predecessor u.
                - rf_separate: Set of Ryan-Foster node pairs that must be separate.
                - rf_together: Set of Ryan-Foster node pairs that must be together.
        """

        forbidden: Set[Tuple[int, int]] = set()
        req_succ: Dict[int, int] = {}
        req_pred: Dict[int, int] = {}
        rf_separate: Set[Tuple[int, int]] = set()
        rf_together: Set[Tuple[int, int]] = set()
        for c in constraints:
            if hasattr(c, "must_use"):
                if not c.must_use:
                    forbidden.add((c.u, c.v))
                else:
                    req_succ[c.u] = c.v
                    req_pred[c.v] = c.u
            elif hasattr(c, "together"):
                pair = tuple(sorted((c.node_r, c.node_s)))
                if not c.together:
                    rf_separate.add(pair)
                else:
                    rf_together.add(pair)
            elif hasattr(c, "forced") and hasattr(c, "node") and not c.forced:
                # y_v = 0 branch: no column may visit v. The master disables existing
                # columns through v; pricing must not regenerate them.
                forbidden.update((u, c.node) for u in range(self.n_nodes + 1) if u != c.node)
        return frozenset(forbidden), req_succ, req_pred, rf_separate, rf_together

    def _compute_route_details(self, nodes: List[int]) -> Route:
        """Compute cost/revenue/load for a customer-only node list.

        Args:
            nodes: Ordered list of customer nodes visited (no depot bookends).

        Returns:
            Route object with computed distance-based costs and profits.
        """

        cost = 0.0
        prev = self.depot
        for n in nodes:
            cost += self.cost_matrix[prev, n] * self.C
            prev = n
        cost += self.cost_matrix[prev, self.depot] * self.C

        if self.node_prizes is not None:
            revenue = sum(self.node_prizes.get(n, 0.0) for n in nodes)
        else:
            revenue = sum(self.wastes.get(n, 0.0) for n in nodes) * self.R

        waste = sum(self.wastes.get(n, 0.0) for n in nodes)
        # Own the node list: callers reuse and mutate theirs (e.g. insertion operators),
        # which would otherwise change a column's nodes but not its cost/revenue.
        return Route(list(nodes), cost, revenue, waste, set(nodes))
