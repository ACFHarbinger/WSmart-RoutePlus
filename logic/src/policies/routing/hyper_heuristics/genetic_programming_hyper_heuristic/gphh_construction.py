"""Constructive insertion for the genetic-programming hyper-heuristic, mixed into GPHHSolver as GPHHConstructionMixin."""

from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

import numpy as np

from .tree import (
    GPNode,
    to_callable,
)


class GPHHConstructionMixin:
    # ------------------------------------------------------------------
    # Constructive solution building
    # ------------------------------------------------------------------

    def _construct_solution(
        self,
        tree: GPNode,
        nodes: List[int],
        wastes: Dict[int, float],
        mandatory: Set[int],
        dm: np.ndarray,
        knn: Dict[int, List[int]],
    ) -> List[List[int]]:
        """
        Build a VRPP solution by greedily inserting nodes scored by the GP tree.

        **K-NN Candidate Filtering** (scaling fix):

        For each existing route, instead of evaluating all N unvisited nodes,
        only the K-nearest neighbours of the route's *first and last* nodes
        are considered as candidates.  This reduces the inner loop from O(N)
        to O(K) candidates per route, giving O(N × K × R) total evaluations
        instead of O(N² × R).

        For opening a new route, the K-nearest neighbours of the depot (node 0)
        are used as candidates, since the depot-return cost dominates.

        **No Profitability Gate**:

        All capacity-feasible candidates are passed to the GP tree for scoring.
        The tree must learn to assign heavily negative scores to unprofitable
        insertions.  This allows the GP to discover strategies that make
        temporarily unprofitable moves to access profitable spatial clusters.

        The loop terminates when no candidate produces a score above
        ``float("-inf")`` (i.e., no feasible candidate exists).  Mandatory nodes
        are inserted even when they would otherwise have no scored candidate by
        falling back to a depot-anchored forced insertion.

        Args:
            tree: GP scoring function.
            nodes: Customer node indices to consider.
            wastes: Waste/value of each node.
            mandatory: Nodes that must be visited regardless.
            dm: Distance matrix for this instance.
            knn: K-NN index keyed by node index (includes depot 0).

        Returns:
            Constructed routes (list of node-index lists).
        """
        routes: List[List[int]] = []
        route_loads: List[float] = []  # Cached loads for O(1) capacity checks
        visited: Set[int] = set()
        cap = self.capacity

        # Pre-index nodes as a set for O(1) membership checks
        node_set: Set[int] = set(nodes)

        # Compile the GP tree once at the start of construction for speed
        func = to_callable(tree)
        while True:
            best_score = -float("inf")
            best_action: Optional[Tuple[int, int, int]] = None  # (node, route_idx, pos)

            # ------------------------------------------------------------------
            # Score candidates for each existing route
            # ------------------------------------------------------------------
            for ri, route in enumerate(routes):
                # Expanded K-NN search: evaluate neighbours of EVERY node in the
                # current route to prevent spatial blindspots.
                candidates = set()
                for node_in_route in route:
                    candidates.update(knn.get(node_in_route, []))
                candidates = (candidates & node_set) - visited

                for node in candidates:
                    node_waste = wastes.get(node, 0.0)
                    if route_loads[ri] + node_waste > cap:
                        continue  # Capacity violation — skip

                    pos, delta = self._cheapest_insertion(route, node, dm)
                    node_revenue = node_waste * self.R

                    # Direct call to compiled lambda (bypasses dict construction and recursion)
                    score = func(
                        node_profit=node_revenue,
                        distance_to_route=self._min_distance_to_route(node, route, dm),
                        insertion_cost=delta,
                        remaining_capacity=cap - route_loads[ri] - node_waste,
                    )

                    if score > best_score:
                        best_score = score
                        best_action = (node, ri, pos)

            # ------------------------------------------------------------------
            # Score candidates for opening a new route
            # ------------------------------------------------------------------
            # Use K-NN of the depot as candidates: nodes closest to the depot
            # are cheapest to anchor as new-route starters.
            depot_candidates = (set(knn.get(0, [])) & node_set) - visited

            for node in depot_candidates:
                node_waste = wastes.get(node, 0.0)
                if node_waste > cap:
                    continue  # Node alone exceeds capacity

                delta_new = float(dm[0][node] + dm[node][0])
                node_revenue = node_waste * self.R

                # Use compiled lambda for speed
                score = func(
                    node_profit=node_revenue,
                    distance_to_route=dm[0][node],
                    insertion_cost=delta_new,
                    remaining_capacity=cap - node_waste,
                )

                if score > best_score:
                    best_score = score
                    best_action = (node, len(routes), 0)

            # ------------------------------------------------------------------
            # Execute best insertion or handle mandatory fallback / stop
            # ------------------------------------------------------------------
            if best_action is None:
                # Force-insert any mandatory nodes not yet visited
                unvisited_mandatory = (mandatory & node_set) - visited
                if not unvisited_mandatory:
                    break

                # Forced insertion: pick mandatory node with smallest depot cost
                forced_node = min(
                    unvisited_mandatory,
                    key=lambda n: float(dm[0][n] + dm[n][0]),
                )
                routes.append([forced_node])
                route_loads.append(wastes.get(forced_node, 0.0))
                visited.add(forced_node)
                continue

            ins_node, ins_ri, ins_pos = best_action
            if ins_ri == len(routes):
                routes.append([ins_node])
                route_loads.append(wastes.get(ins_node, 0.0))
            else:
                routes[ins_ri].insert(ins_pos, ins_node)
                route_loads[ins_ri] += wastes.get(ins_node, 0.0)

            visited.add(ins_node)

        return [r for r in routes if r]

    def _build_insertion_context(
        self,
        node_revenue: float,
        route: List[int],
        node: int,
        insertion_cost: float,
        dm: np.ndarray,
        remaining_capacity: float,
    ) -> Dict[str, float]:
        """
        Build the local context dictionary for scoring a candidate insertion.

        All four keys are synchronised with ``tree._TERMINALS``.

        Args:
            node_revenue: Revenue of the candidate node (wastes[n] × R).
            route: Current route node list (may be empty for new routes).
            node: Candidate node index.
            insertion_cost: Delta route distance at cheapest position.
            dm: Distance matrix.
            remaining_capacity: Capacity left after insertion.

        Returns:
            Context dictionary with 4 terminal features.
        """
        return {
            "node_profit": node_revenue,
            "distance_to_route": self._min_distance_to_route(node, route, dm),
            "insertion_cost": insertion_cost,
            "remaining_capacity": remaining_capacity,
        }

    # ------------------------------------------------------------------
    # Cheapest insertion helper
    # ------------------------------------------------------------------

    def _cheapest_insertion(self, route: List[int], node: int, dm: np.ndarray) -> Tuple[int, float]:
        """
        Find the position in a route that minimises insertion cost delta.

        Evaluates every inter-node gap (including before-first and after-last)
        and returns the position with the smallest distance increase.

        For an empty route, cost = depot→node + node→depot.

        Args:
            route: Current route node list.
            node: Candidate node to insert.
            dm: Distance matrix.

        Returns:
            Tuple of (best_position, delta_cost).
        """
        if not route:
            return 0, float(dm[0][node] + dm[node][0])

        best_pos = 0
        best_delta = float(dm[0][node] + dm[node][route[0]] - dm[0][route[0]])

        for i in range(len(route) - 1):
            delta = float(dm[route[i]][node] + dm[node][route[i + 1]] - dm[route[i]][route[i + 1]])
            if delta < best_delta:
                best_delta = delta
                best_pos = i + 1

        delta = float(dm[route[-1]][node] + dm[node][0] - dm[route[-1]][0])
        if delta < best_delta:
            best_delta = delta
            best_pos = len(route)

        return best_pos, best_delta

    @staticmethod
    def _min_distance_to_route(node: int, route: List[int], dm: np.ndarray) -> float:
        """Min distance from a candidate node to any node already in the route.

        Args:
            node: Candidate node.
            route: Current route.
            dm: Distance matrix.

        Returns:
            float: Minimum distance.
        """
        if not route:
            return float(dm[0][node])
        return float(min(dm[node][n] for n in route))
