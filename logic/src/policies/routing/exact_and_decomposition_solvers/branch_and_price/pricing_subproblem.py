"""Pricing Subproblem for Branch-and-Price VRPP.

Solves the resource-constrained shortest path problem (RCSPP) heuristically
to generate new routes with positive reduced cost.

Based on Section 2.2 and Section 4 of Barnhart et al. (1998).

Branching constraint enforcement
---------------------------------
Active branching constraints are now passed into every route-generation call
and are enforced eagerly during the greedy insertion loop — not post-hoc.
This prevents the heuristic from producing columns that violate the current
B&B node's branching decisions, which would pollute the master LP and cause
the LP bound to be over-estimated.

Both constraint flavours defined in ``branching.py`` are handled:

EdgeBranchingConstraint (must_use=False)
    Forbids arc (u → v).  Before inserting node v after u, check forbidden
    arcs.

EdgeBranchingConstraint (must_use=True)
    Requires arc (u → w).  If the current tail of the partial route is u and
    the candidate node is v ≠ w, reject v (u must be followed by w next).
    Symmetrically, if v has a required predecessor x and the current tail is
    not x, reject v.

RyanFosterBranchingConstraint (together=False)
    Forbids r and s from appearing in the same route.  If v is already in a
    separate constraint with a node already in the partial route, reject v.

RyanFosterBranchingConstraint (together=True)
    Best-effort at heuristic level: enforced lazily at route completion by
    discarding any finished route that contains exactly one node of a
    together-pair (i.e. partial together enforcement).  Full enforcement
    would require a DP.

Attributes:
    PricingSubproblem (class): Heuristic RCSPP solver using greedy insertion.

Example:
    >>> pricer = PricingSubproblem(n_nodes=50, cost_matrix=dist, ...)
    >>> routes = pricer.solve(dual_values)

Parts are relocated to pricing_greedy.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .pricing_greedy import PricingGreedyMixin


class PricingSubproblem(PricingGreedyMixin):
    """Resource-Constrained Shortest Path Problem for Route Generation (Heuristic).

    Uses a greedy insertion heuristic to quickly generate routes with positive
    reduced cost.

    Attributes:
        n_nodes (int): Number of customer nodes.
        cost_matrix (np.ndarray): Symmetric distance matrix.
        wastes (Dict[int, float]): Map from node to waste volume.
        capacity (float): Vehicle capacity.
        R (float): Revenue per kg.
        C (float): Cost per km.
        mandatory_nodes (Set[int]): Nodes that must be visited.
        depot (int): Index of the depot node (always 0).
    """

    IS_EXACT = False

    def __init__(
        self,
        n_nodes: int,
        cost_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        revenue_per_kg: float,
        cost_per_km: float,
        mandatory_nodes: Set[int],
    ) -> None:
        """Initialise the pricing subproblem.

        Args:
            n_nodes: Number of customer nodes (excluding depot, depot = 0).
            cost_matrix: Distance matrix of shape (n_nodes+1, n_nodes+1).
            wastes: Mapping from node ID to waste volume.
            capacity: Vehicle payload capacity.
            revenue_per_kg: Revenue earned per unit of waste collected.
            cost_per_km: Cost per unit of distance travelled.
            mandatory_nodes: Set of node indices that must be visited.
        """
        self.n_nodes = n_nodes
        self.cost_matrix = cost_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = revenue_per_kg
        self.C = cost_per_km
        self.mandatory_nodes = mandatory_nodes
        self.depot = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(
        self,
        dual_values: Dict[Any, Any],
        max_routes: int = 10,
        active_constraints: Optional[List[Any]] = None,
        capacity_cut_duals: Optional[Dict[Any, float]] = None,
        assert_not_exact: bool = False,
        **kwargs: Any,
    ) -> List[Tuple[List[int], float]]:
        """Generate routes with positive reduced cost.

        **Warning:** This pricer is heuristic. It may miss columns with positive reduced
        cost.

        Args:
            dual_values: Dual values from the master problem.
            max_routes: Maximum number of routes to return.
            active_constraints: Active branching constraints at the current B&B node.
            capacity_cut_duals: Optional mapping of node-sets to dual values.
            assert_not_exact: If True, raises RuntimeError to prevent misuse.
            kwargs: Additional parameters (e.g. node_prizes).

        Returns:
            List of ``(route_nodes, reduced_cost)`` tuples.
        """
        if assert_not_exact:
            raise RuntimeError(
                "PricingSubproblem is a heuristic and cannot be used to certify LP "
                "optimality at a B&B node. Use RCSPPSolver instead."
            )

        # Support both flat node duals (legacy) and structured dual dicts (BPC/standardized)
        node_duals: Dict[Any, float]
        if "node_duals" in dual_values:
            node_duals = dual_values["node_duals"]
            if capacity_cut_duals is None:
                capacity_cut_duals = dual_values.get("rcc_duals")
        else:
            node_duals = dual_values  # type: ignore[assignment]

        constraints: List[Any] = active_constraints or []

        # Pre-process constraints into O(1) look-up structures.
        (
            forbidden_arcs,
            req_successors,
            req_predecessors,
            rf_separate_pairs,
            rf_together_pairs,
        ) = self._preprocess_constraints(constraints)

        routes: List[Tuple[List[int], float]] = []

        # Rank start nodes by their standalone attractiveness.
        candidate_starts = sorted(
            range(1, self.n_nodes + 1),
            key=lambda n: node_duals.get(n, 0.0) + self.wastes.get(n, 0.0) * self.R,
            reverse=True,
        )

        for start_node in candidate_starts[:max_routes]:
            route, rc = self._greedy_routing(
                start_node=start_node,
                dual_values=node_duals,
                forbidden_arcs=forbidden_arcs,
                req_successors=req_successors,
                req_predecessors=req_predecessors,
                rf_separate_pairs=rf_separate_pairs,
                rf_together_pairs=rf_together_pairs,
                capacity_cut_duals=capacity_cut_duals,
                node_prizes=kwargs.get("node_prizes"),
            )

            if rc > 1e-4:
                routes.append((route, rc))

            if len(routes) >= max_routes:
                break

        routes.sort(key=lambda x: x[1], reverse=True)
        return routes[:max_routes]

    # ------------------------------------------------------------------
    # Constraint pre-processing
    # ------------------------------------------------------------------

    def _preprocess_constraints(
        self,
        constraints: List[Any],
    ) -> Tuple[Set[Tuple[int, int]], Dict[int, int], Dict[int, int], Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """Convert active constraints into fast look-up structures.

        Separates EdgeBranchingConstraint objects from RyanFosterBranchingConstraint
        objects without importing the classes (uses duck-typing to stay
        decoupled from the branching module at import time).

        Args:
            constraints: Mixed list of active branching constraints.

        Returns:
            Five look-up structures used by the greedy heuristic.
        """
        forbidden_arcs: Set[Tuple[int, int]] = set()
        req_successors: Dict[int, int] = {}
        req_predecessors: Dict[int, int] = {}
        rf_separate: Set[Tuple[int, int]] = set()
        rf_together: Set[Tuple[int, int]] = set()

        for c in constraints:
            # Duck-typed dispatch — avoids a hard import dependency on branching.py.
            if hasattr(c, "must_use"):
                # EdgeBranchingConstraint
                if not c.must_use:
                    forbidden_arcs.add((c.u, c.v))
                else:
                    req_successors[c.u] = c.v
                    req_predecessors[c.v] = c.u
            elif hasattr(c, "together"):
                # RyanFosterBranchingConstraint
                key = (min(c.node_r, c.node_s), max(c.node_r, c.node_s))
                if not c.together:
                    rf_separate.add(key)
                else:
                    rf_together.add(key)

        return forbidden_arcs, req_successors, req_predecessors, rf_separate, rf_together

    # ------------------------------------------------------------------
    # Greedy routing
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # Ryan-Foster separation helper
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Reduced cost and route details
    # ------------------------------------------------------------------

    def _compute_reduced_cost(
        self,
        route: List[int],
        dual_values: Dict[int, float],
        capacity_cut_duals: Optional[Dict[Any, float]] = None,
        node_prizes: Optional[Dict[int, float]] = None,
    ) -> float:
        """Compute the reduced cost of a completed route.

        reduced_cost = profit − Σ_i dual_i − dual_vehicle_limit − crossing_penalty
        profit       = (if node_prizes given) Σ_i node_prizes_i else Σ_i waste_i × R
        cost         = total_distance × C

        Args:
            route: Ordered list of customer nodes (depot excluded).
            dual_values: Node-coverage duals and optional vehicle-limit dual.
            capacity_cut_duals: Optional duals for active capacity cuts.
            node_prizes: Optional fixed prizes per node.

        Returns:
            Scalar reduced cost.
        """
        total_distance = 0.0
        prev = self.depot
        for node in route:
            total_distance += self.cost_matrix[prev, node]
            prev = node
        total_distance += self.cost_matrix[prev, self.depot]

        if node_prizes is not None:
            revenue = sum(node_prizes.get(n, 0.0) for n in route)
        else:
            total_waste = sum(self.wastes.get(n, 0.0) for n in route)
            revenue = total_waste * self.R

        cost = total_distance * self.C
        profit = revenue - cost

        dual_contribution = sum(dual_values.get(n, 0.0) for n in route)
        vehicle_dual = dual_values.get("vehicle_limit", 0.0)  # type: ignore[call-overload]

        crossing_penalty = 0.0
        if capacity_cut_duals:
            full_path = [self.depot] + route + [self.depot]
            for i in range(len(full_path) - 1):
                u, v = full_path[i], full_path[i + 1]
                for node_set, dual in capacity_cut_duals.items():
                    if (u in node_set) != (v in node_set):
                        crossing_penalty += dual

        return profit - dual_contribution - vehicle_dual - crossing_penalty

    def compute_route_details(
        self,
        route: List[int],
    ) -> Tuple[float, float, float, Set[int]]:
        """Compute cost, revenue, load, and node coverage for a given route.

        Args:
            route: Ordered list of customer nodes (depot excluded).

        Returns:
            A 4-tuple of (cost, revenue, load, node_coverage).
        """
        total_distance = 0.0
        prev = self.depot
        for node in route:
            total_distance += self.cost_matrix[prev, node]
            prev = node
        total_distance += self.cost_matrix[prev, self.depot]

        total_waste = sum(self.wastes.get(n, 0.0) for n in route)
        revenue = total_waste * self.R
        cost = total_distance * self.C

        return cost, revenue, total_waste, set(route)
