"""RCSPP Pricing Subproblem Solver.

Solves the Resource-Constrained Shortest Path Problem (RCSPP) to identify
profitable columns (routes) for the BPC Master Problem.

Theoretical Deviation Note:
---------------------------
Unlike the pure ODIMCF formulation in Barnhart, Hane, and Vance (2000) which
manages state-space explosion strictly via branching, this implementation intentionally
incorporates Subset-Row Inequalities (SRIs) and accommodates Ryan-Foster branching.
These algorithmic choices fundamentally expand the DP state space by requiring auxiliary
resource dimensions (e.g., tracking SRI parity constraints). To mitigate this explosion
and maintain tractability, we rely heavily on the ng-route relaxation framework,
balancing bounding strength with computational viability for the VRPP.

Attributes:
    RCSPPSolver: Main solver class for the pricing subproblem.

Example:
    >>> solver = RCSPPSolver(n_nodes=10, cost_matrix=dist, wastes=w, capacity=100.0,
    ...                     revenue_per_kg=1.0, cost_per_km=0.1)
    >>> routes = solver.solve(duals={})
"""

from __future__ import annotations

import heapq
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union, cast

import numpy as np

from logic.src.policies.helpers.solvers_and_matheuristics.common.route import Route
from logic.src.policies.helpers.solvers_and_matheuristics.pricing.labels import Label

if TYPE_CHECKING:
    from logic.src.policies.helpers.solvers_and_matheuristics.branching.constraints import AnyBranchingConstraint

logger = logging.getLogger(__name__)

# Type alias for LCI cover items passed to the DP extension step.
# Each tuple: (cover_set, node_alpha_dict, dual_value, source_arc_or_none)
# source_arc_or_none is (i, j) for arc-saturation LCI, None for node/capacity LCI.
_LCICoverItem = Tuple[FrozenSet[int], Dict[int, float], float, Optional[Tuple[int, int]]]

# Multistar cut item: (cover_set, dual).  Capacity Q comes from self.capacity.
_MultistarItem = Tuple[FrozenSet[int], float]


class RCSPPSolver:
    """Exact / ng-relaxed solver for the Resource-Constrained Shortest Path Problem.

    Attributes:
        n_nodes (int): Number of customer nodes.
        cost_matrix (np.ndarray): Distance/cost matrix.
        wastes (Dict[int, float]): Node waste amounts.
        capacity (float): Vehicle capacity.
        R (float): Revenue per kg.
        C (float): Cost per km.
        ng_neighborhoods (Dict[int, Set[int]]): Memory structure for ng-routes.
        labels_generated (int): Statistics counter.

    Example:
        >>> solver = RCSPPSolver(10, dist, w, 100.0, 1.0, 0.1)
    """

    def __init__(
        self,
        n_nodes: int,
        cost_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        revenue_per_kg: float,
        cost_per_km: float,
        mandatory_nodes: Optional[Set[int]] = None,
        use_ng_routes: bool = True,
        ng_neighborhood_size: int = 8,
        ng_neighborhoods: Optional[Dict[int, Set[int]]] = None,
        node_prizes: Optional[Dict[int, float]] = None,
        timeout: float = 30.0,
        max_labels: int = 1000000,
    ) -> None:
        """Initialize the RCSPP solver for the pricing subproblem.

        Args:
            n_nodes (int): Number of customer nodes (excluding depot).
            cost_matrix (np.ndarray): Distance/cost matrix between all nodes.
            wastes (Dict[int, float]): Mapping of node IDs to waste amounts.
            capacity (float): Maximum vehicle capacity.
            revenue_per_kg (float): Revenue earned per unit of waste collected.
            cost_per_km (float): Cost incurred per unit of distance traveled.
            mandatory_nodes (Optional[Set[int]]): Nodes that MUST be visited if possible.
            use_ng_routes (bool): Whether to use ng-route relaxation.
            ng_neighborhood_size (int): Size of ng-neighborhood for each node.
            ng_neighborhoods (Optional[Dict[int, Set[int]]]): Precomputed ng-neighborhoods.
            node_prizes (Optional[Dict[int, float]]): Alternative profit per node (replaces waste * R).
            timeout (float): Maximum time allowed for solving in seconds.
            max_labels (int): Limit on generated labels to prevent explosion.
        """
        self.n_nodes = n_nodes
        self.cost_matrix = cost_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = revenue_per_kg
        self.C = cost_per_km
        self.node_prizes = node_prizes
        self.mandatory_nodes: Set[int] = mandatory_nodes or set()
        self.depot = 0
        self.use_ng_routes = use_ng_routes
        self.ng_neighborhood_size = ng_neighborhood_size
        self.timeout = timeout
        self.max_labels = max_labels

        self.labels_generated: int = 0
        self.labels_dominated: int = 0
        self.labels_infeasible: int = 0
        self.last_max_rc: float = -float("inf")
        self.dual_values: Dict[int, float] = {}
        self.bounds_to: np.ndarray = np.zeros(self.n_nodes + 1)
        self.bounds_from: np.ndarray = np.zeros(self.n_nodes + 1)
        self.fixed_arcs: Set[Tuple[int, int]] = set()

        if ng_neighborhoods is not None:
            self.ng_neighborhoods = ng_neighborhoods
        else:
            self.ng_neighborhoods = self._compute_ng_neighborhoods()

        # Dual for the vehicle-limit convexity constraint (set by solve() before use)
        self.vehicle_dual: float = 0.0

        # Precompute distance-sorted neighbor lists
        self._sorted_neighbors: Dict[int, List[int]] = self._precompute_sorted_neighbors()

    def _precompute_sorted_neighbors(self) -> Dict[int, List[int]]:
        """Precompute distance-sorted neighbor lists for all nodes.

        Returns:
            Dictionary mapping node ID to a list of other node IDs sorted by distance.
        """
        result: Dict[int, List[int]] = {}
        for i in range(self.n_nodes + 1):
            dists = [(self.cost_matrix[i, j], j) for j in range(self.n_nodes + 1) if j != i]
            dists.sort()
            result[i] = [j for _, j in dists]
        return result

    def _compute_ng_neighborhoods(self) -> Dict[int, Set[int]]:
        """Compute the initial ng-neighborhoods based on distance.

        Returns:
            Dictionary mapping node ID to its closest k-1 neighbors.
        """
        customer_nodes = list(range(1, self.n_nodes + 1))
        k = min(self.ng_neighborhood_size, len(customer_nodes))

        neighborhoods: Dict[int, Set[int]] = {}
        for i in customer_nodes:
            distances = sorted((self.cost_matrix[i, j], j) for j in customer_nodes if j != i)
            closest: Set[int] = {j for _, j in distances[: k - 1]}
            closest.add(i)
            neighborhoods[i] = closest
        neighborhoods[self.depot] = set()
        return neighborhoods

    def save_ng_snapshot(self) -> Dict[int, Set[int]]:
        """Return a deep copy of the current ng-neighborhoods.

        Returns:
            A mapping of node IDs to their current ng-neighbor sets.
        """

        return {k: set(v) for k, v in self.ng_neighborhoods.items()}

    def restore_ng_snapshot(self, snapshot: Dict[int, Set[int]]) -> None:
        """Restore ng-neighborhoods from a previously saved snapshot.

        Args:
            snapshot: A previously saved mapping of node IDs to neighbor sets.
        """

        self.ng_neighborhoods = {k: set(v) for k, v in snapshot.items()}

    def enforce_elementarity(self, nodes: List[int]) -> int:
        """Dynamically expand ng-sets to enforce strict elementarity on fractional paths.

        Args:
            nodes: List of node IDs whose occurrence in ng-sets should be enforced.

        Returns:
            The total number of new neighbor relations added across all nodes.
        """

        added_count = 0
        nodes_to_enforce = [n for n in nodes if n != self.depot]
        if not nodes_to_enforce:
            return 0
        for i in range(self.n_nodes + 1):
            if i == self.depot:
                continue
            before = len(self.ng_neighborhoods[i])
            self.ng_neighborhoods[i].update(nodes_to_enforce)
            added_count += len(self.ng_neighborhoods[i]) - before
        return added_count

    def expand_ng_neighborhoods(self, cycles: List[Tuple[int, ...]]) -> int:
        """Expands ng-neighborhoods to break detected cycles.

        Args:
            cycles (List[Tuple[int, ...]]): List of node cycles detected in fractional solutions.

        Returns:
            int: Total number of neighbor relations added.
        """
        added = 0
        for cycle in cycles:
            cset = set(cycle)
            for i in cycle:
                if i == self.depot:
                    continue
                before = len(self.ng_neighborhoods[i])
                self.ng_neighborhoods[i].update(cset)
                added += len(self.ng_neighborhoods[i]) - before
        return added

    def solve(
        self,
        dual_values: Union[Dict[int, float], Dict[str, Any]],
        max_routes: int = 10,
        branching_constraints: Optional[List["AnyBranchingConstraint"]] = None,
        capacity_cut_duals: Optional[Dict[FrozenSet[int], float]] = None,
        sri_cut_duals: Optional[Dict[FrozenSet[int], float]] = None,
        edge_clique_cut_duals: Optional[Dict[Tuple[int, int], float]] = None,
        forced_nodes: Optional[Set[int]] = None,
        rf_conflicts: Optional[Dict[int, Set[int]]] = None,
        is_farkas: bool = False,
        exact_mode: bool = False,
        timeout: Optional[float] = None,
    ) -> List[Route]:
        """Solve the RCSPP subproblem to find columns with positive reduced cost.

        Args:
            dual_values (Union[Dict[int, float], Dict[str, Any]]): Master problem duals.
            max_routes (int): Maximum number of profitable routes to return.
            branching_constraints (Optional[List[AnyBranchingConstraint]]): Active branching rules.
            capacity_cut_duals (Optional[Dict[FrozenSet[int], float]]): Duals for capacity cuts.
            sri_cut_duals (Optional[Dict[FrozenSet[int], float]]): Duals for SRI cuts.
            edge_clique_cut_duals (Optional[Dict[Tuple[int, int], float]]): Duals for edge clique cuts.
            forced_nodes (Optional[Set[int]]): Nodes required by branching.
            rf_conflicts (Optional[Dict[int, Set[int]]]): Ryan-Foster branching conflicts.
            is_farkas (bool): If True, solve Farkas pricing for feasibility.
            exact_mode (bool): If True, disables neighborhood pruning.
            timeout (Optional[float]): Optional override for solver timeout.

        Returns:
            List[Route]: List of identified profitable routes.
        """
        # Pre-execution setup
        self._rcc_duals_for_bounds = (
            cast(Dict[str, Any], dual_values).get("rcc_duals", {})
            if isinstance(dual_values, dict)
            else capacity_cut_duals or {}
        )

        # 1. Dual handling
        if isinstance(dual_values, dict) and "node_duals" in dual_values:
            complex_duals = cast(Dict[str, Any], dual_values)
            node_duals = complex_duals.get("node_duals", {})
            rcc_duals = complex_duals.get("rcc_duals", {})
            sri_duals = complex_duals.get("sri_duals", {})
            edge_clique_duals = complex_duals.get("edge_clique_duals", {})
            self.vehicle_dual = complex_duals.get("vehicle_limit", 0.0)
            lci_duals_raw = complex_duals.get("lci_duals", {})
            lci_node_alphas_raw = complex_duals.get("lci_node_alphas", {})
            lci_arcs_raw = complex_duals.get("lci_arcs", {})
            multistar_duals_raw = complex_duals.get("multistar_duals", {})
        else:
            node_duals = dual_values  # type: ignore
            rcc_duals = capacity_cut_duals or {}
            sri_duals = sri_cut_duals or {}
            edge_clique_duals = edge_clique_cut_duals or {}
            self.vehicle_dual = 0.0
            lci_duals_raw = {}
            lci_node_alphas_raw = {}
            lci_arcs_raw = {}
            multistar_duals_raw = {}

        # Build LCI cover items for DP extension: list of (cover_set, node_alpha, dual, arc_or_none).
        # Only include cuts with non-negligible dual to avoid wasted iteration in the inner loop.
        lci_cover_items: List[_LCICoverItem] = [
            (cover_set, lci_node_alphas_raw.get(cover_set, {}), dual, lci_arcs_raw.get(cover_set))
            for cover_set, dual in lci_duals_raw.items()
            if dual > 1e-8
        ]

        # Build multistar items: only active duals to minimise inner-loop overhead.
        multistar_items: List[_MultistarItem] = [
            (cover_set, dual) for cover_set, dual in multistar_duals_raw.items() if dual > 1e-8
        ]

        # 2. Reset state
        self.labels_generated = 0
        self.labels_dominated = 0
        self.labels_infeasible = 0
        self.last_max_rc = -float("inf")
        self.dual_values = node_duals
        self.is_farkas = is_farkas
        self.forced_nodes = forced_nodes or set()
        self.rf_conflicts = rf_conflicts or {}

        # Update timeout if provided by the caller (Branch-and-Price engine)
        active_timeout = min(
            timeout if timeout is not None and timeout > 0 else self.timeout,
            self.timeout,
        )
        self._current_timeout = active_timeout

        # Task 2: Completion Bounds
        self._compute_completion_bounds()

        # 3. SRI pre-processing
        active_sri_items = [(k, v) for k, v in sri_duals.items() if v > 1e-4]
        active_sri_items.sort(key=lambda x: x[1], reverse=True)
        active_sri_subsets = sorted([k for k, v in active_sri_items], key=hash)
        sri_dual_values = [sri_duals[s] for s in active_sri_subsets]
        self.sri_dual_values = sri_dual_values
        node_to_sri: Dict[int, List[int]] = {i: [] for i in range(self.n_nodes + 1)}
        for idx, s in enumerate(active_sri_subsets):
            for node in s:
                if node in node_to_sri:
                    node_to_sri[node].append(idx)

        # 4. Precompute reduced costs
        if not self.is_farkas:
            self.node_reduced_costs = {}
            for n in range(1, self.n_nodes + 1):
                rev = self.node_prizes.get(n, 0.0) if self.node_prizes is not None else self.wastes.get(n, 0.0) * self.R
                self.node_reduced_costs[n] = rev - node_duals.get(n, 0.0)
        else:
            self.node_reduced_costs = {n: node_duals.get(n, 0.0) for n in range(1, self.n_nodes + 1)}

        # 5. Constraints
        constraints: List[Any] = branching_constraints or []
        (forbidden_arcs, req_succ, req_pred, rf_sep, rf_tog) = self._preprocess_constraints(constraints)

        original_use_ng = self.use_ng_routes
        try:
            # For large instances (>40 nodes), run heuristic pricing first.
            # It's fast (~0.1s) and finds good columns. If it finds enough
            # positive-RC columns, skip the expensive exact DP entirely.
            # Only fall back to exact DP when heuristic is insufficient.
            heuristic_routes: List[Route] = []
            if self.n_nodes > 40:
                heuristic_routes = self._heuristic_pricing(
                    max_routes=max_routes,
                    node_duals=self.dual_values,
                    rcc_duals=rcc_duals,
                    forbidden_arcs=forbidden_arcs,
                )

            if len(heuristic_routes) >= max_routes // 2:
                # Enough good columns found heuristically — skip exact DP
                routes = heuristic_routes
            else:
                routes = self._label_correcting_algorithm(
                    max_routes=max_routes,
                    forbidden_arcs=forbidden_arcs,
                    required_successors=req_succ,
                    required_predecessors=req_pred,
                    rf_separate=rf_sep,
                    rf_together=rf_tog,
                    rcc_duals=rcc_duals,
                    active_sri_subsets=active_sri_subsets,
                    sri_dual_values=sri_dual_values,
                    node_to_sri=node_to_sri,
                    edge_clique_duals=edge_clique_duals,
                    lci_cover_items=lci_cover_items,
                    multistar_items=multistar_items,
                    exact_mode=exact_mode,
                )
                # Merge: keep best unique routes from both
                seen: set = {frozenset(r.nodes) for r in routes}
                for r in heuristic_routes:
                    if frozenset(r.nodes) not in seen:
                        routes.append(r)
                        seen.add(frozenset(r.nodes))
        finally:
            self.use_ng_routes = original_use_ng

        routes.sort(key=lambda x: x.reduced_cost, reverse=True)  # type: ignore[arg-type,return-value]
        return routes[:max_routes]

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

    def _label_correcting_algorithm(  # noqa: C901
        self,
        max_routes: int,
        forbidden_arcs: FrozenSet[Tuple[int, int]],
        required_successors: Dict[int, int],
        required_predecessors: Dict[int, int],
        rf_separate: Set[Tuple[int, int]],
        rf_together: Set[Tuple[int, int]],
        rcc_duals: Dict[FrozenSet[int], float],
        active_sri_subsets: List[FrozenSet[int]],
        sri_dual_values: List[float],
        node_to_sri: Dict[int, List[int]],
        edge_clique_duals: Dict[Tuple[int, int], float],
        lci_cover_items: List[_LCICoverItem],
        multistar_items: Optional[List[_MultistarItem]] = None,
        exact_mode: bool = False,
    ) -> List[Route]:
        """Forward label-correcting algorithm using a priority queue.

        Args:
            max_routes: Maximum number of routes to return.
            forbidden_arcs: Arcs forbidden by branching.
            required_successors: Successors required by branching.
            required_predecessors: Predecessors required by branching.
            rf_separate: Ryan-Foster separate conflicts.
            rf_together: Ryan-Foster together requirements.
            rcc_duals: Duals for capacity cuts.
            active_sri_subsets: List of active SRI subsets.
            sri_dual_values: Duals for active SRIs.
            node_to_sri: Mapping from node to SRI indices it belongs to.
            edge_clique_duals: Duals for edge clique cuts.
            lci_cover_items: Items for LCI dual penalties.
            multistar_items: Items for Multistar dual penalties (cover_set, dual).
            exact_mode: If True, uses all neighbors.

        Returns:
            List of profitable routes found.
        """
        _multistar_items: List[_MultistarItem] = multistar_items or []

        use_ng = self.use_ng_routes
        initial_sri = tuple([0] * len(active_sri_subsets))

        start = Label(reduced_cost=0.0, node=self.depot, load=0.0, sri_state=initial_sri)

        # Max-heap via negated reduced_cost; integer counter breaks ties.
        queue: List[Tuple[float, int, Label]] = [(-start.reduced_cost, 0, start)]
        _counter = 0

        labels_at_node: Dict[int, List[Label]] = {self.depot: [start]}
        completed_routes: List[Label] = []
        global_max_rc = -float("inf")

        # Reset timeout flag from previous run
        self._timed_out = False

        # Adaptive label cap: scale with instance size.
        # Python processes ~10k labels/sec; cap early so partial results
        # are returned within the timeout budget.
        # n=27: cap = min(max_labels, 27^1.5*200) ~ 28k -> ~3s
        # n=14: cap = min(max_labels, 14^1.5*200) ~ 10k -> ~1s
        adaptive_max_labels = min(self.max_labels, int(self.n_nodes**1.5 * 200))

        start_time = time.perf_counter()
        while queue:
            if self.labels_generated > adaptive_max_labels:
                logger.info(
                    f"RCSPP label cap: {self.labels_generated} labels "
                    f"(cap={adaptive_max_labels}, n_nodes={self.n_nodes})."
                )
                self._timed_out = True
                break

            if self._current_timeout > 0 and (time.perf_counter() - start_time) > self._current_timeout:
                logger.warning(
                    f"RCSPP timeout after {self._current_timeout}s: "
                    f"n_nodes={self.n_nodes}, labels={self.labels_generated}."
                )
                self._timed_out = True
                break

            _, _, current = heapq.heappop(queue)
            u = current.node

            # Candidate successors
            neighbor_limit = self.n_nodes if exact_mode else min(20, max(5, self.n_nodes // 3))
            candidates = (
                [required_successors[u]] if u in required_successors else self._get_neighbors(u, neighbor_limit)
            )

            for v in candidates:
                if v == self.depot:
                    continue

                # Fix 2: Early exit for fixed arcs before expensive feasibility checks
                if (u, v) in self.fixed_arcs:
                    continue

                # Elementarity
                if v in current.visited:
                    continue

                # ng-feasibility
                if (use_ng and v in current.ng_memory) and u not in required_successors:
                    continue

                # Ryan-Foster separation
                if any(tuple(sorted((v, n))) in rf_separate for n in current.visited):
                    continue

                # Edge constraints
                if (u, v) in forbidden_arcs:
                    continue
                if v in required_predecessors and required_predecessors[v] != u:
                    continue

                new_label = self._extend_label(
                    current,
                    v,
                    forbidden_arcs,
                    rcc_duals,
                    active_sri_subsets,
                    sri_dual_values,
                    node_to_sri,
                    edge_clique_duals,
                    lci_cover_items,
                    _multistar_items,
                )
                if new_label is None:
                    continue

                self.labels_generated += 1
                existing = labels_at_node.get(v, [])
                if any(e.dominates(new_label, use_ng=use_ng) for e in existing):
                    self.labels_dominated += 1
                    continue

                labels_at_node[v] = [lbl for lbl in existing if not new_label.dominates(lbl, use_ng=use_ng)]
                labels_at_node[v].append(new_label)
                _counter += 1
                heapq.heappush(queue, (-new_label.reduced_cost, _counter, new_label))
                global_max_rc = max(global_max_rc, new_label.reduced_cost)

            # Attempt depot return
            if u != self.depot:
                can_return = not (u in required_successors and required_successors[u] != self.depot)

                # Completion bounds pruning.
                # Only skip the depot-return attempt when the BEST possible final
                # reduced cost from this label is provably <= 0 — i.e., even the
                # optimal completion cannot produce a column worth adding.
                if current.reduced_cost + self.bounds_to[u] > 0.0 and can_return:
                    final = self._extend_to_depot(
                        current,
                        rcc_duals,
                        edge_clique_duals,
                        lci_cover_items,
                        _multistar_items,
                    )
                    if final:
                        completed_routes.append(final)
                        global_max_rc = max(global_max_rc, final.reduced_cost)

        self.last_max_rc = global_max_rc
        routes: List[Route] = []
        seen_node_sets: set = set()
        for label in sorted(completed_routes, key=lambda x: x.reduced_cost, reverse=True):
            if label.reduced_cost > 1e-6:
                path = label.reconstruct_path()
                nodes = [n for n in path if n != self.depot]
                node_key = frozenset(nodes)
                # Skip duplicate node sets — different orderings of same nodes
                # add no LP information
                if node_key in seen_node_sets:
                    continue
                seen_node_sets.add(node_key)
                rt = self._compute_route_details(nodes)
                rt.reduced_cost = label.reduced_cost
                routes.append(rt)
                if len(routes) >= max_routes:
                    break
        return routes

    def _get_neighbors(self, node: int, limit: int) -> List[int]:
        """Return the nearest neighbors for a given node.

        Used to artificially bound the RCSPP transition state-space during
        heuristic pricing passes by limiting expansions to the closest
        candidates (e.g., limit=20).

        Args:
            node: Source node ID.
            limit: Maximum number of neighbors to return.

        Returns:
            List of nearest neighbor node IDs.
        """

        return self._sorted_neighbors[node][:limit]

    def _extend_to_depot(
        self,
        label: Label,
        rcc_duals: Dict[FrozenSet[int], float],
        edge_clique_duals: Dict[Tuple[int, int], float],
        lci_cover_items: List[_LCICoverItem],
        multistar_items: List[_MultistarItem],
    ) -> Optional[Label]:
        """Extends a label to the depot, applying ALL final dual penalties.

        Args:
            label: Current label at a customer node.
            rcc_duals: Duals for capacity cuts.
            edge_clique_duals: Duals for edge clique cuts.
            lci_cover_items: Items for LCI dual penalties.
            multistar_items: Items for Multistar dual penalties (cover_set, dual).

        Returns:
            Completed label at the depot, or None if infeasible.
        """
        current_node = label.node
        next_node = self.depot

        # Base transition costs
        # Farkas pricing measures feasibility only: travel costs are 0 on every arc,
        # including the return to the depot (as in _extend_label).
        cost = 0.0 if self.is_farkas else self.cost_matrix[current_node, next_node] * self.C
        rc_delta = -cost

        # 1. RCC Duals (Boundary check)
        for subset, dual in rcc_duals.items():
            if (current_node in subset) != (next_node in subset):
                rc_delta += dual if self.is_farkas else -dual

        # 2. Edge Clique Cuts
        for (u, v), dual in edge_clique_duals.items():
            if (current_node == u and next_node == v) or (current_node == v and next_node == u):
                rc_delta += dual if self.is_farkas else -dual

        # 3. SRI Cuts (Depot return does not trigger new visits, but we check state)
        # Standard formulation ignores depot for SRI parity, so no penalty added here.

        # 4. Multistar Cuts
        for cover_set, dual in multistar_items:
            if current_node in cover_set and next_node not in cover_set:
                # Crossing out: cross=1, adj_demand=0.0 (since v is depot)
                arc_a_k = 1.0
                rc_delta += (arc_a_k * dual) if self.is_farkas else -(arc_a_k * dual)

        # 5. Saturated Arc LCIs
        for cover_set, node_alpha, dual, lci_arc in lci_cover_items:
            if lci_arc:
                if (current_node, next_node) == lci_arc:
                    rc_delta += dual if self.is_farkas else -dual
            elif next_node in cover_set:
                # next_node is depot (0). Most LCIs don't include depot.
                rc_delta += node_alpha.get(next_node, 1.0) * (dual if self.is_farkas else -dual)

        # Calculate final reduced cost
        new_rc = label.reduced_cost + rc_delta

        # Final feasibility check
        if new_rc < -1e-6:
            return None

        return Label(
            node=next_node,
            load=label.load,
            reduced_cost=new_rc,
            visited=label.visited,
            ng_memory=label.ng_memory,
            rf_unmatched=label.rf_unmatched,
            parent=label,
            sri_state=label.sri_state,
        )

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

    def _extend_label(  # noqa: C901
        self,
        label: Label,
        next_node: int,
        forbidden: FrozenSet[Tuple[int, int]],
        rcc_duals: Dict[FrozenSet[int], float],
        active_sri: List[FrozenSet[int]],
        sri_duals: List[float],
        node_to_sri: Dict[int, List[int]],
        edge_clique_duals: Dict[Tuple[int, int], float],
        lci_cover_items: List[_LCICoverItem],
        multistar_items: Optional[List[_MultistarItem]] = None,
    ) -> Optional[Label]:
        """Extend a label along an arc to a new node.

        Args:
            label: Starting label.
            next_node: Target node to extend to.
            forbidden: Forbidden arcs by branching.
            rcc_duals: Duals for capacity cuts.
            active_sri: Active SRI subsets.
            sri_duals: Duals for SRIs.
            node_to_sri: Node-to-SRI index mapping.
            edge_clique_duals: Duals for edge clique cuts.
            lci_cover_items: Items for LCI dual penalties.
            multistar_items: Items for Multistar dual penalties.

        Returns:
            New label at next_node, or None if infeasible/pruned.
        """

        if (label.node, next_node) in forbidden:
            return None

        load = label.load + self.wastes.get(next_node, 0.0)
        if load > self.capacity + 1e-6:
            return None

        dist = self.cost_matrix[label.node, next_node]
        cost = (dist * self.C) if not self.is_farkas else 0.0

        if self.node_prizes is not None:
            rev = self.node_prizes.get(next_node, 0.0) if not self.is_farkas else 0.0
        else:
            rev = (self.wastes.get(next_node, 0.0) * self.R) if not self.is_farkas else 0.0

        # Node visitation duals act as rewards in both phases to drive coverage
        rc_delta = (
            (rev - cost - self.dual_values.get(next_node, 0.0))
            if not self.is_farkas
            else self.dual_values.get(next_node, 0.0)
        )

        # 1. RCC duals: penalise each boundary CROSSING
        for subset, dual in rcc_duals.items():
            if (label.node in subset) != (next_node in subset):
                rc_delta -= dual

        # 2. Edge-Clique dual penalty
        if edge_clique_duals:
            can_edge = tuple(sorted((label.node, next_node)))
            rc_delta -= edge_clique_duals.get(can_edge, 0.0)  # type: ignore[arg-type]

        # 3. SRI Dual Penalty
        new_sri = list(label.sri_state)
        for idx in node_to_sri.get(next_node, []):
            if new_sri[idx] == 0:
                new_sri[idx] = 1
            elif new_sri[idx] == 1:
                new_sri[idx] = 2
                rc_delta -= sri_duals[idx]

        # 4. LCI Dual Penalty
        for cover_set, node_alpha, dual, lci_arc in lci_cover_items:
            if lci_arc:
                if (label.node, next_node) == lci_arc:
                    rc_delta -= dual
            elif next_node in cover_set:
                rc_delta -= node_alpha.get(next_node, 1.0) * dual

        # 5. Multistar Dual Penalty
        if multistar_items:
            u, v = label.node, next_node
            Q = self.capacity
            demand_v = self.wastes.get(v, 0.0)
            demand_u = self.wastes.get(u, 0.0)
            for cover_set, dual in multistar_items:
                u_in = u in cover_set
                v_in = v in cover_set
                if u_in and not v_in:
                    adj_v = demand_v if v != self.depot else 0.0
                    arc_a_k = 1.0 - (2.0 / Q) * adj_v
                elif not u_in and v_in:
                    adj_u = demand_u if u != self.depot else 0.0
                    arc_a_k = 1.0 - (2.0 / Q) * adj_u - (2.0 / Q) * demand_v
                elif u_in and v_in:
                    arc_a_k = -(2.0 / Q) * demand_v
                else:
                    continue
                rc_delta -= arc_a_k * dual

        return Label(
            node=next_node,
            load=load,
            reduced_cost=label.reduced_cost + rc_delta,
            visited=label.visited | {next_node},
            parent=label,
            ng_memory=(label.ng_memory & self.ng_neighborhoods[next_node]) | {next_node},
            sri_state=tuple(new_sri),
            rf_unmatched=label.rf_unmatched,
        )
