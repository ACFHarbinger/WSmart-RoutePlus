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

Parts are relocated to solver_support.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union, cast

import numpy as np

from logic.src.policies.helpers.solvers_and_matheuristics.common.route import Route
from logic.src.policies.helpers.solvers_and_matheuristics.pricing.labeling import RCSPPLabelingMixin
from logic.src.policies.helpers.solvers_and_matheuristics.pricing.pricing_types import (
    PricingStatus,
    _LCICoverItem,
    _MultistarItem,
)

from .solver_support import RCSPPSolverSupportMixin

if TYPE_CHECKING:
    from logic.src.policies.helpers.solvers_and_matheuristics.branching.constraints import AnyBranchingConstraint

logger = logging.getLogger(__name__)


class RCSPPSolver(RCSPPSolverSupportMixin, RCSPPLabelingMixin):
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
            n_nodes: Number of customer nodes (excluding depot).
            cost_matrix: Distance/cost matrix between all nodes.
            wastes: Mapping of node IDs to waste amounts.
            capacity: Maximum vehicle capacity.
            revenue_per_kg: Revenue earned per unit of waste collected.
            cost_per_km: Cost incurred per unit of distance traveled.
            mandatory_nodes: Nodes that MUST be visited if possible.
            use_ng_routes: Whether to use ng-route relaxation.
            ng_neighborhood_size: Size of ng-neighborhood for each node.
            ng_neighborhoods: Precomputed ng-neighborhoods.
            node_prizes: Alternative profit per node (replaces waste * R).
            timeout: Maximum time allowed for solving in seconds.
            max_labels: Limit on generated labels to prevent explosion.
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
        self.pricing_status: PricingStatus = PricingStatus.PARTIAL

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
            cycles: List of node cycles detected in fractional solutions.

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
            dual_values: Master problem duals.
            max_routes: Maximum number of profitable routes to return.
            branching_constraints: Active branching rules.
            capacity_cut_duals: Duals for capacity cuts.
            sri_cut_duals: Duals for SRI cuts.
            edge_clique_cut_duals: Duals for edge clique cuts.
            forced_nodes: Nodes required by branching.
            rf_conflicts: Ryan-Foster branching conflicts.
            is_farkas: If True, solve Farkas pricing for feasibility.
            exact_mode: If True, disables neighborhood pruning.
            timeout: Optional override for solver timeout.

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
            v_dual = complex_duals.get("vehicle_limit", 0.0) or complex_duals.get("fleet_dual", 0.0)
            if not v_dual and isinstance(node_duals, dict):
                v_dual = node_duals.get("vehicle_limit", 0.0)
            self.vehicle_dual = float(v_dual or 0.0)
            lci_duals_raw = complex_duals.get("lci_duals", {})
            lci_node_alphas_raw = complex_duals.get("lci_node_alphas", {})
            lci_arcs_raw = complex_duals.get("lci_arcs", {})
            multistar_duals_raw = complex_duals.get("multistar_duals", {})
        else:
            node_duals = dual_values  # type: ignore
            rcc_duals = capacity_cut_duals or {}
            sri_duals = sri_cut_duals or {}
            edge_clique_duals = edge_clique_cut_duals or {}
            v_dual = (
                cast(Dict[Any, Any], node_duals).get("vehicle_limit", 0.0) if isinstance(node_duals, dict) else 0.0
            )
            self.vehicle_dual = float(v_dual or 0.0)
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
        self.dual_values = cast(Dict[int, float], node_duals)
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
                self.node_reduced_costs[n] = rev - cast(Dict[Any, Any], node_duals).get(n, 0.0)
        else:
            self.node_reduced_costs = {
                n: cast(Dict[Any, Any], node_duals).get(n, 0.0) for n in range(1, self.n_nodes + 1)
            }

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
            if not exact_mode and self.n_nodes > 40:
                heuristic_routes = self._heuristic_pricing(
                    max_routes=max_routes,
                    node_duals=self.dual_values,
                    rcc_duals=rcc_duals,
                    forbidden_arcs=forbidden_arcs,
                )

            if not exact_mode and len(heuristic_routes) >= max_routes // 2:
                # Enough good columns found heuristically — skip exact DP
                routes = heuristic_routes
                self.pricing_status = PricingStatus.HEURISTIC
                self.last_max_rc = max((r.reduced_cost for r in routes if r.reduced_cost is not None), default=0.0)
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
