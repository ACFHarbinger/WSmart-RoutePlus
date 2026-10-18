"""Label-correcting dynamic program of the RCSPP pricing solver.

Attributes:
    RCSPPLabelingMixin: Label setting, extension and depot closing for ``RCSPPSolver``.

Example:
    >>> class RCSPPSolver(RCSPPLabelingMixin): ...
"""

from __future__ import annotations

import heapq
import logging
import time
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from logic.src.policies.helpers.solvers_and_matheuristics.common.route import Route
from logic.src.policies.helpers.solvers_and_matheuristics.pricing.labels import Label
from logic.src.policies.helpers.solvers_and_matheuristics.pricing.pricing_types import (
    PricingStatus,
    _LCICoverItem,
    _MultistarItem,
)

logger = logging.getLogger(__name__)


class RCSPPLabelingMixin:
    """Label-correcting algorithm, label extension and depot closing of :class:`RCSPPSolver`."""

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
        adaptive_max_labels = self.max_labels if exact_mode else min(self.max_labels, int(self.n_nodes**1.5 * 200))

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

        if self._timed_out:
            self.pricing_status = PricingStatus.TIMED_OUT
            max_potential = max(
                (lbl.reduced_cost + self.bounds_to[lbl.node] for _, _, lbl in queue),
                default=-float("inf"),
            )
            self.last_max_rc = max(global_max_rc, max_potential)
        elif self.labels_generated >= self.max_labels:
            self.pricing_status = PricingStatus.PARTIAL
            max_potential = max(
                (lbl.reduced_cost + self.bounds_to[lbl.node] for _, _, lbl in queue),
                default=-float("inf"),
            )
            self.last_max_rc = max(global_max_rc, max_potential)
        else:
            self.pricing_status = PricingStatus.EXHAUSTIVE if exact_mode else PricingStatus.HEURISTIC
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

        # Vehicle limit dual penalty: each generated route consumes 1 vehicle
        if not self.is_farkas and self.vehicle_dual > 1e-9:
            rc_delta -= self.vehicle_dual
        elif self.is_farkas and abs(self.vehicle_dual) > 1e-9:
            rc_delta += self.vehicle_dual

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
