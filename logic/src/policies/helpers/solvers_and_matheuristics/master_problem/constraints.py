"""
Constraint addition and management mixin for VRPPMasterProblem.

Attributes:
    VRPPMasterProblemConstraintsMixin: Mixin for adding cutting planes to the master problem.

Example:
    >>> master.add_subset_row_cut({1, 2, 3})

Parts are relocated to connectivity_constraints.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import gurobipy as gp

from .connectivity_constraints import MasterConnectivityConstraintsMixin

if TYPE_CHECKING:
    from .problem_support import MasterProblemSupport

logger = logging.getLogger(__name__)


def _cut_decision(master: Any, cut_type: str, cut_id: Any, accepted: bool) -> bool:
    """Return ``accepted`` and, only when a trace is bound, record the cut."""
    if getattr(master, "_exec_trace", None) is not None:
        from logic.src.policies.helpers.solvers_and_matheuristics.exact_trace import emit_cut

        emit_cut(master, cut_type, cut_id, accepted)
    return accepted


class VRPPMasterProblemConstraintsMixin(MasterConnectivityConstraintsMixin):
    """
    Mixin containing methods for adding cutting planes to the master problem.

    Attributes:
        model: Reference to the Gurobi model.
        routes: List of routes in the master problem.
        lambda_vars: Decision variables for each route.
    """

    def add_edge_clique_cut(
        self: MasterProblemSupport,
        u: int,
        v: int,
        coefficients: Optional[Dict[int, float]] = None,
        rhs: float = 1.0,
    ) -> bool:
        r"""
        Add an Edge Clique cut specifically covering edge (u, v).

        Formulation:
            Σ_{k ∈ C} λ_k + Σ_{k ∈ \bar{C}} α_k λ_k <= |C| - 1

        Where C is a minimal cover of routes using edge (u, v), and α_k are
        lifting coefficients for routes not in the cover.

        Args:
            u: Start node of the edge.
            v: End node of the edge.
            coefficients: Mapping from route index to lifting coefficient α_k.
                If None, defaults to 1.0 for all routes using the edge.
            rhs: Right-hand side of the inequality (|C| - 1).


        Returns:
            True if the cut was added, False if it was redundant.
        """
        if self.model is None or not self.lambda_vars:
            return False

        edge_tuple = (min(u, v), max(u, v))
        key = edge_tuple

        if key in self.active_edge_clique_cuts:
            return _cut_decision(self, "edge_clique", edge_tuple, False)

        lhs = gp.LinExpr()
        found_columns = False
        for idx, route in enumerate(self.routes):
            # Check for edge (u, v) in route path
            nodes = [0] + route.nodes + [0]
            contains_edge = False
            for i in range(len(nodes) - 1):
                if tuple(sorted((nodes[i], nodes[i + 1]))) == edge_tuple:
                    contains_edge = True
                    break

            if contains_edge:
                coeff = 1.0
                if coefficients is not None:
                    coeff = coefficients.get(idx, 1.0)
                lhs.add(self.lambda_vars[idx], coeff)
                found_columns = True

        if not found_columns:
            return _cut_decision(self, "edge_clique", edge_tuple, False)

        constr = self.model.addConstr(lhs <= rhs, name=f"Edge_Clique_{edge_tuple[0]}_{edge_tuple[1]}")
        stored_coeffs = coefficients if coefficients is not None else {}
        self.active_edge_clique_cuts[key] = (constr, stored_coeffs)
        self.global_cut_pool.add_cut("edge_clique", edge_tuple)
        self.model.update()
        return _cut_decision(self, "edge_clique", edge_tuple, True)

    def add_subset_row_cut(
        self: MasterProblemSupport,
        node_set: Union[List[int], Set[int], FrozenSet[int]],
    ) -> bool:
        """
        Add a 3-Subset Row Inequality (3-SRI) to the master problem.

        SRI Formulation:
            Σ_{k} ⌊ 1/2 * |S ∩ Route_k| ⌋ * λ_k <= 1

        For any subset S of 3 nodes, the sum of routes visiting at least 2 nodes
        in S cannot exceed 1. This strengthens the relaxation by cutting off
        fractional solutions where three routes visit pairs (node1, node2),
        (node2, node3), and (node1, node3) with value 0.5 each.

        Args:
            node_set: A list of exactly 3 customer nodes.

        Returns:
            True if the cut was successfully added to the model.
        """
        if self.model is None or len(node_set) != 3 or not self.lambda_vars:
            return False

        nodes = sorted(node_set)
        subset_frozenset = frozenset(nodes)
        if subset_frozenset in self.active_sri_cuts:
            return _cut_decision(self, "sri", subset_frozenset, False)

        cut_name = f"SRI_{nodes[0]}_{nodes[1]}_{nodes[2]}"

        lhs = gp.LinExpr()
        found_columns = False
        coeff_dict: Dict[str, float] = {}
        for idx, route in enumerate(self.routes):
            # Check node coverage for SRI violation calculation
            count = sum(1 for n in nodes if n in route.node_coverage)
            coeff = count // 2
            if coeff > 0:
                lhs.add(self.lambda_vars[idx], float(coeff))
                content = ",".join(map(str, route.nodes))
                route_h = hashlib.md5(content.encode()).hexdigest()
                coeff_dict[route_h] = float(coeff)
                found_columns = True

        if not found_columns:
            return _cut_decision(self, "sri", subset_frozenset, False)

        new_cut = self.model.addConstr(lhs <= 1.0, name=cut_name)
        self.active_sri_cuts[subset_frozenset] = new_cut
        # Archiving to Global Pool (SRIs are globally valid); store vector for orthogonality checks
        self.global_cut_pool.add_cut("sri", (subset_frozenset, coeff_dict))
        self.model.update()
        return _cut_decision(self, "sri", subset_frozenset, True)

    def add_capacity_cut(
        self: MasterProblemSupport,
        node_list: List[int],
        rhs: float,
        coefficients: Optional[Dict[int, float]] = None,
        is_global: bool = True,
        _skip_pool: bool = False,
    ) -> bool:
        """Add a Rounded Capacity Cut (RCC) to the master problem.

        The cut enforces that the number of edges crossing the boundary δ(S)
        of the node set S must be at least twice the minimum number of
        vehicles required to serve S.

        Constraint:  Σ_{k: route_k crosses δ(S)} crossings_k(S) * λ_k  >=  rhs

        Args:
            node_list: Nodes in set S.
            rhs: Right-hand side (2 * ⌈demand(S) / Q⌉).
            coefficients: Optional mapping for route lifting.
            is_global: Whether the cut is globally valid.
            _skip_pool: Internal flag to avoid re-archiving.

        Returns:
            True if the cut was newly added, False if it already exists/failed.
        """

        if self.model is None:
            return False

        node_set = frozenset(node_list)
        if node_set in self.active_capacity_cuts:
            # Already have this cut
            return _cut_decision(self, "capacity", node_set, False)

        # Build column-based representation of the cut
        lhs = gp.LinExpr()
        for idx, route in enumerate(self.routes):
            crossings = self._count_crossings(route, node_set)
            if crossings > 0:
                lhs += float(crossings) * self.lambda_vars[idx]

        name = f"capacity_{abs(hash(node_set))}"
        constr = self.model.addConstr(lhs >= rhs, name=name)
        self.active_capacity_cuts[node_set] = constr
        if is_global and not _skip_pool:
            # Archive with the actual RHS so descendant nodes replay the correct
            # strength (2*ceil(demand(S)/Q)), not a hard-coded placeholder.
            self.global_cut_pool.add_cut("rcc", (node_set, rhs))

        if self.model is not None:
            self.model.update()
        return _cut_decision(self, "capacity", node_set, True)

    def add_lci_cut(
        self: MasterProblemSupport,
        node_list: List[int],
        rhs: float,
        coefficients: Dict[int, float],
        node_alphas: Optional[Dict[int, float]] = None,
        arc: Optional[Tuple[int, int]] = None,
    ) -> bool:
        """
        Add a Lifted Cover Inequality (LCI) to the master problem.

        LCI formulation (Barnhart et al. 2000, §4):
            Σ_{k ∈ C} λ_k + Σ_{k ∉ C} α_k λ_k  ≤  |C| - 1

        LCI cuts are stored in `active_lci_cuts` (separate from `active_capacity_cuts`
        which holds >= RCC constraints) so that dual extraction uses the correct
        sign: Pi >= 0 for a <= constraint in a MAX LP.

        The optional ``node_alphas`` mapping (node_id → lifting coefficient α_i)
        enables the pricing subproblem to apply the correct per-node penalty when
        the cut's dual variable γ is nonzero.  Per Barnhart et al. (2000) §4.2:

            c'_lm^k = c_lm^k + π_lm + α_lm^k · γ_lm

        The optional ``arc`` parameter identifies the specific saturated arc (i, j)
        that generated this LCI (used by SaturatedArcLCIEngine).  When set, the
        pricing DP applies the dual **only** when traversing arc (i, j), matching
        the paper's arc-level formula exactly.  For node/capacity LCI (e.g.
        PhysicalCapacityLCIEngine) leave ``arc=None``; the DP then falls back to
        the node-visit approximation.

        Args:
            node_list: Customer nodes defining the cover set S.
            rhs: Right-hand side of the inequality (|C| - 1 or lifting bound K).
            coefficients: Route-index → route-level lifting coefficient (α_k).
            node_alphas: Node-id → per-node lifting coefficient (α_i).
                         If None, defaults to 1.0 for cover nodes, 0.0 for others.
            arc: Optional source arc (i, j) for arc-saturation LCI.

        Returns:
            bool: True if the cut was successfully added.
        """

        if self.model is None:
            return False

        node_set = frozenset(node_list)
        if node_set in self.active_lci_cuts:
            return _cut_decision(self, "lci", node_set, False)

        lhs = gp.LinExpr()
        for idx, coeff in coefficients.items():
            if idx < len(self.lambda_vars):
                lhs += coeff * self.lambda_vars[idx]

        if lhs.size() == 0:
            return _cut_decision(self, "lci", node_set, False)

        name = f"lci_{abs(hash(node_set))}"
        constr = self.model.addConstr(lhs <= rhs, name=name)
        self.active_lci_cuts[node_set] = constr
        # Store per-node alphas for pricing dual integration.
        effective_node_alphas: Dict[int, float] = node_alphas if node_alphas is not None else {}
        self.active_lci_node_alphas[node_set] = effective_node_alphas
        # Store source arc (None for node/capacity LCI, set for arc-saturation LCI).
        self.active_lci_arcs[node_set] = arc
        # Archive with rhs, route coefficients, node_alphas, AND arc for descendant replay.
        self.global_cut_pool.add_cut("lci", (node_set, rhs, coefficients, effective_node_alphas, arc))
        if self.model is not None:
            self.model.update()
        return _cut_decision(self, "lci", node_set, True)

    def add_multistar_cut(
        self: "MasterProblemSupport",
        node_list: List[int],
        coefficients: Dict[int, float],
    ) -> bool:
        """Add a Generalized Multistar Inequality cut (Letchford, Eglese, Lysgaard 2002).

        Constraint stored in Gurobi as:
            Σ_k (-a_k) · λ_k  ≤  0
        where ``coefficients[k] = -a_k`` for routes with |a_k| > 1e-6.

        The dual γ_S ≥ 0 (Pi of a ≤ constraint in a MAX LP) is extracted in
        ``_extract_duals`` into ``self.dual_multistar_cuts`` and emitted by
        ``get_reduced_cost_coefficients`` under the ``"multistar_duals"`` key,
        where ``solver.py._extend_label`` applies the arc-level penalty.

        Args:
            node_list: Customer nodes forming the cover set S.
            coefficients: {route_index: -a_k} for routes with |a_k| > 1e-6.

        Returns:
            True if the cut was newly added to the model; False if duplicate or
            if no column has a non-negligible coefficient.
        """
        if self.model is None or not self.lambda_vars:
            return False

        node_set = frozenset(node_list)
        if node_set in self.active_multistar_cuts:
            return _cut_decision(self, "multistar", node_set, False)

        lhs = gp.LinExpr()
        for idx, coeff in coefficients.items():
            if idx < len(self.lambda_vars) and abs(coeff) > 1e-9:
                lhs += coeff * self.lambda_vars[idx]

        if lhs.size() == 0:
            return _cut_decision(self, "multistar", node_set, False)

        name = f"multistar_{abs(hash(node_set))}"
        constr = self.model.addConstr(lhs <= 0.0, name=name)
        self.active_multistar_cuts[node_set] = constr
        # Archive centrally; the cut persists across B&B nodes through the single
        # shared master object, which is never rebuilt mid-search.
        self.global_cut_pool.add_cut("multistar", (node_set, coefficients))
        self.model.update()
        return _cut_decision(self, "multistar", node_set, True)

    def add_set_packing_capacity_cut(self: MasterProblemSupport, node_list: List[int], rhs: float) -> bool:
        """
        Add a Rounded Capacity Cut (RCC) to the master problem.

        The cut enforces that the number of edges crossing the boundary δ(S)
        of the node set S must be at least twice the minimum number of
        vehicles required to serve S.

        Constraint:  Σ_{k: route_k crosses δ(S)} crossings_k(S) * λ_k  >=  rhs

        Args:
            node_list: Nodes in set S.
            rhs: Right-hand side (2 * ⌈demand(S) / Q⌉).

        Returns:
            True if the cut was newly added, False if it already exists/failed.
        """
        if self.model is None:
            return False

        node_set = frozenset(node_list)
        if node_set in self.active_rcc_cuts:
            # Already have this cut
            return _cut_decision(self, "capacity", node_set, False)

        # Build column-based representation of the cut
        lhs = gp.LinExpr()
        for idx, route in enumerate(self.routes):
            crossings = self._count_crossings(route, node_set)
            if crossings > 0:
                lhs += float(crossings) * self.lambda_vars[idx]

        name = f"rcc_{abs(hash(node_set))}"
        constr = self.model.addConstr(lhs >= rhs, name=name)
        self.active_rcc_cuts[node_set] = constr
        self.model.update()
        return _cut_decision(self, "capacity", node_set, True)
