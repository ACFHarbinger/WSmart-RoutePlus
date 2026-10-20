"""Branching strategies and heuristics for VRPP Column Generation.

This module provides various branching strategies including edge-based,
Ryan-Foster node-pair, and advanced divergence branching with spatial partitioning.

Attributes:
    EdgeBranching: Strategy for branching on fractional arcs.
    MultiEdgePartitionBranching: Advanced divergence branching with spatial fleet partitioning.
    RyanFosterBranching: Strategy for branching on node pairs.
    FleetSizeBranching: Strategy for branching on total fleet usage.
    NodeVisitationBranching: Strategy for branching on optional node visitation.

Example:
    >>> arc, flow = EdgeBranching.find_branching_arc(routes, route_values)
    >>> if arc:
    ...     left, right = EdgeBranching.create_child_nodes(parent, arc[0], arc[1], flow)

Parts are relocated to partition_branching, node_fleet_branching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from logic.src.policies.helpers.solvers_and_matheuristics.branching.constraints import (
    EdgeBranchingConstraint,
    NodeVisitationBranchingConstraint,
)
from logic.src.policies.helpers.solvers_and_matheuristics.common.route import Route

if TYPE_CHECKING:
    from logic.src.policies.helpers.solvers_and_matheuristics.common.node import BranchNode


class EdgeBranching:
    """
    Produces two child nodes:
        left  → x_{uv} = 1  (arc MUST be used)
        right → x_{uv} = 0  (arc is FORBIDDEN)

    Attributes:
        compute_arc_flow: Static method to aggregate fractional arc flows.
        find_branching_arc: Static method to select the best arc to branch on.
        create_child_nodes: Static method to generate child B&B nodes.
    """

    @staticmethod
    def compute_arc_flow(
        routes: List[Route],
        route_values: Dict[int, float],
    ) -> Dict[Tuple[int, int], float]:
        """
        Aggregate fractional arc flows from the LP solution.

        Args:
            routes: All routes in the master problem column pool.
            route_values: LP solution values {route_index: λ_k}.

        Returns:
            Mapping (u, v) → aggregated fractional flow.
        """
        arc_flow: Dict[Tuple[int, int], float] = {}
        for idx, lam in route_values.items():
            if lam < 1e-9:
                continue
            full_path = [0] + routes[idx].nodes + [0]
            for i in range(len(full_path) - 1):
                arc = (full_path[i], full_path[i + 1])
                arc_flow[arc] = arc_flow.get(arc, 0.0) + lam
        return arc_flow

    @staticmethod
    def find_branching_arc(
        routes: List[Route],
        route_values: Dict[int, float],
        tol: float = 1e-5,
    ) -> Optional[Tuple[Tuple[int, int], float]]:
        """
        Select the arc with fractional flow closest to 0.5.

        Args:
            routes: All routes in the master problem.
            route_values: Current LP solution {route_index: λ_k}.
            tol: Integrality tolerance.

        Returns:
            Arc (u, v) to branch on, or None if the solution is already integer.
        """
        arc_flow = EdgeBranching.compute_arc_flow(routes, route_values)
        best_arc: Optional[Tuple[int, int]] = None
        best_frac = -1.0
        best_arc_flow = 0.0

        for arc, flow in arc_flow.items():
            frac = min(flow, 1.0 - flow)
            if frac > tol and frac > best_frac:
                best_frac = frac
                best_arc = arc
                best_arc_flow = flow

        if best_arc is None:
            return None
        return best_arc, best_arc_flow

    @staticmethod
    def create_child_nodes(
        parent: "BranchNode",
        u: int,
        v: int,
        flow: float = 0.5,
    ) -> Tuple["BranchNode", "BranchNode"]:
        """Create left (must-use) and right (forbidden) child nodes.

        Args:
            parent: The node being branched.
            u: Arc origin.
            v: Arc destination.
            flow: Fractional flow on the arc (used for priority hints).

        Returns:
            (left_child, right_child)
        """

        from logic.src.policies.helpers.solvers_and_matheuristics.common.node import BranchNode

        left_bc = EdgeBranchingConstraint(u, v, must_use=True)
        right_bc = EdgeBranchingConstraint(u, v, must_use=False)

        return (
            BranchNode(
                constraints=[left_bc],
                parent=parent,
                depth=parent.depth + 1,
                lp_bound_hint=flow,
                branching_rule="edge",
            ),
            BranchNode(
                constraints=[right_bc],
                parent=parent,
                depth=parent.depth + 1,
                lp_bound_hint=1.0 - flow,
                branching_rule="edge",
            ),
        )


class HierarchicalStrongBranching:
    """Hierarchical strong branching for VRPP (Pessoa, Sadykov, Uchoa 2020).

    Addresses the profit-selection × routing duality by branching on
    node-visitation decisions y_v ∈ {0,1} BEFORE arc-level decisions.
    This directly resolves the "which nodes to visit" half of the duality.

    Three-phase pipeline (VRPSolver §3.2):
      Phase 0 (heuristic, O(1)):  score candidates by most-fractional y_v.
      Phase 1 (restricted LP, O(k)):  solve restricted master LP without
                                       re-pricing for top k₁ candidates.
      Phase 2 (full LP, O(k₁)):  solve restricted master LP for top k₂ ≤ k₁
                                   candidates (optionally with column generation).

    Branching hierarchy for VRPP:
      1. Node-visitation (y_v): directly resolves selection–routing duality.
      2. Ryan-Foster (node pair co-visit): tightens set-partitioning.
      3. Edge (arc):  fallback for residual arc fractionality.

    References:
        Boussier, Feillet, Gendreau (2007) 4OR 5:211–230.
        Pessoa, Sadykov, Uchoa, Vanderbeck (2020) Math. Prog. 183:483–523.

    Attributes:
        find_best_y_branching: Static method to find the best optional node to branch on.
    """

    @staticmethod
    def find_best_y_branching(
        routes: List[Route],
        route_values: Dict[int, float],
        optional_nodes: Set[int],
        master: Optional[Any] = None,
        phase1_size: int = 8,
        phase2_size: int = 4,
        tol: float = 1e-4,
    ) -> Optional[Tuple[int, float]]:
        """Select the best optional node to branch on via hierarchical strong branching.

        Phase 0: Rank all fractional y_v by |y_v - 0.5| (most-fractional first).
        Phase 1: Evaluate top phase1_size candidates by solving restricted
                 master LP (UBs of incompatible λ set to 0, no re-pricing).
        Phase 2: Among top phase2_size from Phase 1, pick by product score.

        Args:
            routes: Current column pool.
            route_values: LP solution {route_index: λ_k}.
            optional_nodes: Set of optional customer indices.
            master: Master problem (needed for Phase 1/2 LP solves). If None,
                    falls back to Phase 0 only.
            phase1_size: Number of candidates to evaluate in Phase 1.
            phase2_size: Number of candidates to evaluate in Phase 2.
            tol: Integrality tolerance.

        Returns:
            (node, y_value) for the chosen branching node, or None if integer.
        """
        # Phase 0: collect fractional y_v candidates
        node_visits: Dict[int, float] = {}
        for idx, lam in route_values.items():
            if lam < 1e-9:
                continue
            for v in routes[idx].node_coverage:
                if v in optional_nodes:
                    node_visits[v] = node_visits.get(v, 0.0) + lam

        fractional = [(v, y) for v, y in node_visits.items() if tol < y < 1.0 - tol]
        if not fractional:
            return None

        # Sort by most-fractional (closest to 0.5)
        fractional.sort(key=lambda x: -abs(x[1] - 0.5))
        candidates_p0 = fractional[:phase1_size]

        if master is None or not hasattr(master, "model") or master.model is None:
            # Phase 0 only
            return candidates_p0[0]

        # Phase 1: evaluate by restricted LP (no column generation)

        parent_obj = master.model.ObjVal if master.model.Status == 2 else 0.0  # GRB.OPTIMAL=2

        def _eval_y_branch(node: int, forced: bool) -> float:
            """Temporarily fix node visitation and re-solve restricted master."""
            disabled = []
            for idx, var in enumerate(master.lambda_vars):
                if var.UB < 0.5:
                    continue
                has_node = node in routes[idx].node_coverage
                # For forced=True branch: disable routes that DON'T visit node
                # For forced=False branch: disable routes that DO visit node
                if forced and not has_node or not forced and has_node:
                    var.UB = 0.0
                    disabled.append(var)
            try:
                master.model.optimize()
                obj = master.model.ObjVal if master.model.Status == 2 else -float("inf")
            except Exception:
                obj = -float("inf")
            for var in disabled:
                var.UB = 1.0
            return parent_obj - obj if obj > -float("inf") else float("inf")

        phase1_scores: List[Tuple[float, int, float]] = []
        for node, y_val in candidates_p0:
            dg_forced = _eval_y_branch(node, forced=True)
            dg_excluded = _eval_y_branch(node, forced=False)
            score = max(dg_forced, 1e-6) * max(dg_excluded, 1e-6)
            phase1_scores.append((score, node, y_val))

        # Restore master LP
        master.model.optimize()

        phase1_scores.sort(key=lambda x: -x[0])
        top = phase1_scores[:phase2_size]

        if not top:
            return candidates_p0[0]

        # Phase 2: pick by product score (already computed in Phase 1)
        best_score, best_node, best_y = top[0]
        return best_node, best_y

    @staticmethod
    def create_child_nodes(
        parent: "BranchNode",
        node: int,
        y_value: float,
    ) -> "Tuple[BranchNode, BranchNode]":
        """Create y_v = 0 (exclude) and y_v = 1 (force) child nodes.

        Following Boussier et al. (2007): the "visit" child is explored
        first in DFS to find a good primal solution early.

        Args:
            parent: The node being branched.
            node: Optional customer to branch on.
            y_value: Current fractional LP value of y_v.

        Returns:
            (visit_child, skip_child) — visit first for better primal heuristic.
        """
        from logic.src.policies.helpers.solvers_and_matheuristics.common.node import BranchNode

        visit_child = BranchNode(
            constraints=[NodeVisitationBranchingConstraint(node, forced=True)],
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=parent.lp_bound,
            branching_rule="node_visit",
        )
        skip_child = BranchNode(
            constraints=[NodeVisitationBranchingConstraint(node, forced=False)],
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=parent.lp_bound,
            branching_rule="node_visit",
        )
        return visit_child, skip_child
