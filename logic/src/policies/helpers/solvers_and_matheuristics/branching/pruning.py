"""Branching utilities for Branch-and-Price-and-Cut solvers.

Provides reusable branching components — branching constraint application,
strong branching evaluation, and forced-set extraction — that are independent
of any specific BPC pipeline orchestrator.

Attributes:
-----------
BPCPruningException               — Raised when a B&B node is pruned by bound.
reset_master_constraints          — Restore master LP to base-state constraint senses.
apply_route_level_branching_filters — Disable routes violating a single constraint.
apply_branching_to_master         — Apply all active constraints to master column pool.
perform_strong_branching          — Evaluate branching candidates by restricted LP.
extract_forced_sets_from_constraints — Derive forced/forbidden node sets from constraints.

Example:
    None

References
----------
Barnhart, Hane & Vance (2000) Operations Research 48(2):318-326.
Ryan & Foster (1981) — node-pair branching.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import gurobipy as gp
from gurobipy import GRB

from logic.src.policies.helpers.solvers_and_matheuristics import (
    AnyBranchingConstraint,
    BranchNode,
    VRPPMasterProblem,
)
from logic.src.policies.helpers.solvers_and_matheuristics.branching.constraints import (
    EdgeBranchingConstraint,
    FleetSizeBranchingConstraint,
    NodeVisitationBranchingConstraint,
    RyanFosterBranchingConstraint,
)

logger = logging.getLogger(__name__)


# Public aliases (no leading underscore) for external use
class BPCPruningException(Exception):
    """Exception raised when a node is pruned by a bound (e.g., Lagrangian).

    Attributes:
        message (str): Explanation of the pruning reason.
    """

    pass


# ---------------------------------------------------------------------------
# Static Helpers for Master Problem State Management
# ---------------------------------------------------------------------------


def reset_master_constraints(master: VRPPMasterProblem) -> None:
    """Resets vehicle limits and node senses to their base problem state.

    Args:
        master: The master problem instance to reset.
    """
    if master.model is None:
        return

    temp_c = master.model.getConstrByName("temp_min_vehicles")
    if temp_c:
        master.model.remove(temp_c)

    constr = master.model.getConstrByName("vehicle_limit")
    if constr:
        if master.vehicle_limit is not None:
            constr.RHS = float(master.vehicle_limit)
            constr.Sense = GRB.LESS_EQUAL
        else:
            master.model.remove(constr)

    # Reset coverage senses
    for node in range(1, master.n_nodes + 1):
        constr = master.model.getConstrByName(f"coverage_{node}")
        if constr:
            if node in master.mandatory_nodes:
                constr.Sense = GRB.EQUAL if master.strict_set_partitioning else GRB.GREATER_EQUAL
            else:
                constr.Sense = GRB.LESS_EQUAL
            constr.RHS = 1.0


def apply_route_level_branching_filters(master: VRPPMasterProblem, bc: AnyBranchingConstraint) -> None:
    """Applies route-level feasibility filters based on branching constraints.

    Args:
        master: The master problem instance.
        bc: The branching constraint to apply.
    """
    for route, var in zip(master.routes, master.lambda_vars, strict=False):
        if var.UB > 0.5 and not bc.is_route_feasible(route):
            var.UB = 0.0


def apply_branching_to_master(
    master: VRPPMasterProblem,
    branching_constraints: List[AnyBranchingConstraint],
    branching_strategy: str = "divergence",
) -> None:
    """
    Filter the Master Problem column pool by disabling routes that violate
    branching constraints at the current B&B node.

    Column Contamination Problem:
    ------------------------------
    The Master Problem maintains a global column pool across the entire B&B tree.
    When we branch at a node, we add constraints (e.g., "nodes r and s must be
    together" or "arc (u,v) is forbidden"). However, the existing column pool
    may contain routes generated at ancestor nodes that violate these new constraints.

    If we don't filter these routes, the LP solver will select them, producing
    fractional solutions that violate the branching decisions. This causes:
    1. Incorrect LP bounds (too optimistic).
    2. Infinite branching loops (same fractional solution reappears).
    3. Invalid integer solutions.

    COLUMN BACKTRACKING & RE-ACTIVATION:
    ------------------------------------
    Since we use a global column pool, we must reset all columns to a "clean"
    state (UB=1.0) before applying the current path of branching constraints.
    Failure to do this causes columns disabled in deep branches to remain
    disabled when backtracking, leading to incorrect bounds and potential
    missed optimal solutions.

    Solution:
    ---------
    We temporarily disable violating routes by setting their upper bound to 0:
        var.UB = 0.0

    This is more efficient than:
    - Deleting and re-adding columns (expensive Gurobi operations).
    - Maintaining separate column pools per node (memory intensive).
    - Adding explicit branching constraints to Gurobi (creates dense constraint matrix).

    The routes remain in the model structure but cannot be selected in the LP solution.

    Implementation:
    ---------------
    For each route k and its corresponding Gurobi variable λ_k:
    1. Reset upper bound to 1.0 (clean state for this node).
    2. Check if route satisfies all active branching constraints (ancestors + current).
    3. If any constraint is violated, set var.UB = 0.0 to disable the route.

    Args:
        master: Master problem instance with Gurobi model.
        branching_constraints: Complete list of branching constraints from root
            to the current node (all ancestors + current node).
        branching_strategy: Current branching rule name (e.g., "ryan_foster").

    References:
    -----------
    Ryan-Foster branching (1981) is used here because it appropriately modifies the
    Resource-Constrained Shortest Path Problem (RCSPP) used for VRPP pricing by
    forbidding or enforcing pairs of nodes in the generated routes.
    """
    if master.model is None:
        return

    # Task 3/6: Harden Ryan-Foster Exactness
    if branching_strategy == "ryan_foster" and not master.strict_set_partitioning:
        raise RuntimeError(
            "Mathematical Exactness Violation: Ryan-Foster branching requires "
            "strict Set Partitioning (== 1.0). Current Master Problem allows "
            "Set Covering (>= 1.0), which can erroneously prune optimal solutions."
        )

    # RESET-THEN-REFILTER PROTOCOL
    # Step 1: Reset all columns to UB=1.0 (clean slate for this node).
    #   Prevents constraints from a now-dead sibling branch from sticking
    #   to columns that are valid in the current node's branch.
    # Step 2 (the constraint loop below): Re-apply every constraint in the
    #   current node's ancestor chain. A column added in a deep descendant
    #   branch that is infeasible at the root level will be filtered here
    #   because the root-level constraint that makes it infeasible is always
    #   present in the ancestor chain.
    # INVARIANT: Every column in master.routes satisfies the root LP's
    #   structural constraints (capacity, degree). Only branching constraints
    #   can cause a column to be filtered at a descendant node.
    for var in master.lambda_vars:
        var.UB = 1.0

    # Task 2: Reset vehicle limit and node senses to base state
    reset_master_constraints(master)

    if not branching_constraints:
        master.model.update()
        return

    # Track effective bounds for global constraints
    max_vehicles: float = master.vehicle_limit if master.vehicle_limit is not None else float("inf")
    min_vehicles: float = 0.0
    forced_nodes: Set[int] = set()

    # Apply all constraints in the current B&B path
    for bc in branching_constraints or []:
        # Global: Fleet Size
        if isinstance(bc, FleetSizeBranchingConstraint):
            if bc.is_upper:
                max_vehicles = min(max_vehicles, bc.limit)
            else:
                min_vehicles = max(min_vehicles, bc.limit)
        # Global: Node Visitation override
        elif isinstance(bc, NodeVisitationBranchingConstraint) and bc.forced:
            forced_nodes.add(bc.node)

        # Route-level filtering
        apply_route_level_branching_filters(master, bc)

    # Apply global fleet limits to the master model
    vl_constr = master.model.getConstrByName("vehicle_limit")
    if vl_constr:
        # If we have a lower bound that equals the upper bound, force equality
        if min_vehicles >= max_vehicles - 1e-4:
            vl_constr.Sense = GRB.EQUAL
            vl_constr.RHS = max_vehicles
        else:
            vl_constr.Sense = GRB.LESS_EQUAL
            vl_constr.RHS = max_vehicles
    elif max_vehicles < float("inf"):
        master.model.addConstr(gp.quicksum(master.lambda_vars) <= max_vehicles, name="vehicle_limit")

    if min_vehicles > 0.5:
        master.model.addConstr(gp.quicksum(master.lambda_vars) >= min_vehicles, name="temp_min_vehicles")

    # Apply node visitation forcing (Sense change)
    for node in forced_nodes:
        constr = master.model.getConstrByName(f"coverage_{node}")
        if constr:
            constr.Sense = GRB.EQUAL
            constr.RHS = 1.0

    master.model.update()


def perform_strong_branching(  # noqa: C901
    master: VRPPMasterProblem,
    candidates: List[Tuple[int, List[Tuple[int, int]], List[Tuple[int, int]], float]],
    current_node: Optional[BranchNode] = None,
    strong_branching_size: int = 5,
) -> Optional[Tuple[int, List[Tuple[int, int]], List[Tuple[int, int]], float]]:
    """Evaluates branching candidates by solving child LP relaxations (lookahead).

    Selects the branch that maximizes the estimated lower bound improvement.
    Note: This is a restricted lookahead that does NOT solve full column
    generation subproblems at children, only the sifted master problem.

    Args:
        master: The master problem instance.
        candidates: List of branching candidates.
        current_node: Current B&B node.
        strong_branching_size: Number of candidates to evaluate.

    Returns:
        Optional[Tuple]: The best candidate selected by lookahead evaluation.
    """
    if not candidates:
        return None

    # Limit the number of candidates to evaluate to prevent excessive overhead
    eval_candidates = candidates[:strong_branching_size]
    if len(eval_candidates) <= 1:
        return eval_candidates[0]

    parent_obj = master.model.ObjVal if master.model is not None and master.model.Status == GRB.OPTIMAL else 0.0
    best_candidate = eval_candidates[0]
    best_score = -1.0

    # Cache the current basis for warm-starting
    parent_basis = None
    if master.model is not None and master.model.Status == GRB.OPTIMAL:
        parent_basis = master.save_basis()

    for cand in eval_candidates:
        cand_id, left_branch, right_branch, _ = cand

        def evaluate_branch(arc_set: List[Tuple[int, int]]) -> float:
            """Evaluates a potential branch by solving the master LP with disabled columns."""
            forbidden_arcs_set = set(arc_set)
            disabled_vars = []

            # Temporarily disable columns that violate the branch (x_e = 0)
            for idx, var in enumerate(master.lambda_vars):
                if var.UB < 0.5:
                    continue
                route = master.routes[idx]
                full_path = [0] + route.nodes + [0]
                if any((full_path[i], full_path[i + 1]) in forbidden_arcs_set for i in range(len(full_path) - 1)):
                    var.UB = 0.0
                    disabled_vars.append(var)

            if master.model is not None:
                master.model.optimize()
                obj = master.model.ObjVal if master.model.Status == GRB.OPTIMAL else -float("inf")
            else:
                obj = -float("inf")

            # Revert bounds
            for var in disabled_vars:
                var.UB = 1.0

            if parent_basis is not None:
                master.restore_basis(*parent_basis)

            return parent_obj - obj if obj != -float("inf") else float("inf")

        left_deg = evaluate_branch(left_branch)
        right_deg = evaluate_branch(right_branch)

        # Product-based evaluation metric (Score = max(ΔL, ε) * max(ΔR, ε))
        score = max(left_deg, 1e-6) * max(right_deg, 1e-6)

        if score > best_score:
            best_score = score
            best_candidate = cand

    # Restore master state fully and resolve once to ensure basis is clean
    if master.model is not None:
        if parent_basis is not None:
            master.restore_basis(*parent_basis)

        master.model.optimize()
        if master.model.Status != GRB.OPTIMAL:
            logger.warning(
                "Strong branching left master in non-optimal state (status=%d). Falling back to first candidate.",
                master.model.Status,
            )
            # Attempt a full reset by re-enabling all UBs and resolving.
            for var in master.lambda_vars:
                var.UB = 1.0
            master.model.optimize()
            if master.model.Status != GRB.OPTIMAL:
                logger.error("Master could not be restored after strong branching.")
            return eval_candidates[0]  # safe fallback

    return best_candidate


def extract_forced_sets_from_constraints(
    branching_constraints: Optional[List[AnyBranchingConstraint]],
) -> Tuple[Set[int], Set[int]]:
    """
    Extract forced-in and forced-out customer sets from the active branching path.

    Only `NodeVisitationBranchingConstraint` instances carry hard node-level
    fixings. Edge and Ryan-Foster constraints fix arcs, not nodes directly,
    so they are not reflected here (the LR subproblem is node-selection based).

    Args:
        branching_constraints: Active constraints from root to current node.

    Returns:
        (forced_in, forced_out) sets of customer indices.
    """
    forced_in: Set[int] = set()
    forced_out: Set[int] = set()
    for bc in branching_constraints or []:
        if isinstance(bc, NodeVisitationBranchingConstraint):
            if bc.forced:
                forced_in.add(bc.node)
            else:
                forced_out.add(bc.node)

    return forced_in, forced_out


def _prepare_node_relaxation(
    master: VRPPMasterProblem,
    pricing_solver: Any,
    branching_constraints: Optional[List[AnyBranchingConstraint]],
) -> Tuple[int, int]:
    """Harden the pricing relaxation for a B&B node and check its arc branches.

    Enforces strict elementarity for nodes in Ryan-Foster pairs, forced node
    visits and active SRI subsets, so the ng-relaxation cannot bypass them via
    cycles; then raises ``RuntimeError`` if the must-use arc branches conflict
    (more than one must-use arc leaving or entering a node).

    Args:
        master: Master problem instance.
        pricing_solver: RCSPP pricing solver.
        branching_constraints: Active constraints.

    Returns:
        Tuple[int, int]: (number of elementary nodes, value returned by
        ``enforce_elementarity``, or 0 when no node needed it).
    """
    # Task 5: Arc Conflict Pre-check (Exactness Rule)
    # We must enforce strict elementarity for nodes involved in active SRIs or
    # Ryan-Foster node-pair branching to prevent the ng-relaxation from
    # bypassing these constraints via cycles.
    elementary_nodes: Set[int] = set()

    # Extract nodes from Ryan-Foster branches
    for bc in branching_constraints or []:
        if isinstance(bc, RyanFosterBranchingConstraint):
            elementary_nodes.add(bc.node_r)
            elementary_nodes.add(bc.node_s)
        elif isinstance(bc, NodeVisitationBranchingConstraint) and bc.forced:
            # Force nodes are also strictly elementary to prevent cyclic bypass
            elementary_nodes.add(bc.node)

    # Extract nodes from active SRI subsets
    for subset in master.active_sri_cuts.keys():
        elementary_nodes.update(subset)

    added_elem = 0
    if elementary_nodes:
        added_elem = pricing_solver.enforce_elementarity(list(elementary_nodes))

    # Task 5: Arc Conflict Pre-check (Exactness Rule)
    # Scan branching constraints for conflicting must_use arcs.
    # At most one must_use arc can exit node u, and at most one can enter node v.
    out_arcs: Dict[int, int] = {}
    in_arcs: Dict[int, int] = {}
    for bc in branching_constraints or []:
        if isinstance(bc, EdgeBranchingConstraint) and bc.must_use:
            if bc.u in out_arcs and out_arcs[bc.u] != bc.v:
                raise RuntimeError(f"Conflicting must_use out-arcs at node {bc.u}")
            if bc.v in in_arcs and in_arcs[bc.v] != bc.u:
                raise RuntimeError(f"Conflicting must_use in-arcs at node {bc.v}")
            out_arcs[bc.u] = bc.v
            in_arcs[bc.v] = bc.u
    return len(elementary_nodes), added_elem
