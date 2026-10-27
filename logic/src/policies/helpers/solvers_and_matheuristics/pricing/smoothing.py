"""Pricing utilities for Branch-and-Price-and-Cut solvers.

Provides reusable pricing components that are independent of any specific
BPC pipeline, so they can be imported and used by TCF–ALNS–BPC–SP pipelines
or any other matheuristic that embeds a BPC pricing subproblem.

Attributes:
-----------
apply_reduced_cost_edge_fixing — LP-bound edge fixing on the master.
detect_cycles                 — Find repeated vertices in a route.
is_solution_integer           — Test integrality of a master LP solution.
                                Weak check by default; opt-in elementary +
                                set-partitioning (MS-BPC-SP) audit available.
solve_pricing_step            — Phase II positive-RC column generation.
solve_farkas_pricing_step     — Phase I Farkas column generation.
separate_cuts                 — Trigger cut separation on the master.

Example:
    None

References
----------
Irnich, Desaulniers, Desrosiers, Hadjar (2010) EJOR 211(1):75-87 — arc fixing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, cast

from logic.src.policies.helpers.solvers_and_matheuristics.branching import AnyBranchingConstraint
from logic.src.policies.helpers.solvers_and_matheuristics.common import Route
from logic.src.policies.helpers.solvers_and_matheuristics.master_problem import VRPPMasterProblem
from logic.src.policies.helpers.solvers_and_matheuristics.pricing.solver import RCSPPSolver
from logic.src.policies.helpers.solvers_and_matheuristics.search.cutting_planes import CuttingPlaneEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------


def detect_cycles(nodes: List[int]) -> List[Tuple[int, ...]]:
    """Return all repeated-vertex cycles in a route's node sequence.

    Args:
        nodes: Ordered customer-node list (no depot bookends).

    Returns:
        List of tuples, each being the repeated vertex and its occurrence
        positions.  Empty list when the path is elementary.
    """
    seen: Dict[int, List[int]] = {}
    for pos, v in enumerate(nodes):
        seen.setdefault(v, []).append(pos)
    return [tuple([v] + positions) for v, positions in seen.items() if len(positions) > 1]


# ---------------------------------------------------------------------------
# Solution integrality
# ---------------------------------------------------------------------------


def is_solution_integer(
    routes: List[Route],
    route_values: Dict[int, float],
    tol: float = 1e-6,
    require_elementary: bool = False,
) -> bool:
    """Return True when every non-zero λ_k is within *tol* of an integer.

    Args:
        routes:       Column pool.
        route_values: LP solution {route_index: λ_k}.
        tol:          Integrality tolerance.
        require_elementary: Opt-in strict audit (MS-BPC-SP semantics): besides
            integrality, every selected route (λ_k > 0.5) must be elementary
            (no repeated nodes) and the selected columns must form a valid
            node partition — no customer covered by more than one route.

    Returns:
        True iff the LP solution is integer-feasible.
    """
    if not require_elementary:
        return all(abs(val - round(val)) <= tol for idx, val in route_values.items())

    # Strict audit (ported from the former MS-BPC-SP engine fork): integrality
    # with clamping, per-route elementarity, then a node-usage consistency
    # check enforcing the set-partitioning semantics in node space.
    for idx, val in route_values.items():
        # Clamp all values to [0, 1] before testing to guard against slight
        # numerical drift in Set Covering/Partitioning LP solutions.
        clamped = max(0.0, min(1.0, val))
        if abs(clamped - round(clamped)) > tol:
            return False

        # Enforce elementarity in integer solutions. A numerically integer
        # λ_k = 1.0 on a cyclic ng-route is NOT a valid integer solution for
        # the original VRPP — it triggers further branching / ng-expansion.
        if clamped > 0.5:
            route = routes[idx]
            if len(set(route.nodes)) != len(route.nodes):
                return False

    # Global LP/IP consistency check. A numerically binary solution in
    # lambda-space (λ ∈ {0,1}) must also respect the Set Partitioning
    # semantics in node-space (Σ a_ik λ_k = 1 for all visited i): no node may
    # be covered by more than one selected column.
    node_usage: Dict[int, int] = {}
    for idx, val in route_values.items():
        if val > 0.5:
            for node in routes[idx].nodes:
                node_usage[node] = node_usage.get(node, 0) + 1
                if node_usage[node] > 1:
                    return False

    return True


# ---------------------------------------------------------------------------
# Cut separation
# ---------------------------------------------------------------------------


def separate_cuts(
    master: VRPPMasterProblem,
    cut_engine: CuttingPlaneEngine,
    max_cuts: int,
    iteration: int = 0,
    node_depth: int = 0,
    cut_orthogonality_threshold: float = 0.8,
) -> int:
    """Invoke the cut engine and return the number of cuts added.

    Args:
        master:                      Master problem instance.
        cut_engine:                  Composite or single-family cut engine.
        max_cuts:                    Maximum cuts to add in this round.
        iteration:                   Current column generation iteration.
        node_depth:                  Current B&B tree depth (affects separation
                                     strategy inside some engines).
        cut_orthogonality_threshold: Cosine-similarity ceiling for filtering
                                     near-duplicate cuts.

    Returns:
        Number of cuts successfully added to the master.
    """
    return cut_engine.separate_and_add_cuts(
        master,
        max_cuts,
        iteration=iteration,
        node_depth=node_depth,
        cut_orthogonality_threshold=cut_orthogonality_threshold,
    )


# ---------------------------------------------------------------------------
# Phase I — Farkas pricing
# ---------------------------------------------------------------------------


def solve_farkas_pricing_step(
    master: VRPPMasterProblem,
    pricing_solver: RCSPPSolver,
    branching_constraints: Optional[List[AnyBranchingConstraint]] = None,
    max_routes: int = 5,
    timeout: Optional[float] = None,
    farkas_duals: Optional[Dict[str, Any]] = None,
    strict_certification: bool = False,
) -> Tuple[int, bool]:
    """Generate columns that restore LP feasibility (Phase I / Farkas pricing).

    Uses the Farkas certificate from an infeasible master LP as a reduced-cost
    surrogate to find routes that increase LP feasibility.

    Args:
        master:                Master problem (must be in Phase I / infeasible).
        pricing_solver:        RCSPP solver.
        branching_constraints: Active branching constraints.
        max_routes:            Maximum columns to add per call.
        timeout:               Per-call wall-clock limit.
        farkas_duals:          Farkas certificate override (defaults to the
                               master's ``farkas_duals`` attribute).
        strict_certification:  Return a solver-certified exhaustion flag
                               (MS-BPC-SP semantics) instead of the add-based
                               heuristic one.

    Returns:
        (n_added, pricing_exhausted)
            n_added           — number of columns added.
            pricing_exhausted — True when no positive-Farkas column exists.
    """
    from logic.src.policies.helpers.solvers_and_matheuristics.branching.constraints import (
        EdgeBranchingConstraint,
        NodeVisitationBranchingConstraint,
        RyanFosterBranchingConstraint,
    )
    from logic.src.policies.helpers.solvers_and_matheuristics.pricing.solver import PricingStatus

    _FARKAS_TOL: float = 1e-6

    # Use the Farkas ray of the infeasible LP (master.farkas_duals); the regular
    # dual accessor still holds the previous optimal LP's duals.
    dual_info = farkas_duals or getattr(master, "farkas_duals", None) or master.get_reduced_cost_coefficients()
    node_duals: Dict[int, float] = dual_info.get("node_duals", {})
    rcc_duals: Dict = dual_info.get("rcc_duals", {})

    forced_nodes: Set[int] = set()
    rf_conflicts: Dict = {}
    forbidden_arcs: FrozenSet = frozenset()

    if branching_constraints:
        for bc in branching_constraints:
            if isinstance(bc, NodeVisitationBranchingConstraint) and bc.forced:
                forced_nodes.add(bc.node)
            elif isinstance(bc, RyanFosterBranchingConstraint) and not bc.together:
                # Ryan-Foster "separate" branches imply separation conflicts
                # (per branching/constraints.py docs and the RCSPP preprocessing
                # of branching constraints in pricing/solver.py). NOTE: latent
                # today — the kwarg is stored on the solver (pricing/solver.py)
                # but never fed into the label-correcting algorithm, so this
                # sign fix changes no active behaviour.
                rf_conflicts.setdefault(bc.node_r, set()).add(bc.node_s)
                rf_conflicts.setdefault(bc.node_s, set()).add(bc.node_r)
            elif isinstance(bc, EdgeBranchingConstraint) and not bc.must_use:
                forbidden_arcs = forbidden_arcs | frozenset([(bc.u, bc.v)])

    # Build the composite dual dict that solve() accepts as a single arg
    farkas_dual_dict: Dict[str, Any] = {
        "node_duals": node_duals,
        "rcc_duals": rcc_duals,
        "sri_duals": {},
        "edge_clique_duals": {},
    }
    # Branching constraints must reach the RCSPP here too: otherwise it proposes
    # columns the node forbids, the master rejects them as duplicates, and a
    # feasible node is declared infeasible.
    routes = pricing_solver.solve(
        dual_values=farkas_dual_dict,
        max_routes=max_routes,
        branching_constraints=branching_constraints,
        forced_nodes=forced_nodes,
        rf_conflicts=rf_conflicts,
        is_farkas=True,
        # Truncated neighbourhoods can drop every arc a branch still allows, which
        # would declare a feasible node infeasible; feasibility pricing is exact.
        exact_mode=True,
        timeout=timeout,
    )

    added = 0
    exhausted = True
    for route in routes:
        farkas_weight = sum(node_duals.get(v, 0.0) for v in route.nodes)
        if farkas_weight > _FARKAS_TOL and master.add_route(route):
            added += 1
            exhausted = False

    if strict_certification:
        # Explicit certification: only an exhaustive solve with no timeout and
        # last_max_rc <= _FARKAS_TOL can certify exhaustion.
        pricer_status = getattr(pricing_solver, "pricing_status", PricingStatus.PARTIAL)
        pricer_timed_out = getattr(pricing_solver, "_timed_out", False)
        last_max_rc = getattr(pricing_solver, "last_max_rc", -float("inf"))
        exhausted = (
            pricer_status == PricingStatus.EXHAUSTIVE
            and not pricer_timed_out
            and last_max_rc <= _FARKAS_TOL
        )

    return added, exhausted


# ---------------------------------------------------------------------------
# Phase II — Standard pricing
# ---------------------------------------------------------------------------


def solve_pricing_step(
    master: VRPPMasterProblem,
    pricing_solver: RCSPPSolver,
    branching_constraints: Optional[List[AnyBranchingConstraint]] = None,
    max_routes: int = 5,
    optimality_gap: float = 1e-4,
    rc_tolerance: float = 1e-5,
    timeout: Optional[float] = None,
    exact_mode: bool = False,
    strict_certification: bool = False,
) -> Tuple[int, bool]:
    """Generate profitable columns for Phase II column generation.

    Resolves the LP dual signal into RCSPP pricing calls, adding all routes
    with reduced cost above *rc_tolerance* to the master.

    Args:
        master:                Master problem in Phase II.
        pricing_solver:        RCSPP solver.
        branching_constraints: Active branching constraints.
        max_routes:            Maximum columns to add per call.
        optimality_gap:        Convergence tolerance on reduced cost.
        rc_tolerance:          Minimum reduced cost to accept a column.
        timeout:               Per-call wall-clock limit.
        exact_mode:            Forwarded to the RCSPP (disables neighbourhood truncation).
        strict_certification:  Compute a solver-certified exhaustion flag
                               (MS-BPC-SP semantics) instead of the weak
                               ``last_max_rc <= optimality_gap`` heuristic.

    Returns:
        (n_added, pricing_exhausted)
            n_added           — number of columns added.
            pricing_exhausted — True when max reduced cost ≤ optimality_gap
                                (weak) or when the solver certified exhaustion
                                (strict).
    """
    from logic.src.policies.helpers.solvers_and_matheuristics.branching.constraints import (
        EdgeBranchingConstraint,
        NodeVisitationBranchingConstraint,
        RyanFosterBranchingConstraint,
    )
    from logic.src.policies.helpers.solvers_and_matheuristics.pricing.solver import PricingStatus

    dual_info = master.get_reduced_cost_coefficients()
    node_duals: Dict[int, float] = dual_info.get("node_duals", {})
    rcc_duals: Dict = dual_info.get("rcc_duals", {})
    sri_duals: Dict = dual_info.get("sri_duals", {})

    forced_nodes: Set[int] = set()
    rf_conflicts: Dict = {}
    forbidden_arcs: FrozenSet = frozenset()
    required_successors: Dict = {}
    if branching_constraints:
        for bc in branching_constraints:
            if isinstance(bc, NodeVisitationBranchingConstraint) and bc.forced:
                forced_nodes.add(bc.node)
            elif isinstance(bc, RyanFosterBranchingConstraint):
                if not bc.together:
                    # Ryan-Foster "separate" branches imply separation conflicts
                    # (per branching/constraints.py docs and the RCSPP
                    # preprocessing of branching constraints in pricing/solver.py).
                    # NOTE: latent today — the kwarg is stored on the solver
                    # (pricing/solver.py) but never fed into the label-correcting
                    # algorithm, so this sign fix changes no active behaviour.
                    rf_conflicts.setdefault(bc.node_r, set()).add(bc.node_s)
                    rf_conflicts.setdefault(bc.node_s, set()).add(bc.node_r)
            elif isinstance(bc, EdgeBranchingConstraint):
                if not bc.must_use:
                    forbidden_arcs = forbidden_arcs | frozenset([(bc.u, bc.v)])
                else:
                    required_successors[bc.u] = bc.v

    # Pricing must see every dual the master produces (vehicle limit, edge-clique,
    # LCI, multistar, ...) and the branching constraints of this node. Without the
    # constraints it regenerates columns the node forbids; the master rejects them as
    # duplicates and CG stops early with a wrong node bound.
    dual_dict: Dict[str, Any] = {
        **dual_info,
        "node_duals": node_duals,
        "rcc_duals": rcc_duals,
        "sri_duals": sri_duals,
    }
    routes = pricing_solver.solve(
        dual_values=dual_dict,
        max_routes=max_routes,
        branching_constraints=branching_constraints,
        forced_nodes=forced_nodes,
        rf_conflicts=cast(Optional[Dict[int, Set[int]]], rf_conflicts),
        exact_mode=exact_mode,
        timeout=timeout,
    )

    added = 0
    for route in routes:
        if cast(float, route.reduced_cost) > rc_tolerance:  # noqa: SIM102
            if master.add_route(route):
                added += 1

    if strict_certification:
        # Explicit pricing certification: only an EXHAUSTIVE solve that didn't
        # time out with maximum reduced cost <= rc_tolerance can certify
        # exhaustion (MS-BPC-SP semantics).
        pricer_status = getattr(pricing_solver, "pricing_status", PricingStatus.PARTIAL)
        pricer_timed_out = getattr(pricing_solver, "_timed_out", False)
        last_max_rc = getattr(pricing_solver, "last_max_rc", -float("inf"))
        pricing_exhausted = (
            pricer_status == PricingStatus.EXHAUSTIVE
            and not pricer_timed_out
            and last_max_rc <= rc_tolerance
        )
    else:
        pricing_exhausted = getattr(pricing_solver, "last_max_rc", 0.0) <= optimality_gap
    return added, pricing_exhausted


def apply_reduced_cost_edge_fixing(
    master: VRPPMasterProblem,
    pricing_solver: Any,
    z_ub: float,
    z_lb: float,
) -> int:
    """LP-bound edge fixing on the master problem (original implementation).

    Uses exact DP completion bounds stored in ``pricing_solver.bounds_from``
    / ``pricing_solver.bounds_to`` to eliminate arcs from the pricing graph.
    This is the full, non-conservative variant that uses the Lagrangian
    completion-bound values.

    Args:
        master:         Master problem (provides dual values).
        pricing_solver: RCSPP solver (provides completion bounds).
        z_ub:           Upper bound (best known integer solution value).
        z_lb:           Lower bound (current LP relaxation value).

    Returns:
        Number of edges fixed to zero.
    """
    gap = z_ub - z_lb
    if gap <= 0:
        return 0

    dual_values = master.get_reduced_cost_coefficients()
    node_duals: Dict[int, float] = dual_values.get("node_duals", {})

    fixed_count = 0
    n = pricing_solver.n_nodes + 1
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cost = pricing_solver.cost_matrix[i, j] * pricing_solver.C
            rev = pricing_solver.wastes.get(j, 0.0) * pricing_solver.R
            rc_ij = rev - cost - node_duals.get(j, 0.0)
            max_path_rc = pricing_solver.bounds_from[i] + rc_ij + pricing_solver.bounds_to[j]
            if z_ub + max_path_rc < z_lb - 1e-6 and (i, j) not in pricing_solver.fixed_arcs:
                pricing_solver.fixed_arcs.add((i, j))
                fixed_count += 1

    if fixed_count > 0:
        logger.info("[ArcFix] Fixed %d edges (gap=%.4f).", fixed_count, gap)
    return fixed_count
