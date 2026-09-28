"""Column Generation loop for Branch-and-Price-and-Cut solvers.

Provides the core column generation engine — Phase I (Farkas) → Phase II
(profit-maximising) → cut separation loop — as a reusable function that
can be embedded in any BPC orchestrator or matheuristic pipeline.

Attributes:
-----------
column_generation_loop  — Full CG loop: Farkas → Phase II → cuts → convergence.

Example:
    None

References
----------
Barnhart, Hane & Vance (2000) Operations Research 48(2):318-326 §3.
Jepsen et al. (2008) — SRI cut integration.
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

from gurobipy import GRB

from logic.src.policies.helpers.solvers_and_matheuristics.branching import AnyBranchingConstraint
from logic.src.policies.helpers.solvers_and_matheuristics.branching.pruning import (
    BPCPruningException,
)
from logic.src.policies.helpers.solvers_and_matheuristics.branching.pruning import (
    apply_branching_to_master as _apply_branching_to_master,
)
from logic.src.policies.helpers.solvers_and_matheuristics.master_problem import VRPPMasterProblem
from logic.src.policies.helpers.solvers_and_matheuristics.pricing import RCSPPSolver
from logic.src.policies.helpers.solvers_and_matheuristics.pricing.smoothing import (
    detect_cycles as _detect_cycles,
)
from logic.src.policies.helpers.solvers_and_matheuristics.pricing.smoothing import (
    separate_cuts,
    solve_farkas_pricing_step,
    solve_pricing_step,
)
from logic.src.policies.helpers.solvers_and_matheuristics.search.cutting_planes import CuttingPlaneEngine

logger = logging.getLogger(__name__)


def column_generation_loop(  # noqa: C901
    master: VRPPMasterProblem,
    pricing_solver: RCSPPSolver,
    cut_engine: CuttingPlaneEngine,
    branching_constraints: Optional[List[AnyBranchingConstraint]],
    max_cg_iterations: int,
    max_cuts: int,
    time_limit: float,
    start_time: float,
    max_routes_per_pricing: int = 5,
    vehicle_limit: Optional[int] = None,
    optimality_gap: float = 1e-4,
    early_termination_gap: float = 1e-3,
    parent_basis: Optional[Any] = None,
    incumbent_value: float = -float("inf"),
    node_depth: int = 0,
    rc_tolerance: float = 1e-5,
    cut_orthogonality_threshold: float = 0.8,
    exact_mode: bool = False,
    cg_at_root_only: bool = False,
    branching_strategy: str = "divergence",
    rcspp_timeout: float = 30.0,
) -> Tuple[float, Dict[int, float], Optional[Any], bool]:
    """Run the Column Generation and Cutting Plane loop at a B&B node.

    Adopts the Barnhart et al. (1998) protocol for sequencing Cut Separation
    after CG convergence. Maintains the ng-route relaxation state and
    enforces strict elementarity for nodes in active SRI or node-pair
    branching constraints.

    When ``cg_at_root_only=True`` and ``node_depth > 0``, the pricing
    subproblem is not called: the LP is re-solved using only the columns
    already in the pool (sifted from the global pool at the root).  This
    replicates the "CG at root only" experiment from Table 2 of Barnhart,
    Hane, and Vance (2000), which trades bound quality for solve speed.

    Args:
        master (VRPPMasterProblem): Master problem instance.
        pricing_solver (RCSPPSolver): RCSPP pricing solver.
        cut_engine (CuttingPlaneEngine): Cutting plane separation engine.
        branching_constraints (Optional[List[AnyBranchingConstraint]]): Active constraints.
        max_cg_iterations (int): Maximum number of iterations per node.
        max_cuts (int): Maximum number of cuts per iteration.
        time_limit (Optional[float]): Time budget in seconds.
        start_time (float): Start timestamp of the overall solve.
        max_routes_per_pricing (int): Maximum number of columns to return.
        vehicle_limit (Optional[int]): Fleet size limit.
        optimality_gap (float): Target optimality gap.
        early_termination_gap (float): Heuristic stopping threshold.
        parent_basis (Optional[Any]): Gurobi basis from parent node.
        incumbent_value (float): Best known integer solution value.
        node_depth (int): Current B&B tree depth.
        rc_tolerance (float): Minimum reduced cost for column acceptance.
        cut_orthogonality_threshold (float): Minimum cut independence.
        exact_mode (bool): If True, disables dual smoothing.
        cg_at_root_only (bool): If True, skips pricing at descendant nodes.
        branching_strategy (str): Active branching rule name.
        rcspp_timeout (float): Timeout for the RCSPP solver.

    Returns:
        Tuple[float, Dict[int, float], Optional[Any], bool]:
            (Objective value, Route values λ_k, LP basis, Timed out flag).
    """
    timed_out = False
    converged = False
    pricing_exhausted = False
    obj_val = -float("inf")
    route_vals: Dict[int, float] = {}
    prev_obj_val = -float("inf")
    _iteration = 0
    consecutive_pricing_timeouts = 0
    max_consecutive_pricing_timeouts = 3
    # Task 3: Fix Lagrangian default.
    # If no fleet limit is active, the worst-case number of vehicles is n_nodes.
    fleet_size: int = vehicle_limit if vehicle_limit is not None else master.n_nodes

    # Task 5: Arc Conflict Pre-check (Exactness Rule)
    # We must enforce strict elementarity for nodes involved in active SRIs or
    # Ryan-Foster node-pair branching to prevent the ng-relaxation from
    # bypassing these constraints via cycles.
    elementary_nodes: Set[int] = set()

    # Extract nodes from Ryan-Foster branches
    from logic.src.policies.helpers.solvers_and_matheuristics.branching import (
        NodeVisitationBranchingConstraint,
        RyanFosterBranchingConstraint,
    )

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

    if elementary_nodes:
        added_elem = pricing_solver.enforce_elementarity(list(elementary_nodes))
        if added_elem > 0:
            logger.info(
                f"[Theoretical Hardening] Enforcing strict elementarity for "
                f"{len(elementary_nodes)} nodes (SRI/Ryan-Foster parity)."
            )

    # Task 5: Arc Conflict Pre-check (Exactness Rule)
    # Scan branching constraints for conflicting must_use arcs.
    # At most one must_use arc can exit node u, and at most one can enter node v.
    from logic.src.policies.helpers.solvers_and_matheuristics.branching import EdgeBranchingConstraint

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

    for _iteration in range(max_cg_iterations):
        if time_limit is not None and time_limit > 0 and (time.perf_counter() - start_time) > time_limit:
            timed_out = True
            break

        # PHASE 1: Column Generation (price until convergence)
        _inner_iter = 0
        while True:
            if time_limit is not None and time_limit > 0 and (time.perf_counter() - start_time) > time_limit:
                timed_out = True
                break

            # Basis restoration relocated after structural master updates
            # (branching, constraints, sifting) but before the core LP solve.
            if _inner_iter == 0 and isinstance(parent_basis, tuple):
                master.restore_basis(*parent_basis)

            # Only sift the pool after at least one LP solve has occurred at this node,
            # so the duals reflect the current node's constraint set.
            if _inner_iter > 0 and hasattr(master, "sift_global_column_pool"):
                duals = master.get_reduced_cost_coefficients()
                if duals:
                    # Pass all composite duals including LCI to sift_global_column_pool.
                    # This ensures columns re-activated from the global pool are evaluated
                    # against the full dual signal, including LCI penalties.
                    reactivated = master.sift_global_column_pool(
                        node_duals=duals["node_duals"],
                        rcc_duals=duals["rcc_duals"],
                        sri_duals=duals["sri_duals"],
                        edge_clique_duals=duals["edge_clique_duals"],
                        lci_duals=duals.get("lci_duals", {}),
                        lci_node_alphas=duals.get("lci_node_alphas", {}),
                        branching_constraints=branching_constraints,
                    )
                    if reactivated > 0:
                        # Re-apply branching filters to any newly activated columns.
                        # Sifting sets UB=1.0 for all positive-RC columns regardless of branching,
                        # so we must re-enforce the branching constraints immediately.
                        _apply_branching_to_master(
                            master, branching_constraints or [], branching_strategy=branching_strategy
                        )
                        # After the UB-reset inside _apply_branching_to_master, re-disable any
                        # non-elementary (cyclic) routes that cycle detection had suppressed.
                        # Without this, the sifting→reset→LP cycle keeps re-selecting the same
                        # cyclic routes, causing an infinite loop within one day's BPC solve.
                        for _nr, _nv in zip(master.routes, master.lambda_vars, strict=False):
                            if _nv.UB > 0.5 and len(set(_nr.nodes)) != len(_nr.nodes):
                                _nv.UB = 0.0
                        _inner_iter += 1
                        continue

            try:
                # Bound the LP solve to remaining wall-clock time so a single degenerate
                # master LP cannot blow past the node's overall time budget. Python-side
                # time checks only fire between Gurobi calls, so a single optimize() call
                # with no TimeLimit can overshoot the global limit on degenerate instances.
                if time_limit is not None and time_limit > 0 and master.model is not None:
                    _remaining_t = max(0.1, time_limit - (time.perf_counter() - start_time))
                    master.model.Params.TimeLimit = _remaining_t
                obj_val, route_vals = master.solve_lp_relaxation()  # type: ignore[assignment]
                if obj_val is None:
                    raise RuntimeError("LP relaxation returned non-optimal status at B&B node")

                if master.model is None or master.model.Status == GRB.INFEASIBLE:
                    logger.info("RMP node is infeasible. Starting Phase I Farkas Pricing.")
                    farkas = getattr(master, "farkas_duals", None)
                    if not farkas:
                        raise RuntimeError(
                            "LP is infeasible but Farkas duals are unavailable. "
                            "Gurobi may have returned INF_OR_UNBD — check model bounds."
                        )
                    if time_limit and time_limit > 0:
                        _elapsed_in_node = time.perf_counter() - start_time
                        _rem_t = max(0.1, time_limit - _elapsed_in_node)
                        _rem_t = min(_rem_t, 5.0)  # hard cap: no single RCSPP call exceeds 5s
                    else:
                        _rem_t = 5.0
                    added, _ = solve_farkas_pricing_step(
                        master,
                        pricing_solver,
                        branching_constraints,  # type: ignore[arg-type]
                        timeout=_rem_t,
                        farkas_duals=farkas,
                    )
                    # If the pricer timed out, it cannot certify that no improving
                    # column exists. Track consecutive timeouts and bail out of CG
                    # when they accumulate — spinning here wastes the entire budget.
                    pricer_timed_out = getattr(pricing_solver, "_timed_out", False)
                    if pricer_timed_out:
                        consecutive_pricing_timeouts += 1
                        if consecutive_pricing_timeouts >= max_consecutive_pricing_timeouts:
                            logger.warning(
                                f"CG aborting: {consecutive_pricing_timeouts} consecutive "
                                "RCSPP timeouts. Budget too tight for exact pricing at this node."
                            )
                            timed_out = True
                            break
                    else:
                        consecutive_pricing_timeouts = 0
                    if added == 0:
                        raise RuntimeError("LP infeasible at B&B node - Farkas pricing failed to find columns")
                    _inner_iter += 1
                    continue

                # Fix 1: Phase I → Phase II transition (Barnhart et al. 2000, §3).
                # After Farkas pricing restores feasibility (obj_val becomes finite),
                # switch route objectives from 0.0 (Phase I) back to profit (Phase II)
                # so the LP correctly maximises route profits going forward.
                if master.phase == 1 and master.model is not None and master.model.Status == GRB.OPTIMAL:
                    master.set_phase(2)
                    logger.info(
                        "Phase I complete — LP feasibility restored (GRB.OPTIMAL). "
                        "Switching to Phase II (profit maximization)."
                    )
                    # The duals in hand are Phase I duals (all objectives were 0), so
                    # pricing with them finds nothing and would end CG at the warm-start
                    # columns. Re-solve the LP under the Phase II objective first.
                    _inner_iter += 1
                    continue

            except Exception as e:
                if "Farkas pricing failed" in str(e):
                    raise
                raise RuntimeError("LP relaxation failed at B&B node") from e

            # Paper Table 2: "CG at root only" mode — skip pricing at non-root nodes.
            # The global column pool (built during root CG) is already sifted above;
            # no new columns are generated, so we converge immediately.
            if cg_at_root_only and node_depth > 0:
                converged = True
                break

            _elapsed = time.perf_counter() - start_time
            _remaining = max(0.1, time_limit - _elapsed) if time_limit is not None and time_limit > 0 else None
            _rem_t = (
                min(_remaining, time_limit * 0.40)
                if _remaining is not None and time_limit is not None and time_limit > 0 and node_depth == 0
                else _remaining
            )
            added, pricing_exhausted = solve_pricing_step(
                master,
                pricing_solver,
                branching_constraints,
                max_routes=max(max_routes_per_pricing, 50),  # return more routes per call
                optimality_gap=optimality_gap,
                rc_tolerance=rc_tolerance,
                timeout=_rem_t,
                exact_mode=exact_mode,
            )
            # Consecutive RCSPP timeout detection.
            # A timed-out pricer returns partial results — it cannot certify
            # that no improving column exists. Spinning here burns the entire
            # budget without converging. Abort CG after N consecutive timeouts.
            if getattr(pricing_solver, "_timed_out", False):
                consecutive_pricing_timeouts += 1
                if consecutive_pricing_timeouts >= max_consecutive_pricing_timeouts:
                    logger.warning(
                        f"CG aborting at depth {node_depth}: "
                        f"{consecutive_pricing_timeouts} consecutive RCSPP timeouts. "
                        "Treating best available LP solution as node result."
                    )
                    timed_out = True
                    break
            else:
                consecutive_pricing_timeouts = 0

            if added == 0 and not exact_mode:
                # Heuristic pricing (truncated neighbourhoods) cannot prove that no
                # improving column exists; confirm with one exact pass before converging.
                added, pricing_exhausted = solve_pricing_step(
                    master,
                    pricing_solver,
                    branching_constraints,
                    max_routes=max(max_routes_per_pricing, 50),
                    optimality_gap=optimality_gap,
                    rc_tolerance=rc_tolerance,
                    timeout=_rem_t,
                    exact_mode=True,
                )

            if added == 0:
                # Task 1b: Check for fractional cycles in ng-relaxation if CG has converged
                # locally. Dynamic ng-expansion serves as a lightweight cut separation.
                cycles: List[Tuple[int, ...]] = []
                for idx, val in route_vals.items():
                    if val > 0.1:
                        route = master.routes[idx]
                        route_cycles = _detect_cycles(route.nodes)
                        if route_cycles:
                            cycles.extend(route_cycles)

                if cycles:
                    added_ng = pricing_solver.expand_ng_neighborhoods(cycles)

                    # Task 1: Physically disable the cyclic columns in the Master Problem.
                    # Just expanding ng-sets is insufficient if the cyclic column remains
                    # active and dominating in the current RMP.
                    for idx, val in route_vals.items():
                        if val > 1e-4:
                            route = master.routes[idx]
                            if len(set(route.nodes)) != len(route.nodes):
                                master.lambda_vars[idx].UB = 0.0

                    if added_ng > 0:
                        logger.info(
                            f"[Dynamic ng-Expansion] Added {added_ng} neighborhood pairs "
                            f"to suppress {len(cycles)} cycles in iteration {_iteration}."
                        )
                    else:
                        logger.info(
                            f"[Cycle Suppression] Manually disabled {len(cycles)} cyclic routes "
                            f"in iteration {_iteration}."
                        )
                    continue  # Re-run LP and pricing with tightened relaxation

                # Truly converged for this cut-iteration
                # Perform a final LP solve to ensure the primal solution is consistent
                # if cycle suppression or purging just happened.
                master.solve_lp_relaxation()
                converged = True
                break

            # Task 2b: Periodic Column Purging
            if _inner_iter > 0 and _inner_iter % 20 == 0:
                purged = master.purge_useless_columns(tolerance=-0.1)
                if purged > 0:
                    logger.info(f"Periodic purge removed {purged} useless columns.")

            _inner_iter += 1

        # Fix 3: Lagrangian UB check — only valid AFTER CG convergence.
        # Per Barnhart et al. (2000) §4, z_UB = z_LP + K * max_rc is a valid
        # upper bound only when all pricing subproblems have been exhausted
        # (no more positive reduced cost columns). Computing it mid-CG yields
        # an invalid, overly tight bound that can prune optimal nodes.
        # Lagrangian UB is only valid when CG has fully converged: i.e., the pricing
        # subproblem confirmed that NO column with positive reduced cost exists.
        # If we compute it while positive-RC columns may still exist (e.g., the batch
        # was capped by max_routes_per_pricing), last_max_rc underestimates the true
        # max RC, producing an UB that is tighter than valid and can prune optimal nodes.
        if (
            converged
            and pricing_exhausted
            and not timed_out
            and hasattr(pricing_solver, "last_max_rc")
        ):
            max_rc = getattr(pricing_solver, "last_max_rc", -float("inf"))
            effective_fleet = fleet_size if fleet_size > 0 else master.n_nodes
            lagrangian_ub = obj_val + (effective_fleet * max(0.0, max_rc))

            logger.info(f"CG Iter {_iteration}: RMP={obj_val:.4f}, max_rc={max_rc:.6f}, z_UB={lagrangian_ub:.4f}")

            if lagrangian_ub < incumbent_value - 1e-6:
                logger.info(f"Exact Pruning: z_UB {lagrangian_ub:.4f} < Incumbent {incumbent_value:.4f}")
                raise BPCPruningException(f"Node pruned by exact Lagrangian bound: {lagrangian_ub}")

            # Task 11: Heuristic Early Termination
            # Allows stopping column generation early if the Lagrangian gap is
            # sufficiently small, even if positive reduced cost columns exist.
            if incumbent_value > -float("inf"):
                gap_to_incumbent = (lagrangian_ub - incumbent_value) / max(1.0, abs(incumbent_value))
                if gap_to_incumbent <= early_termination_gap:
                    logger.warning(
                        "Heuristic early termination: bound gap %.6f <= threshold %.2e. "
                        "Solution optimality is NOT guaranteed.",
                        gap_to_incumbent,
                        early_termination_gap,
                    )
                    converged = True
                    break

        if timed_out:
            break

        # PHASE 2: Cutting Planes — Task 11: only separate when LP is optimal
        cuts_added = separate_cuts(
            master,
            cut_engine,
            max_cuts,
            iteration=_iteration,
            node_depth=node_depth,
            cut_orthogonality_threshold=cut_orthogonality_threshold,
        )

        if cuts_added == 0:
            converged = True
            break

        # Fix 4 / Task 10 (SOTA): Tail-Off Management with relative tolerance.
        # Using absolute delta is fragile for problems with large objective values.
        # Relative delta normalises by objective magnitude for consistent behaviour.
        obj_delta = abs(obj_val - prev_obj_val) if _iteration > 0 else float("inf")
        denominator = max(1.0, abs(obj_val))
        relative_delta = obj_delta / denominator
        if relative_delta < 1e-5:
            logger.info(
                f"Tail-off detected: relative delta {relative_delta:.8f} < 1e-5 "
                f"(abs delta={obj_delta:.6f}, |obj|={denominator:.4f}). Stopping separation."
            )
            converged = True
            break
        prev_obj_val = obj_val

    # One final LP solve to get consistent obj_val / route_vals after any cuts/columns
    obj_val, route_vals = master.solve_lp_relaxation()  # type: ignore[assignment]
    if obj_val is None:
        # Non-optimal final LP (typically the time limit): no valid bound for this
        # node. Report it as timed out so the engine stops and returns its best plan.
        timed_out = True
        obj_val = -float("inf")

    # Warn only when the iteration cap truncated an unconverged loop
    if not converged and not timed_out and _iteration == max_cg_iterations - 1:
        warnings.warn(
            f"CG+Cut loop hit max_cg_iterations={max_cg_iterations} without full convergence.",
            stacklevel=3,
        )

    final_basis = master.save_basis()
    return obj_val, route_vals, final_basis, timed_out
