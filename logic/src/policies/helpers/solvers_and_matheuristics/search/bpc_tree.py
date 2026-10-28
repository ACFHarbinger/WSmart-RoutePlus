"""Shared Branch-and-Bound tree search for the BPC family (WP-11, P-33).

Released BPC and unreleased MS-BPC-SP share one tree walk. The engines keep
their own setup (MS: knapsack pre-selection and index remap; BPC: full
instance plus Rank-1 supplemental cuts) and their own endings (BPC:
incumbent or restricted-master IP fallback; MS: remap, Phase 3 regret-2,
Phase 5 set-partitioning, empty-result retries). Certificates stay on each
engine's own config defaults — this module does not coerce
``MSBPCSPConfig`` into ``BPCConfig``.

Attributes:
-----------
BPCSearchOptions  — per-engine flags that change node certification, not
                    the walk itself.
search_bpc_tree   — B&B loop: LR pre-pruning → CG → integer/bound/branch.
total_travel_cost — C × depot-wrapped route distance (shared accounting).

Example:
    >>> nodes_explored = search_bpc_tree(
    ...     master=master,
    ...     pricing_solver=pricing_solver,
    ...     cut_engine=cut_engine,
    ...     bb_tree=bb_tree,
    ...     cfg=cfg,
    ...     branching_strategy_name=cfg.branching_strategy,
    ...     time_limit=cfg.time_limit,
    ...     start_time=start_time,
    ...     red_dist=red_dist,
    ...     red_wastes=red_wastes,
    ...     capacity=capacity,
    ...     R=R,
    ...     C=C,
    ...     m_set=m_set,
    ...     env=env,
    ...     recorder=recorder,
    ...     k_nodes=k_nodes,
    ...     options=BPCSearchOptions(),
    ... )

References
----------
Barnhart, Hane & Vance (2000) Operations Research 48(2):318-326.

Parts are relocated to bpc_options.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Set

import gurobipy as gp
import numpy as np

from logic.src.policies.helpers.solvers_and_matheuristics.branching import BranchAndBoundTree
from logic.src.policies.helpers.solvers_and_matheuristics.branching.pruning import (
    BPCPruningException,
)
from logic.src.policies.helpers.solvers_and_matheuristics.branching.pruning import (
    apply_branching_to_master as _apply_branching_to_master,
)
from logic.src.policies.helpers.solvers_and_matheuristics.branching.pruning import (
    extract_forced_sets_from_constraints as _extract_forced_sets_from_constraints,
)
from logic.src.policies.helpers.solvers_and_matheuristics.branching.pruning import (
    perform_strong_branching as _perform_strong_branching,
)
from logic.src.policies.helpers.solvers_and_matheuristics.exact_trace import (
    emit_branch,
    emit_incumbent,
    emit_node,
    emit_node_bound,
    selected_route_nodes,
)
from logic.src.policies.helpers.solvers_and_matheuristics.exact_trace import (
    emit_column as _emit_column,
)
from logic.src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.pre_pruning import (
    compute_lr_bound_at_node as _compute_lr_bound_at_node,
)
from logic.src.policies.helpers.solvers_and_matheuristics.master_problem import VRPPMasterProblem
from logic.src.policies.helpers.solvers_and_matheuristics.pricing import RCSPPSolver
from logic.src.policies.helpers.solvers_and_matheuristics.pricing.smoothing import (
    apply_reduced_cost_edge_fixing as _apply_reduced_cost_edge_fixing,
)
from logic.src.policies.helpers.solvers_and_matheuristics.pricing.smoothing import (
    is_solution_integer as _is_solution_integer,
)
from logic.src.policies.helpers.solvers_and_matheuristics.search.column_generation import (
    column_generation_loop as _column_generation_loop,
)
from logic.src.policies.helpers.solvers_and_matheuristics.search.cutting_planes import CuttingPlaneEngine

from .bpc_options import BPCSearchOptions as BPCSearchOptions

logger = logging.getLogger(__name__)


def search_bpc_tree(  # noqa: C901
    *,
    master: VRPPMasterProblem,
    pricing_solver: RCSPPSolver,
    cut_engine: CuttingPlaneEngine,
    bb_tree: BranchAndBoundTree,
    cfg: Any,
    branching_strategy_name: str,
    time_limit: float,
    start_time: float,
    red_dist: np.ndarray,
    red_wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    m_set: Set[int],
    env: Optional[Any],
    recorder: Optional[Any],
    k_nodes: int,
    options: Optional[BPCSearchOptions] = None,
) -> int:
    """Run the shared B&B walk. Mutates ``bb_tree`` and ``master``.

    Args:
        master: Built restricted master (Phase I/II already set).
        pricing_solver: RCSPP pricer on the (possibly reduced) instance.
        cut_engine: Active cutting-plane engine for this solve.
        bb_tree: Tree with the root already inserted.
        cfg: Engine config (``BPCConfig`` or ``MSBPCSPConfig``). Fields are
            read by name; missing optional fields use ``getattr`` defaults.
        branching_strategy_name: Primary branching rule name.
        time_limit: Global wall-clock budget (0 = unlimited).
        start_time: ``time.perf_counter()`` at engine entry.
        red_dist: Distance matrix for the priced instance.
        red_wastes: Waste map for the priced instance.
        capacity: Vehicle capacity.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        m_set: Forced-in nodes for LR pre-pruning (caller's index space).
        env: Optional Gurobi environment.
        recorder: Optional telemetry recorder.
        k_nodes: Number of customer nodes on the priced instance.
        options: Per-engine certification flags.

    Returns:
        Number of B&B nodes explored.
    """
    if options is None:
        options = BPCSearchOptions()

    max_cg_iter = cfg.max_cg_iterations
    max_cuts = cfg.max_cuts_per_iteration
    max_routes_per_pricing = cfg.max_routes_per_pricing
    max_bb_nodes = cfg.max_bb_nodes
    nodes_explored = 0

    while not bb_tree.is_empty() and nodes_explored < max_bb_nodes:
        if time_limit > 0 and (time.perf_counter() - start_time) > time_limit:
            break

        current_node = bb_tree.get_next_node()
        if current_node is None:
            break

        nodes_explored += 1
        bb_tree.record_explored()
        if getattr(master, "_exec_trace", None) is not None:
            emit_node(master, current_node.depth, nodes_explored)

        master.remove_local_cuts()
        branching_constraints = current_node.get_all_constraints()
        _apply_branching_to_master(master, branching_constraints, branching_strategy_name)
        if master.model:
            master.model.update()

        _lr_ok = (
            cfg.lr_pre_pruning
            and bb_tree.best_integer_solution is not None
            and (cfg.lr_pre_pruning_depth_limit < 0 or current_node.depth <= cfg.lr_pre_pruning_depth_limit)
        )
        if options.lr_requires_vehicle_limit:
            _lr_ok = _lr_ok and master.vehicle_limit is not None

        if _lr_ok:
            _lr_forced_in, _lr_forced_out = _extract_forced_sets_from_constraints(current_node.constraints)
            _lr_forced_in |= m_set

            _remaining = time_limit - (time.perf_counter() - start_time) if time_limit > 0 else 300.0
            _lr_budget = min(
                _remaining * 0.05,
                cfg.lr_op_time_limit * cfg.lr_max_subgradient_iters,
            )

            if _lr_budget > 0.5:
                _lr_ub, _lam_star, _lr_op_visited = _compute_lr_bound_at_node(
                    dist_matrix=red_dist,
                    wastes=red_wastes,
                    capacity=capacity,
                    R=R,
                    C=C,
                    mandatory=_lr_forced_in,
                    forced_out=_lr_forced_out,
                    params=cfg,
                    time_budget=_lr_budget,
                    env=env,
                    recorder=recorder,
                )

                if options.scale_lr_by_vehicle_limit and master.vehicle_limit is not None:
                    _lr_ub = _lr_ub * master.vehicle_limit
                _gap_threshold = bb_tree.best_integer_solution * (1.0 + cfg.optimality_gap)

                if _lr_ub <= _gap_threshold:
                    logger.debug(
                        f"[LR Pre-Pruning] Node depth={current_node.depth} pruned. "
                        f"LR bound={_lr_ub:.4f} <= incumbent={bb_tree.best_integer_solution:.4f}."
                    )
                    if recorder:
                        recorder.record(
                            engine="bpc_lr_pre_pruning",
                            node_depth=current_node.depth,
                            lr_upper_bound=_lr_ub,
                            lam_star=_lam_star,
                            incumbent=bb_tree.best_integer_solution,
                            pruned=True,
                        )
                    bb_tree.prune_by_bound()
                    continue

                if recorder:
                    recorder.record(
                        engine="bpc_lr_pre_pruning",
                        node_depth=current_node.depth,
                        lr_upper_bound=_lr_ub,
                        lam_star=_lam_star,
                        incumbent=bb_tree.best_integer_solution,
                        pruned=False,
                    )

                if cfg.lr_warm_start_cg and _lr_op_visited:
                    _seed_nodes = sorted(_lr_op_visited - _lr_forced_out)
                    if _seed_nodes:
                        try:
                            _seed_route = pricing_solver._compute_route_details(_seed_nodes)
                            _seed_rc = master.calculate_reduced_cost(
                                _seed_route,
                                master.get_reduced_cost_coefficients(),
                            )
                            if _seed_rc > cfg.rc_tolerance:
                                _seeded = master.add_route(_seed_route)
                                if _seeded and getattr(master, "_exec_trace", None) is not None:
                                    _emit_column(
                                        master,
                                        _seed_route.nodes,
                                        _seed_rc,
                                        "ordinary",
                                        int(getattr(master, "phase", 2)),
                                    )
                                logger.debug(
                                    f"[LR Warm-Start] Seeded column from UOP at λ*={_lam_star:.4f}, "
                                    f"rc={_seed_rc:.4f}, nodes={_seed_nodes}."
                                )
                        except Exception as _e:
                            logger.debug(f"[LR Warm-Start] Column seeding failed: {_e}. Continuing.")

        try:
            ng_snapshot = pricing_solver.save_ng_snapshot()
            try:
                _node_start = time.perf_counter()
                _elapsed_global = _node_start - start_time
                _remaining_global = max(0.1, time_limit - _elapsed_global) if time_limit > 0 else 0.0
                if _remaining_global > 0.0 and current_node.depth == 0:
                    _node_budget = min(_remaining_global, time_limit * 0.40)
                else:
                    _node_budget = _remaining_global

                lp_obj, route_values, node_final_basis, node_timed_out = _column_generation_loop(
                    master=master,
                    pricing_solver=pricing_solver,
                    cut_engine=cut_engine,
                    branching_constraints=branching_constraints,
                    max_cg_iterations=max_cg_iter,
                    max_cuts=max_cuts,
                    time_limit=_node_budget,
                    start_time=_node_start,
                    max_routes_per_pricing=max_routes_per_pricing,
                    vehicle_limit=master.vehicle_limit,
                    optimality_gap=cfg.optimality_gap,
                    early_termination_gap=cfg.early_termination_gap,
                    parent_basis=current_node.parent.lp_basis if current_node.parent else None,
                    incumbent_value=bb_tree.best_integer_solution
                    if bb_tree.best_integer_solution is not None
                    else -float("inf"),
                    node_depth=current_node.depth,
                    rc_tolerance=cfg.rc_tolerance,
                    cut_orthogonality_threshold=cfg.cut_orthogonality_threshold,
                    exact_mode=cfg.exact_mode,
                    cg_at_root_only=cfg.cg_at_root_only,
                    branching_strategy=branching_strategy_name,
                    rcspp_timeout=cfg.rcspp_timeout,
                    strict_certification=options.strict_certification,
                )
            finally:
                pricing_solver.restore_ng_snapshot(ng_snapshot)

            if node_timed_out:
                if options.incomplete_node_ip:
                    logger.warning(
                        f"B&B node at depth {current_node.depth} timed out or incomplete. "
                        "Preserving feasible incumbent before terminating."
                    )
                    current_node.lp_bound = float("inf")
                    if _is_solution_integer(master.routes, route_values, require_elementary=options.require_elementary):
                        current_node.is_integer = True
                        current_node.ip_solution = lp_obj
                        current_node.route_values = route_values
                        if bb_tree.update_incumbent(current_node, lp_obj) and getattr(master, "_exec_trace", None) is not None:
                            emit_incumbent(
                                master._exec_trace,
                                master._exec_trace_node,
                                selected_route_nodes(master, route_values),
                                lp_obj,
                            )
                    else:
                        try:
                            _rem_ip = (
                                max(0.1, time_limit - (time.perf_counter() - start_time)) if time_limit > 0 else None
                            )
                            ip_obj, ip_routes = master.solve_ip(time_limit=_rem_ip)
                            if ip_obj > (
                                bb_tree.best_integer_solution
                                if bb_tree.best_integer_solution is not None
                                else -float("inf")
                            ):
                                selected_indices = {}
                                for r in ip_routes:
                                    try:
                                        idx = master.routes.index(r)
                                        selected_indices[idx] = 1.0
                                    except ValueError:
                                        pass
                                current_node.is_integer = True
                                current_node.ip_solution = ip_obj
                                current_node.route_values = selected_indices
                                if bb_tree.update_incumbent(current_node, ip_obj) and getattr(
                                    master, "_exec_trace", None
                                ) is not None:
                                    emit_incumbent(
                                        master._exec_trace,
                                        master._exec_trace_node,
                                        selected_route_nodes(master, selected_indices),
                                        ip_obj,
                                    )
                        except Exception as e:
                            logger.debug(f"IP solve at incomplete node failed: {e}")
                    break
                logger.warning(f"B&B node at depth {current_node.depth} timed out. Terminating search.")
                break
            current_node.lp_basis = node_final_basis
        except BPCPruningException:
            continue
        except gp.GurobiError as e:
            logger.warning(
                "GurobiError at B&B node depth=%d: %s. Marking node infeasible.",
                current_node.depth,
                e,
            )
            current_node.is_infeasible = True
            continue
        except RuntimeError as e:
            if "Conflicting must_use" in str(e):
                current_node.is_infeasible = True
                continue
            current_node.is_infeasible = True
            continue

        current_node.lp_bound = lp_obj
        current_node.route_values = route_values
        if getattr(master, "_exec_trace", None) is not None:
            emit_node_bound(master, current_node.depth, lp_obj, bb_tree.best_integer_solution, root=False)

        if bb_tree.best_integer_solution is not None:
            stats = bb_tree.get_statistics()
            best_open_lp_bound = stats.get("best_bound")
            if best_open_lp_bound is None:
                best_open_lp_bound = lp_obj

            global_upper_bound = max(best_open_lp_bound, lp_obj)
            denom = max(
                abs(bb_tree.best_integer_solution),
                abs(global_upper_bound),
                1.0,
            )
            global_gap = (global_upper_bound - bb_tree.best_integer_solution) / denom

            if global_gap <= cfg.optimality_gap:
                logger.info(f"Global optimality gap reached: {global_gap:.6f} <= {cfg.optimality_gap}")
                break

            _apply_reduced_cost_edge_fixing(
                master=master,
                pricing_solver=pricing_solver,
                z_ub=global_upper_bound,
                z_lb=bb_tree.best_integer_solution,
            )

        if _is_solution_integer(master.routes, route_values, require_elementary=options.require_elementary):
            current_node.is_integer = True
            current_node.ip_solution = lp_obj
            if bb_tree.update_incumbent(current_node, lp_obj) and getattr(master, "_exec_trace", None) is not None:
                emit_incumbent(
                    master._exec_trace,
                    master._exec_trace_node,
                    selected_route_nodes(master, route_values),
                    lp_obj,
                )
            bb_tree.prune_by_bound()
            continue

        if bb_tree.best_integer_solution is not None and lp_obj <= bb_tree.best_integer_solution:
            bb_tree.prune_by_bound()
            continue

        if getattr(cfg, "enable_column_pool_deduplication", False):
            purged = master.deduplicate_column_pool()
            if purged > 0:
                logger.info(f"Deduplicated {purged} equivalent routes from column pool.")

        strong_candidate = None
        if getattr(cfg, "enable_strong_branching_heuristic", False):
            candidates = bb_tree.find_strong_branching_candidates(master.routes, route_values, max_candidates=5)
            if candidates:
                strong_candidate = _perform_strong_branching(
                    master=master,
                    candidates=candidates,
                    current_node=current_node,
                    strong_branching_size=getattr(cfg, "strong_branching_size", 5),
                )

        optional_nodes = {i for i in range(1, k_nodes + 1) if i not in getattr(master, "mandatory_nodes", set())}
        if optional_nodes and getattr(cfg, "enable_node_visitation_branching", False):
            from logic.src.policies.helpers.solvers_and_matheuristics.branching.strategies import (
                HierarchicalStrongBranching,
            )

            _y_branch_result = HierarchicalStrongBranching.find_best_y_branching(
                routes=master.routes,
                route_values=route_values,
                optional_nodes=optional_nodes,
                master=master if getattr(cfg, "enable_strong_branching_heuristic", False) else None,
                phase1_size=getattr(cfg, "strong_branching_size", 5),
                phase2_size=max(2, getattr(cfg, "strong_branching_size", 5) // 2),
            )
            if _y_branch_result is not None:
                _y_node, _y_val = _y_branch_result
                left_child, right_child = HierarchicalStrongBranching.create_child_nodes(current_node, _y_node, _y_val)
                left_child.lp_bound = lp_obj
                right_child.lp_bound = lp_obj
                bb_tree.add_node(left_child)
                bb_tree.add_node(right_child)
                logger.debug(f"Node-visitation branch on y_{_y_node}={_y_val:.3f} at depth {current_node.depth}.")
                continue

        children = bb_tree.branch(
            current_node,
            master.routes,
            route_values,
            mandatory_nodes=master.mandatory_nodes,
            strong_candidate=strong_candidate,
        )

        if children is not None:
            left_child, right_child = children
            rf_used = getattr(left_child, "branching_rule", None) == "ryan_foster"
            if rf_used and not master.strict_set_partitioning:
                logger.warning(
                    "Ryan-Foster branching activated at a non-RF-primary node. "
                    "Hardening master to strict Set Partitioning for correctness."
                )
                master.strict_set_partitioning = True

        if children is None:
            logger.warning(
                f"Primary branching strategy '{branching_strategy_name}' returned no "
                "branching candidate at a fractional node. Falling back to edge branching."
            )
            from logic.src.policies.helpers.solvers_and_matheuristics.branching import EdgeBranching

            res = EdgeBranching.find_branching_arc(master.routes, route_values)
            if res is None:
                logger.warning("Edge branching fallback also found no candidate. Skipping node.")
                continue
            (u, v), flow = res
            left_child, right_child = EdgeBranching.create_child_nodes(current_node, u, v, flow)
            children = (left_child, right_child)

        left_child, right_child = children
        if getattr(master, "_exec_trace", None) is not None:
            emit_branch(master, left_child, right_child)
        use_paper_ordering = (
            getattr(cfg, "prefer_shorter_path_dfs", True)
            and branching_strategy_name == "divergence"
            and bb_tree.search_strategy not in ("best_first", "best-first")
        )

        if use_paper_ordering:
            bb_tree.add_node(left_child)
            bb_tree.add_node(right_child)
        else:
            hint_l = left_child.lp_bound if left_child.lp_bound is not None else -float("inf")
            hint_r = right_child.lp_bound if right_child.lp_bound is not None else -float("inf")

            if hint_l < hint_r:
                bb_tree.add_node(left_child)
                bb_tree.add_node(right_child)
            elif hint_r < hint_l:
                bb_tree.add_node(right_child)
                bb_tree.add_node(left_child)
            else:
                if len(left_child.constraints) <= len(right_child.constraints):
                    bb_tree.add_node(right_child)
                    bb_tree.add_node(left_child)
                else:
                    bb_tree.add_node(left_child)
                    bb_tree.add_node(right_child)

    return nodes_explored
