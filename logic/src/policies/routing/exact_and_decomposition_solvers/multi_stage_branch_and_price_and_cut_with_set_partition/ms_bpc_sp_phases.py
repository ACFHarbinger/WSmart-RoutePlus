"""Private stages of :func:`ms_bpc_sp_engine.run_ms_bpc_sp`.

The instance remap and the tree and cut-engine set-up before the tree; after it, the best tree solution, Phase 3
(regret-2 insertion), Phase 5 (set-partitioning merge) and the empty-result
fallbacks.

Attributes:
    logger (logging.Logger): Module-level logger.
"""

import logging
import time
from dataclasses import replace
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from logic.src.constants import MAX_CAPACITY_PERCENT
from logic.src.policies.helpers.solvers_and_matheuristics import (
    BranchAndBoundTree,
    SeparationEngine,
    create_cutting_plane_engine,
)
from logic.src.policies.helpers.solvers_and_matheuristics.search.bpc_tree import (
    total_travel_cost as _total_travel_cost,
)
from logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model import VRPPModel
from logic.src.policies.routing.base.sub_problem import SolveResult, SubProblem

logger = logging.getLogger(__name__)

# The empty-result relaxation retry (see run_ms_bpc_sp) is only applied on
# instances with at most this many customer nodes to bound the extra work.
# Instance size is not an optimality certificate.
_RELAX_RETRY_MAX_NODES = 8


def _regret2_insertion(
    final_routes: List[List[int]],
    final_cost: float,
    visited_orig: Set[int],
    phase3_budget: float,
    n_nodes: int,
    m_set: Set[int],
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    dist_matrix: np.ndarray,
    vehicle_limit: Optional[int],
) -> Tuple[List[List[int]], float]:
    """Phase 3: insert profitable unvisited nodes greedily by regret-2, within ``phase3_budget`` seconds.

    Near-overflow bins (fill >= MAX_CAPACITY_PERCENT) are prioritised; a new route
    is opened when the fleet limit allows. ``final_routes`` and ``visited_orig``
    are updated in place.

    Returns:
        Tuple[List[List[int]], float]: The routes and their profit.
    """
    remaining_unvisited = [i for i in range(1, n_nodes + 1) if i not in visited_orig and i not in m_set]
    # Initial sort: near-overflow first, then by revenue
    remaining_unvisited.sort(
        key=lambda i: (
            -(wastes.get(i, 0.0) / (capacity + 1e-9) >= MAX_CAPACITY_PERCENT),
            -wastes.get(i, 0.0) * R,
        )
    )

    p3_start = time.perf_counter()
    n_p3 = 0

    while remaining_unvisited and (time.perf_counter() - p3_start) < phase3_budget:
        best_regret = -float("inf")
        chosen_node, chosen_route_idx, chosen_pos, chosen_gain = -1, -1, -1, 0.0

        for node_orig in remaining_unvisited:
            waste_i = wastes.get(node_orig, 0.0)
            revenue_i = R * waste_i
            insertions: List[Tuple[float, int, int]] = []

            for r_idx, route in enumerate(final_routes):
                if sum(wastes.get(n, 0.0) for n in route) + waste_i > capacity + 1e-6:
                    continue
                full = [0] + route + [0]
                for pos in range(len(full) - 1):
                    u, v = full[pos], full[pos + 1]
                    gain = revenue_i - C * (dist_matrix[u, node_orig] + dist_matrix[node_orig, v] - dist_matrix[u, v])
                    insertions.append((gain, r_idx, pos))

            if vehicle_limit is None or len(final_routes) < vehicle_limit:
                # New route option (round-trip cost)
                new_route_gain = revenue_i - C * 2.0 * dist_matrix[0, node_orig]
                insertions.append((new_route_gain, -1, -1))
            insertions.sort(reverse=True)
            if not insertions:
                continue  # fleet limit reached and no route has spare capacity

            best_gain_i = insertions[0][0]
            if best_gain_i <= 0:
                continue  # not profitable at all

            regret_i = insertions[0][0] - insertions[1][0] if len(insertions) >= 2 else float("inf")
            # Overflow bins must be inserted regardless — force priority
            fill_pct = waste_i / (capacity + 1e-9)
            if fill_pct >= MAX_CAPACITY_PERCENT:
                regret_i += 1e6

            if regret_i > best_regret:
                best_regret = regret_i
                chosen_node = node_orig
                chosen_gain = best_gain_i
                chosen_route_idx = insertions[0][1]
                chosen_pos = insertions[0][2]

        if chosen_node == -1:
            break  # nothing profitable left

        # Apply insertion
        if chosen_route_idx == -1:
            final_routes.append([chosen_node])
        else:
            full = [0] + final_routes[chosen_route_idx] + [0]
            final_routes[chosen_route_idx] = full[1 : chosen_pos + 1] + [chosen_node] + full[chosen_pos + 1 : -1]
        final_cost += chosen_gain
        visited_orig.add(chosen_node)
        remaining_unvisited.remove(chosen_node)
        n_p3 += 1

    logger.info(f"[Phase 3] Regret-2 greedy added {n_p3} nodes ({len(final_routes)} routes, profit={final_cost:.2f}).")
    return final_routes, final_cost


def _set_partition_merge(
    master_routes: List,
    local_to_orig: Dict[int, int],
    final_routes: List[List[int]],
    final_cost: float,
    phase5_budget: float,
    n_nodes: int,
    m_set: Set[int],
    wastes: Dict[int, float],
    R: float,
    C: float,
    dist_matrix: np.ndarray,
    vehicle_limit: Optional[int],
) -> Tuple[List[List[int]], float]:
    """Phase 5: set partitioning over the BPC master's routes and the Phase 3 routes.

    The SP solution replaces ``final_routes`` only when it is more profitable; a
    solver failure keeps the Phase 3 solution.

    Returns:
        Tuple[List[List[int]], float]: The routes and their profit.
    """
    try:
        # Pool: all BPC master routes + Phase 3 routes
        pool: Dict[FrozenSet[int], Tuple[List[int], float]] = {}

        def _pool_route(nodes_orig: List[int]) -> None:
            if not nodes_orig:
                return
            full = [0] + nodes_orig + [0]
            cost_r = sum(dist_matrix[full[i], full[i + 1]] * C for i in range(len(full) - 1))
            profit_r = sum(wastes.get(n, 0.0) * R for n in nodes_orig) - cost_r
            key = frozenset(nodes_orig)
            if key not in pool or profit_r > pool[key][1]:
                pool[key] = (nodes_orig, profit_r)

        for r in master_routes:
            _pool_route([local_to_orig[n] for n in r.nodes])
        for route_orig in final_routes:
            _pool_route(route_orig)

        pool_list = list(pool.values())
        logger.info(f"[Phase 5] SP pool: {len(pool_list)} routes.")

        if len(pool_list) > 1:
            sp_m = gp.Model("sp_merge")
            sp_m.Params.OutputFlag = 0
            sp_m.Params.TimeLimit = min(phase5_budget, 15.0)
            sp_m.Params.MIPGap = 1e-4

            x_sp = sp_m.addVars(len(pool_list), vtype=GRB.BINARY, name="x")
            sp_m.setObjective(
                gp.quicksum(pool_list[k][1] * x_sp[k] for k in range(len(pool_list))),
                GRB.MAXIMIZE,
            )
            for node_orig in range(1, n_nodes + 1):
                covers = [k for k, (rn, _) in enumerate(pool_list) if node_orig in rn]
                if not covers:
                    continue
                expr = gp.quicksum(x_sp[k] for k in covers)
                if node_orig in m_set:
                    sp_m.addConstr(expr == 1)
                else:
                    sp_m.addConstr(expr <= 1)

            # Enforce the fleet limit: the SP re-optimization must not undo the
            # limit the BPC master enforced (pool routes can exceed it when
            # Phase 3 opened extra routes or the pool mixes bounded/unbounded columns).
            if vehicle_limit is not None:
                sp_m.addConstr(gp.quicksum(x_sp[k] for k in range(len(pool_list))) <= vehicle_limit, name="fleet_limit")

            sp_m.optimize()
            if sp_m.SolCount > 0:
                sp_routes = [pool_list[k][0] for k in range(len(pool_list)) if x_sp[k].X > 0.5]
                sp_profit = sum(pool_list[k][1] for k in range(len(pool_list)) if x_sp[k].X > 0.5)
                if sp_profit > final_cost:
                    final_routes = sp_routes
                    final_cost = sp_profit
                    logger.info(f"[Phase 5] SP improved profit to {final_cost:.2f} ({len(final_routes)} routes).")
    except Exception as _e5:
        logger.warning(f"[Phase 5] SP failed: {_e5}. Using Phase 3 solution.")
    return final_routes, final_cost


def _remap_instance(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    m_set: Set[int],
    selected_nodes: Set[int],
) -> Tuple[int, Dict[int, int], np.ndarray, Dict[int, float], Set[int]]:
    """Restrict the instance to the selected nodes, renumbered 1..k (the depot stays 0).

    Returns:
        Tuple: (k_nodes, local_to_orig, red_dist, red_wastes, red_mandatory).
    """
    # Build reduced problem: remap selected nodes to contiguous 1..k indices
    selected_list = sorted(selected_nodes)
    k_nodes = len(selected_list)

    # Maps: local index (1..k) ↔ original index (1..n)
    local_to_orig = {local: orig for local, orig in enumerate(selected_list, 1)}
    orig_to_local = {orig: local for local, orig in local_to_orig.items()}

    # Reduced distance matrix (k+1 × k+1), depot stays at index 0
    red_dist = np.zeros((k_nodes + 1, k_nodes + 1))
    for i_loc in range(k_nodes + 1):
        i_orig = local_to_orig.get(i_loc, 0)  # 0 → depot
        for j_loc in range(k_nodes + 1):
            j_orig = local_to_orig.get(j_loc, 0)
            red_dist[i_loc, j_orig if j_loc == 0 else orig_to_local.get(local_to_orig.get(j_loc, 0), 0)] = dist_matrix[
                i_orig, j_orig
            ]

    # Simpler: build reduced dist matrix directly
    orig_indices = [0] + selected_list  # depot + selected customers
    red_dist = dist_matrix[np.ix_(orig_indices, orig_indices)]

    red_wastes = {local: wastes.get(orig, 0.0) for local, orig in local_to_orig.items()}
    red_mandatory = {orig_to_local[i] for i in m_set if i in orig_to_local}
    return k_nodes, local_to_orig, red_dist, red_wastes, red_mandatory


def _best_bpc_solution(
    bb_tree: Any,
    master: Any,
    pricing_solver: Any,
    initial_routes_nodes: List[List[int]],
    local_to_orig: Dict[int, int],
) -> Tuple[List[List[int]], float]:
    """The tree's best integer solution in original node ids, or the greedy warm start if none.

    Returns:
        Tuple[List[List[int]], float]: (routes, profit).
    """
    if bb_tree.best_integer_node is None:
        # No integer solution — use initial greedy as BPC result
        bpc_cost = 0.0
        for r_nodes in initial_routes_nodes:
            bpc_cost += pricing_solver._compute_route_details(r_nodes).profit
        bpc_routes_orig = [[local_to_orig[n] for n in r] for r in initial_routes_nodes]
    else:
        # Reconstruct solution from best node
        best_node = bb_tree.best_integer_node
        best_routes_objects = []
        for idx, val in best_node.route_values.items():  # type: ignore[union-attr]
            if val > 0.5:
                best_routes_objects.append(master.routes[idx])
        bpc_routes_orig = [[local_to_orig[n] for n in r.nodes] for r in best_routes_objects]
        bpc_cost = sum(r.profit for r in best_routes_objects)
    return bpc_routes_orig, bpc_cost


def _empty_result_fallback(
    solve: Callable[..., SolveResult],
    problem: SubProblem,
    params: Any,
    start_time: float,
    time_limit: float,
    final_routes: List[List[int]],
    k_nodes: int,
    n_nodes: int,
    vehicle_limit: Optional[int],
    m_set: Set[int],
    dist_matrix: np.ndarray,
    C: float,
    presel_retry: bool,
    keep_all_nodes: bool,
) -> Optional[SolveResult]:
    """Re-solve a fleet-constrained empty result, or None to keep it.

    ``solve`` is ``run_ms_bpc_sp`` (passed in to avoid a circular import). The
    retries share the remaining finite budget of the original solve.
    """
    # Share the original finite budget across all fallback solves. Never pass
    # zero on expiry: the solver interprets zero as an unlimited time budget.
    retry_params = params
    if time_limit > 0:
        retry_remaining = time_limit - (time.perf_counter() - start_time)
        retry_params = replace(params, time_limit=retry_remaining) if retry_remaining > 0 else None

    # Retry only fleet-constrained empty results; preserve unlimited behavior.
    if (
        not final_routes
        and k_nodes < n_nodes
        and not presel_retry
        and vehicle_limit is not None
        and retry_params is not None
    ):
        logger.info(
            f"[Node Selection] Empty result after dropping {n_nodes - k_nodes} nodes; "
            "re-solving without the capacity budget."
        )
        return solve(
            problem=problem,
            cfg=retry_params,
            _presel_retry=True,
        )

    # Try a full-instance fleet relaxation when time remains. A returned
    # incumbent fitting the original fleet is a feasible fallback only: this
    # API returns no convergence/bound certificate, even on small instances.
    if (
        not final_routes
        and vehicle_limit is not None
        and not keep_all_nodes
        and retry_params is not None
        and n_nodes <= _RELAX_RETRY_MAX_NODES
    ):
        # Relaxation solves without the fleet limit (the incumbent is only
        # accepted above when it fits the original fleet), keeping every node.
        relaxed_problem = SubProblem(
            dist_matrix=problem.dist_matrix,
            wastes=problem.wastes,
            capacity=problem.capacity,
            revenue=problem.revenue,
            cost_unit=problem.cost_unit,
            mandatory_nodes=problem.mandatory_nodes,
            subset_indices=problem.subset_indices,
            x_coords=problem.x_coords,
            y_coords=problem.y_coords,
            values=problem.values,
            n_vehicles=None,
            env=problem.env,
            node_coords=problem.node_coords,
            recorder=problem.recorder,
            dual_values=problem.dual_values,
        )
        fallback = solve(
            problem=relaxed_problem,
            cfg=retry_params,
            _presel_retry=True,
            _keep_all_nodes=True,
        )
        if (
            fallback.routes
            and len(fallback.routes) <= vehicle_limit
            and m_set.issubset({node for route in fallback.routes for node in route})
        ):
            logger.info(
                f"[Node Selection] Unconstrained relaxation ({len(fallback.routes)} routes) "
                f"fits the fleet limit {vehicle_limit}; accepting it as a feasible fallback."
            )
            return SolveResult(
                routes=fallback.routes,
                profit=fallback.profit,
                cost=_total_travel_cost(fallback.routes, dist_matrix, C),
            )
        logger.info(
            "[Node Selection] Relaxation has no non-empty fleet-feasible candidate covering "
            "the mandatory nodes; keeping the fleet-constrained (empty) result."
        )
    return None


def _build_tree_and_cuts(
    k_nodes: int,
    red_dist: np.ndarray,
    red_wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    red_mandatory: Set[int],
    params: Any,
    cutting_planes_name: str,
) -> Tuple[Any, Any]:
    """Build the separation model, the B&B tree and the cutting-plane engine on the reduced instance.

    Returns:
        Tuple: (bb_tree, cut_engine).
    """
    # 3. Initialize pricing and separation — use REDUCED instance throughout
    # pricing_solver is already initialized above on red_dist/red_wastes — reuse it.
    n_total_nodes = k_nodes + 1  # VRPPModel counts total nodes including depot (index 0)
    v_model = VRPPModel(
        n_nodes=n_total_nodes,
        cost_matrix=red_dist,
        wastes=red_wastes,
        capacity=capacity,
        revenue_per_kg=R,
        cost_per_km=C,
        mandatory_nodes=red_mandatory,
    )
    sep_engine = SeparationEngine(
        model=v_model,
        enable_heuristic_rcc_separation=params.enable_heuristic_rcc_separation,
        enable_comb_cuts=params.enable_comb_cuts,
    )

    # 4. Initialize Branch-and-Bound Tree with search strategy
    bb_tree = BranchAndBoundTree(v_model=v_model, params=params)

    # 5. Initialize cutting engine
    cut_engine = create_cutting_plane_engine(
        cutting_planes_name,
        v_model,
        sep_engine,
    )
    return bb_tree, cut_engine
