r"""Stage 1 — Logic-Based Benders Decomposition (LBBD).

Decomposes the VRPP into:

Master problem (Gurobi MIP)
    Knapsack-style node-selection problem: choose *which* customers to visit
    subject to a profit-coverage lower bound.  The master accumulates Benders
    cuts from the sub-problem across iterations.

Sub-problem (routing oracle)
    Given the selected customer set Y ⊆ V, solve the routing problem to
    find the optimal route(s) covering Y.  The routing oracle is pluggable:
    'greedy' (fast), 'alns' (heuristic), or 'bpc' (exact).

Cut generation (Hooker 2000, Hooker & Ottosson 2003)
----------------------------------------------------
Three cut families are implemented:

1. No-good cuts  (always valid)
   Forbid the exact selection Y* found in the current iteration:
       Σ_{i ∈ Y*} y_i  ≤  |Y*| - 1

2. Optimality cuts  (valid when sub-problem is solved to optimality)
   If the routing cost for Y* is c*, any superset of Y* requires
   routing cost ≥ c*:
       θ  ≥  c*  −  M · (|Y*| − Σ_{i ∈ Y*} y_i)
   where θ is the routing cost proxy in the master.

3. Pareto-optimal (Magnanti-Wang 1981) cuts
   Strengthen optimality cuts by tightening M using the best-known
   primal bound π:
       θ  ≥  (c* − π)  ·  Σ_{i ∈ Y*} y_i  −  (c* − π) · |Y*| + π

All routes produced by the sub-problem are added to the shared RoutePool
for the SP-merge stage.

References:
    Hooker, J. N. (2000). Logic-Based Methods for Optimization. Wiley.
    Hooker, J. N., & Ottosson, G. (2003). Logic-based Benders decomposition.
        Mathematical Programming, 96(1), 33–60.
    Magnanti, T. L., & Wong, R. T. (1981). Accelerating Benders decomposition.
        Operations Research, 29(3), 464–484.
    Fachini, V., & Armentano, V. A. (2020). Logic-based Benders decomposition
        for the heterogeneous fixed fleet VRP with TW.
        Computers & Industrial Engineering, 148, 106652.

Attributes:
    run_lbbd_stage: Main entry point for Stage 1.

Example:
    >>> from logic.src.policies.routing.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd import run_lbbd_stage
    >>> routes, _ = run_lbbd_stage(env, wastes, capacity, R, C)

Routing subproblem oracles live in ``lbbd_subproblems``. The Gurobi master lives in ``lbbd_master``. ``run_lbbd_stage`` stays here and both pieces are re-exported.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from logic.src.policies.routing.matheuristics.exact_guided_heuristic.route_pool import RoutePool, VRPPRoute

from .lbbd_master import _LBBDMaster as _LBBDMaster
from .lbbd_subproblems import (
    _make_vrpp_route as _make_vrpp_route,
)
from .lbbd_subproblems import (
    _route_cost_from_nodes as _route_cost_from_nodes,
)
from .lbbd_subproblems import (
    _route_revenue_from_nodes as _route_revenue_from_nodes,
)
from .lbbd_subproblems import (
    _solve_sub_alns as _solve_sub_alns,
)
from .lbbd_subproblems import (
    _solve_sub_bpc as _solve_sub_bpc,
)
from .lbbd_subproblems import (
    _solve_sub_greedy as _solve_sub_greedy,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_lbbd_stage(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory: Set[int],
    n_vehicles: int,
    time_limit: float,
    max_iterations: int,
    sub_solver: str,
    cut_families: List[str],
    pareto_eps: float,
    min_cover_ratio: float,
    master_time_frac: float,
    sub_time_frac: float,
    pool: Optional[RoutePool],
    seed: int,
    env: Any,
    incumbent: float = 0.0,
) -> Tuple[List[VRPPRoute], float, float]:
    """Execute LBBD and return routes, best profit, and LP upper bound.

    Args:
        dist_matrix:      Full distance matrix (depot at index 0).
        wastes:           {local node → fill_level}.
        capacity:         Vehicle capacity Q.
        R:                Revenue per unit waste.
        C:                Cost per unit distance.
        mandatory:        Local indices of mandatory customers.
        n_vehicles:       Fleet size K.
        time_limit:       Total LBBD stage budget (seconds).
        max_iterations:   Maximum outer Benders iterations.
        sub_solver:       'greedy' | 'alns' | 'bpc'.
        cut_families:     Active cut families (list of strings).
        pareto_eps:       Tolerance for Pareto-optimal cut.
        min_cover_ratio:  Min fraction of nodes the master must cover.
        master_time_frac: Fraction of per-iteration budget for master.
        sub_time_frac:    Fraction of per-iteration budget for sub.
        pool:             Shared RoutePool.
        seed:             RNG seed.
        env:              Shared Gurobi environment.
        incumbent:        Best profit known before LBBD starts.

    Returns:
        (best_routes, best_profit, lp_upper_bound)
    """
    n_nodes = len(dist_matrix) - 1
    t_start = time.perf_counter()
    per_iter = max(2.0, time_limit / max(1, max_iterations))

    # Resolve sub-problem oracle
    vehicle_lim = n_vehicles if n_vehicles > 0 else None
    _sub_dispatch: Dict[str, Callable] = {
        "greedy": lambda Y, tl: _solve_sub_greedy(Y, dist_matrix, wastes, capacity, R, C, mandatory),
        "alns": lambda Y, tl: _solve_sub_alns(Y, dist_matrix, wastes, capacity, R, C, mandatory, tl, seed),
        "bpc": lambda Y, tl: _solve_sub_bpc(
            Y, dist_matrix, wastes, capacity, R, C, mandatory, tl, seed, vehicle_lim, env
        ),
    }
    sub_fn = _sub_dispatch.get(sub_solver, _sub_dispatch["greedy"])

    master = _LBBDMaster(
        n_nodes=n_nodes,
        wastes=wastes,
        R=R,
        mandatory=mandatory,
        min_cover_ratio=min_cover_ratio,
        vehicle_limit=vehicle_lim,
        env=env,
        seed=seed,
    )

    best_routes: List[VRPPRoute] = []
    best_profit: float = incumbent
    lp_ub: float = float("inf")
    visited: Set[frozenset] = set()

    for iteration in range(max_iterations):
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_limit:
            break

        remaining = time_limit - elapsed
        master_tl = min(per_iter * master_time_frac, remaining * 0.3)
        sub_tl = min(per_iter * sub_time_frac, remaining * 0.7)

        # ── Master solve ────────────────────────────────────────────────
        Y = master.solve(master_tl)
        if Y is None:
            logger.info("[LBBD] Master infeasible at iter %d — stopping", iteration)
            break

        lp_ub = min(lp_ub, master.get_lp_bound())

        key = frozenset(Y)
        if key in visited:
            # Pure no-good loop — add another no-good to escape and continue
            master.add_nogood_cut(Y)
            continue
        visited.add(key)

        # ── Sub-problem solve ───────────────────────────────────────────
        sub_routes_raw, sub_profit = sub_fn(Y, sub_tl)

        # Record sub-problem routes into pool
        for r in sub_routes_raw:
            inner = [nd for nd in r if nd != 0]
            if inner:
                vr = _make_vrpp_route(inner, dist_matrix, wastes, R, C, f"lbbd_{sub_solver}")
                if pool is not None:
                    pool.add(vr)
                if sub_profit > best_profit + 1e-9:
                    best_routes = [vr]
                    best_profit = sub_profit

        # Routing cost for cut generation
        sub_cost = sum(_route_cost_from_nodes([nd for nd in r if nd != 0], dist_matrix, C) for r in sub_routes_raw)

        # ── Cut generation ──────────────────────────────────────────────
        if "nogood" in cut_families:
            master.add_nogood_cut(Y)

        if "optimality" in cut_families and sub_cost < float("inf"):
            master.add_optimality_cut(Y, sub_cost)

        if "pareto" in cut_families and sub_cost < float("inf"):
            master.add_pareto_cut(Y, sub_cost, best_profit)

        if "combinatorial" in cut_families and sub_cost < float("inf"):
            master.add_combinatorial_cut(Y, sub_cost)

        logger.debug(
            "[LBBD] iter=%d  |Y|=%d  sub_profit=%.4f  best=%.4f  cuts=%d",
            iteration,
            len(Y),
            sub_profit,
            best_profit,
            master.n_cuts,
        )

        # Convergence: LP bound ≈ primal
        if lp_ub - best_profit <= pareto_eps * max(1.0, abs(best_profit)):
            logger.info("[LBBD] Converged at iter %d (gap=%.6f)", iteration, lp_ub - best_profit)
            break

    logger.info(
        "[LBBD] done  iters=%d  profit=%.4f  lp_ub=%.4f  cuts=%d  pool=%s",
        iteration + 1,
        best_profit,
        lp_ub,
        master.n_cuts,
        len(pool) if pool else "n/a",
    )
    return best_routes, best_profit, lp_ub
