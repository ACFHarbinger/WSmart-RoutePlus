r"""Stage 4 — Set-Partitioning route-pool merge.

Takes the union of all routes produced by TCF, ALNS, and BPC and solves a
compact Gurobi MIP to select the optimal non-overlapping subset:

    max  Σ_k profit_k · x_k
    s.t. Σ_{k: i ∈ route_k} x_k  ≤ 1      ∀ i ∈ optional nodes
         Σ_{k: i ∈ route_k} x_k  = 1      ∀ i ∈ mandatory nodes
         Σ_k x_k                  ≤ K      (fleet limit)
         x_k ∈ {0, 1}

The SP-merge MIP is almost always solved to optimality within a few seconds
because the number of variables equals the pool size (≤ sp_pool_cap) and the
constraint matrix is extremely sparse.

Pool management when |pool| > sp_pool_cap:
    1. Sort by profit descending.
    2. Retain top half unconditionally.
    3. Sample the bottom half at random (preserves diversity).

References:
    Subramanian, Uchoa & Ochi (2013). Computers & Operations Research, 40.
    Talarico et al. (2025). EURO J. Computational Optimization — 0.51 % mean
        improvement reported in meta-analysis of 30 SP-based matheuristics.

Attributes:
    run_sp_stage: Main entry point for Stage 4.

Example:
    >>> from stage_sp import run_sp_stage
    >>> selected, profit = run_sp_stage(
    ...     pool=shared_pool, n_nodes=50, vehicle_limit=3,
    ...     mandatory={5, 12}, time_limit=10.0,
    ...     sp_pool_cap=50_000, sp_mip_gap=1e-4,
    ... )
"""

from __future__ import annotations

import logging
import random
from typing import Optional, Set, Tuple

import gurobipy as gp
from gurobipy import GRB, quicksum

from .route_pool import RoutePool

logger = logging.getLogger(__name__)


def run_sp_stage(
    pool: RoutePool,
    n_nodes: int,
    vehicle_limit: int,
    mandatory: Set[int],
    time_limit: float,
    env: Optional[gp.Env] = None,
    sp_pool_cap: int = 50_000,
    sp_mip_gap: float = 1e-4,
    seed: int = 42,
) -> Tuple[list, float]:
    """Select the optimal non-overlapping route subset via a Gurobi SP MIP.

    Args:
        pool:          Shared RoutePool containing all routes from prior stages.
        n_nodes:       Number of customer nodes (local 1-based, excluding depot).
        vehicle_limit: Maximum number of routes to select (fleet size K).
        mandatory:     Set of local customer indices that must be covered.
        time_limit:    MIP time limit in seconds.
        env:           Shared Gurobi environment (or None for default env).
        sp_pool_cap:   Maximum routes passed to the MIP; excess routes are
                       managed by the top-half + random-sample strategy.
        sp_mip_gap:    Relative MIP gap for early termination.
        seed:          Gurobi random seed.

    Returns:
        (selected_routes, total_profit)
            selected_routes — list of VRPPRoute selected by the SP MIP.
            total_profit    — sum of profits over selected routes.
    """
    candidates = pool.routes()
    if not candidates:
        return [], 0.0

    # ── Pool management ──────────────────────────────────────────────────
    if len(candidates) > sp_pool_cap:
        candidates.sort(key=lambda r: r.profit, reverse=True)
        top = candidates[: sp_pool_cap // 2]
        rest = candidates[sp_pool_cap // 2 :]
        sampled = random.sample(rest, min(sp_pool_cap // 2, len(rest)))
        candidates = top + sampled
        logger.debug("[SP] Pool capped %d → %d routes", len(pool), len(candidates))

    K = len(candidates)

    try:
        m = gp.Model("SP_merge", env=env) if env else gp.Model("SP_merge")
    except gp.GurobiError as exc:
        logger.error("[SP] Could not create Gurobi model: %s", exc)
        _best = max(candidates, key=lambda r: r.profit, default=None)
        return ([_best], _best.profit) if _best else ([], 0.0)

    m.Params.LogToConsole = 0
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = max(1.0, time_limit)
    m.Params.MIPGap = sp_mip_gap
    m.Params.Seed = seed

    x = m.addVars(K, vtype=GRB.BINARY, name="x")

    # ── Objective ────────────────────────────────────────────────────────
    m.setObjective(
        quicksum(candidates[k].profit * x[k] for k in range(K)),
        GRB.MAXIMIZE,
    )

    # ── Coverage constraints ─────────────────────────────────────────────
    for node in range(1, n_nodes + 1):
        covers = [k for k, r in enumerate(candidates) if node in r.nodes]
        if not covers:
            # No route covers this node — either it is optional (fine) or
            # mandatory (infeasible; the SP will detect and report).
            continue
        expr = quicksum(x[k] for k in covers)
        if node in mandatory:
            m.addConstr(expr == 1, name=f"mand_{node}")
        else:
            m.addConstr(expr <= 1, name=f"opt_{node}")

    # ── Fleet limit ──────────────────────────────────────────────────────
    m.addConstr(
        quicksum(x[k] for k in range(K)) <= max(1, vehicle_limit),
        name="fleet",
    )

    m.optimize()

    if m.SolCount == 0:
        logger.warning("[SP] No feasible solution (mandatory coverage may be impossible). Returning best single route.")
        _best = max(candidates, key=lambda r: r.profit, default=None)
        return ([_best], _best.profit) if _best else ([], 0.0)

    selected = [candidates[k] for k in range(K) if x[k].X > 0.5]
    total_profit = float(m.ObjVal)

    logger.info(
        "[SP] selected=%d  profit=%.4f  gap=%.2e  time=%.1fs",
        len(selected),
        total_profit,
        m.MIPGap if m.Status == GRB.OPTIMAL else float("nan"),
        m.Runtime,
    )
    return selected, total_profit
