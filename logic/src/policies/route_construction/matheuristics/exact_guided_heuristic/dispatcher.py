r"""Pipeline dispatcher — TCF → ALNS → BPC → SP-merge.

Orchestrates the four stages, manages the shared RoutePool, converts the
global ID map, and returns a solution in the same format expected by
``BaseRoutingPolicy._run_solver``.

The single public function ``run_pipeline`` is the seam between the per-stage
modules and the policy adapter.  It can also be used directly as a drop-in
replacement for ``_run_gurobi_optimizer``.

Attributes:
    run_pipeline: Main orchestration entry point.

Example:
    >>> from dispatcher import run_pipeline
    >>> routes, profit, cost = run_pipeline(
    ...     bins, dist_matrix, env, values, binsids, mandatory,
    ...     n_vehicles=2, params=ExactGuidedHeuristicParams(alpha=0.5, time_limit=120),
    ... )
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from .params import ExactGuidedHeuristicParams
from .route_pool import RoutePool
from .stage_alns import run_alns_stage
from .stage_bpc import run_bpc_stage
from .stage_sp import run_sp_stage
from .stage_tcf import run_tcf_stage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_alns_params(pipeline_params: ExactGuidedHeuristicParams, time_limit: float):
    """Build an ALNSParams object from the pipeline config.

    Args:
        pipeline_params: Source pipeline parameters.
        time_limit:      Stage-specific time budget (injected here).

    Returns:
        ALNSParams instance ready for the ALNS stage.
    """
    from logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.params import (
        ALNSParams,
    )

    values_dict = pipeline_params.as_alns_values_dict()
    values_dict["time_limit"] = time_limit
    return ALNSParams.from_config(values_dict)


def _mandatory_local_set(
    bins: NDArray[np.float64],
    binsids: List[int],
    mandatory: List[int],
    psi: float,
) -> Set[int]:
    """Compute the set of locally-indexed mandatory customer nodes.

    A node is mandatory if its global bin ID appears in ``mandatory`` OR its
    fill level is at or above the ``psi`` threshold.

    Args:
        bins:      Array of fill levels (local 1-based indexing).
        binsids:   Global bin IDs (length n_bins or n_bins+1).
        mandatory: Global IDs of bins that must be collected.
        psi:       Fill-level threshold (0–1); bins at ``psi*100 %`` are forced.

    Returns:
        Set of local 1-based node indices that are mandatory.
    """
    n_bins = len(bins)
    pure = binsids[1:] if len(binsids) == n_bins + 1 else binsids
    result: Set[int] = set()
    for i, bid in enumerate(pure, 1):
        if bid in mandatory:
            result.add(i)
    for i, fill in enumerate(bins, 1):
        if fill >= psi * 100:
            result.add(i)
    return result


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_pipeline(
    bins: NDArray[np.float64],
    dist_matrix: List[List[float]],
    env,
    values: Dict[str, Any],
    binsids: List[int],
    mandatory: List[int],
    n_vehicles: int = 1,
    params: Optional[ExactGuidedHeuristicParams] = None,
    recorder=None,
) -> Tuple[List[int], float, float]:
    """Execute the four-stage pipeline and return the best VRPP solution.

    Stages:
        1. TCF   — SWC-TCF MIP (fast primal + dual warm-start).
        2. ALNS  — Adaptive Large Neighbourhood Search seeded by TCF.
        3. BPC   — Branch-and-Price-and-Cut (skipped when alpha ≤ 0 or
                   skip_bpc=True or time budget < 5 s).
        4. SP    — Set-Partitioning route-pool merge.

    Args:
        bins:        Array of bin fill levels (local 1-based indexing).
        dist_matrix: Full distance matrix including depot at index 0.
        env:         Shared Gurobi environment.
        values:      Problem parameters dict (Q, R, C, Omega, delta, psi, …).
        binsids:     Global bin identifiers.
        mandatory:   Global IDs of bins that must be collected.
        n_vehicles:  Fleet size K.
        params:      ExactGuidedHeuristicParams (defaults to ExactGuidedHeuristicParams() if None).
        recorder:    Optional telemetry recorder.

    Returns:
        (flat_route, profit, total_cost)
            flat_route  — sequence of global IDs, starting with depot 0,
                          with 0 separating routes for multi-vehicle solutions.
            profit      — net profit of the selected routes.
            total_cost  — total travel cost of the selected routes.
    """
    p = params or ExactGuidedHeuristicParams()
    tau_tcf, tau_alns, tau_bpc, tau_sp = p.stage_budgets()
    t_start = time.perf_counter()

    n_bins = len(bins)
    Q = values["Q"]
    R = values["R"]
    C = values["C"]
    psi = values.get("psi", 0.99)

    # Local fill-level dict (depot at index 0 has fill 0)
    enchimentos = np.insert(bins, 0, 0.0)
    wastes_local: Dict[int, float] = {i: float(enchimentos[i]) for i in range(n_bins + 1)}

    mandatory_set = _mandatory_local_set(bins, binsids, mandatory, psi)

    # Global ID map: local index → global bin ID
    pure_binsids = binsids[1:] if len(binsids) == n_bins + 1 else binsids
    id_map: Dict[int, int] = {0: 0}
    for i, bid in enumerate(pure_binsids, 1):
        id_map[i] = bid

    pool = RoutePool()
    best_profit: float = 0.0

    # ── Stage 1: TCF ──────────────────────────────────────────────────────
    logger.info("[Pipeline] Stage 1 TCF   budget=%.1fs", tau_tcf)
    tcf_routes, tcf_profit, _ = run_tcf_stage(
        bins,
        dist_matrix,
        env,
        values,
        binsids,
        mandatory,
        n_vehicles,
        tau_tcf,
        seed=p.seed or 42,
        pool=pool,
    )
    best_profit = max(best_profit, tcf_profit)
    logger.info("[Pipeline] TCF  profit=%.4f  pool=%d", tcf_profit, len(pool))

    # ── Stage 2: ALNS ─────────────────────────────────────────────────────
    logger.info("[Pipeline] Stage 2 ALNS  budget=%.1fs", tau_alns)
    dm_np = np.array(dist_matrix)
    alns_prms = _build_alns_params(p, tau_alns)
    alns_routes, alns_profit = run_alns_stage(
        dist_matrix=dm_np,
        wastes=wastes_local,
        capacity=Q,
        R=R,
        C=C,
        params=alns_prms,
        mandatory_nodes=sorted(mandatory_set),
        time_limit=tau_alns,
        initial_routes=tcf_routes or None,
        pool=pool,
        recorder=recorder,
    )
    best_profit = max(best_profit, alns_profit)
    logger.info("[Pipeline] ALNS profit=%.4f  pool=%d", alns_profit, len(pool))

    # ── Stage 3: BPC ──────────────────────────────────────────────────────
    if not p.skip_bpc and tau_bpc > 5.0:
        logger.info("[Pipeline] Stage 3 BPC   budget=%.1fs", tau_bpc)
        _, bpc_profit = run_bpc_stage(
            dist_matrix=dm_np,
            wastes={k: v for k, v in wastes_local.items() if k > 0},
            capacity=Q,
            R=R,
            C=C,
            pipeline_params=p,
            mandatory_nodes=mandatory_set,
            time_limit=tau_bpc,
            incumbent=best_profit,
            pool=pool,
            vehicle_limit=n_vehicles if n_vehicles > 0 else None,
            env=env,
            recorder=recorder,
        )
        best_profit = max(best_profit, bpc_profit)
        logger.info("[Pipeline] BPC  profit=%.4f  pool=%d", bpc_profit, len(pool))
    else:
        logger.info("[Pipeline] Stage 3 BPC   SKIPPED (alpha=%.2f)", p.alpha)

    # Filter infeasible routes before SP
    pool.filter_feasible(Q)

    # ── Stage 4: SP merge ─────────────────────────────────────────────────
    logger.info("[Pipeline] Stage 4 SP    budget=%.1fs  pool=%d", tau_sp, len(pool))
    selected, sp_profit = run_sp_stage(
        pool=pool,
        n_nodes=n_bins,
        vehicle_limit=max(1, n_vehicles),
        mandatory=mandatory_set,
        time_limit=tau_sp,
        env=env,
        sp_pool_cap=p.sp_pool_cap,
        sp_mip_gap=p.sp_mip_gap,
        seed=p.seed or 42,
    )
    best_profit = max(best_profit, sp_profit)

    # ── Assemble flat route & compute total cost ───────────────────────────
    flat_route: List[int] = [0]
    total_cost: float = 0.0
    for r in selected:
        for local_n in r.nodes:
            flat_route.append(id_map.get(local_n, local_n))
        flat_route.append(0)  # depot return between routes
        total_cost += r.cost

    elapsed = time.perf_counter() - t_start
    logger.info(
        "[Pipeline] DONE elapsed=%.1fs profit=%.4f cost=%.4f nodes=%d",
        elapsed,
        best_profit,
        total_cost,
        len(flat_route),
    )
    return flat_route, best_profit, total_cost
