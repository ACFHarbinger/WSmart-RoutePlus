r"""LBBD → ALNS → BPC → RL → SP pipeline dispatcher.

Orchestrates the five computational stages:

    Stage 1  LBBD   — node-selection MIP + routing sub-problem + Benders cuts.
    Stage 2  ALNS   — metaheuristic improvement seeded by LBBD incumbent.
    Stage 3  BPC    — exact column generation (skipped when alpha ≤ 0).
    Stage 4  RL     — bandit controller updates time allocations for next call.
    Stage 5  SP     — Set-Partitioning merge over the global route pool.

The RL controller (Stage 4) is stateful: it persists across calls within the
same simulation run.  A module-level singleton ``_GLOBAL_RL_CONTROLLER`` is
created on first call and reused thereafter.  If ``rl_policy_path`` changes
between runs, call ``reset_rl_controller()`` to force re-initialisation.

Attributes:
    run_lasm_pipeline: Main entry point.
    reset_rl_controller: Force re-initialise the global RL controller.

Example:
    >>> from dispatcher import run_lasm_pipeline
    >>> routes, profit, cost = run_lasm_pipeline(
    ...     bins, dist_matrix, env, values, binsids, mandatory,
    ...     n_vehicles=2,
    ...     params=LASMPipelineParams(alpha=0.5, time_limit=120),
    ... )
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass as _dc
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

import numpy as np
from numpy.typing import NDArray

from logic.src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool import RoutePool

from .params import LASMPipelineParams
from .rl_controller import RLController
from .stage_lbbd import run_lbbd_stage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level RL controller singleton
# ---------------------------------------------------------------------------

_GLOBAL_RL: Optional[RLController] = None


def reset_rl_controller() -> None:
    """Force re-initialisation of the global RL controller on the next call.

    Returns:
        None
    """
    global _GLOBAL_RL
    _GLOBAL_RL = None


def _get_rl_controller(params: LASMPipelineParams) -> RLController:
    """Get the global RL controller (singleton).

    Args:
        params: Parameters for the RL controller.

    Returns:
        RLController: The global RL controller.
    """
    global _GLOBAL_RL
    if _GLOBAL_RL is None:
        _GLOBAL_RL = RLController(params)
        logger.info("[RL] Controller initialised (mode=%s)", params.rl_mode)
    return _GLOBAL_RL


# ---------------------------------------------------------------------------
# Shared-stage imports (with graceful fallback)
# ---------------------------------------------------------------------------


def _import_alns_stage() -> Optional[Callable[[Any, Any, Any, Any, Any, Any, Any, Any], Optional[Any]]]:
    """Import ALNS stage with fallback.

    Returns:
        Optional[Callable[[Any, Any, Any, Any, Any, Any, Any, Any], Optional[Any]]]: The ALNS stage function.
    """
    try:
        from logic.src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_alns import run_alns_stage

        return run_alns_stage
    except ImportError:
        return None


def _import_bpc_stage() -> Optional[Callable[[Any, Any, Any, Any, Any, Any, Any, Any], Optional[Any]]]:
    """Import BPC stage with fallback.

    Returns:
        Optional[Callable[[Any, Any, Any, Any, Any, Any, Any, Any], Optional[Any]]]: The BPC stage function.
    """
    try:
        from logic.src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_bpc import run_bpc_stage

        return run_bpc_stage
    except ImportError:
        return None


def _import_sp_stage() -> Optional[Callable[[Any, Any, Any, Any, Any, Any, Any, Any], Optional[Any]]]:
    """Import SP stage with fallback.

    Returns:
        Optional[Callable[[Any, Any, Any, Any, Any, Any, Any, Any], Optional[Any]]]: The SP stage function.
    """
    try:
        from logic.src.policies.route_construction.matheuristics.exact_guided_heuristic.stage_sp import run_sp_stage

        return run_sp_stage
    except ImportError:
        # Inline fallback — greedy best route if SP unavailable
        def _fallback_sp(
            pool, n_nodes, vehicle_limit, mandatory, time_limit, env=None, sp_pool_cap=50_000, sp_mip_gap=1e-4, seed=42
        ):
            routes = pool.routes()
            if not routes:
                return [], 0.0
            routes.sort(key=lambda r: r.profit, reverse=True)
            return routes[:vehicle_limit], sum(r.profit for r in routes[:vehicle_limit])

        return _fallback_sp


def _import_alns_params() -> Optional[Type[Any]]:
    """Import ALNS parameters with fallback.

    Returns:
        Optional[Type[Any]]: The ALNS parameters class.
    """
    try:
        from logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.params import (
            ALNSParams,
        )

        return ALNSParams
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Helper: mandatory set
# ---------------------------------------------------------------------------


def _mandatory_local_set(
    bins: NDArray[np.float64],
    binsids: List[int],
    mandatory: List[int],
    psi: float,
) -> Set[int]:
    """Compute the set of mandatory local bin indices.

    Args:
        bins: Fill levels of the bins.
        binsids: Global identifiers of the bins.
        mandatory: Global identifiers of the mandatory bins.
        psi: The threshold for mandatory bins.

    Returns:
        Set[int]: The set of mandatory local bin indices.
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


def run_lasm_pipeline(
    bins: NDArray[np.float64],
    dist_matrix: List[List[float]],
    env: Any,
    values: Dict[str, Any],
    binsids: List[int],
    mandatory: List[int],
    n_vehicles: int = 1,
    params: Optional[LASMPipelineParams] = None,
    recorder: Any = None,
) -> Tuple[List[int], float, float]:
    """Execute the LBBD → ALNS → BPC → RL → SP pipeline.

    Args:
        bins:        Fill levels (local 1-based indices).
        dist_matrix: Full distance matrix (depot at index 0).
        env:         Shared Gurobi environment.
        values:      Problem-parameter dict (Q, R, C, Omega, delta, psi).
        binsids:     Global bin identifiers.
        mandatory:   Global IDs of bins that MUST be collected.
        n_vehicles:  Fleet size K.
        params:      LASMPipelineParams (defaults to LASMPipelineParams()).
        recorder:    Optional telemetry recorder.

    Returns:
        (flat_route, profit, total_cost)
            flat_route — sequence of global IDs with 0 as depot/separator.
            profit     — net profit of the selected routes.
            total_cost — total travel cost.
    """
    p = params or LASMPipelineParams()
    t_start = time.perf_counter()

    n_bins = len(bins)
    Q = values["Q"]
    R = values["R"]
    C = values["C"]
    psi = values.get("psi", 0.99)

    enchimentos = np.insert(bins, 0, 0.0)
    wastes_local = {i: float(enchimentos[i]) for i in range(n_bins + 1)}
    mandatory_set = _mandatory_local_set(bins, binsids, mandatory, psi)

    pure_binsids = binsids[1:] if len(binsids) == n_bins + 1 else binsids
    id_map: Dict[int, int] = {0: 0}
    for i, bid in enumerate(pure_binsids, 1):
        id_map[i] = bid

    pool = RoutePool()
    best_profit = 0.0
    lp_ub = float("inf")
    dm_np = np.array(dist_matrix)

    # ── RL controller context ──────────────────────────────────────────────
    rl_ctrl = _get_rl_controller(p)
    fill_arr = bins / 100.0  # normalise to [0,1]
    mand_ratio = len(mandatory_set) / max(1, n_bins)

    rl_ctx = rl_ctrl.make_context(
        n_nodes=n_bins,
        fill_levels=fill_arr,
        mandatory_ratio=mand_ratio,
        lp_ub=lp_ub,
        best_profit=best_profit,
        pool_size=0,
        time_remaining=p.time_limit,
        time_total=p.time_limit,
    )

    # Alpha-derived stage budgets as fallback
    tau_lbbd, tau_alns, tau_bpc, tau_rl, tau_sp = p.stage_budgets()
    alpha_fracs = {
        "lbbd": tau_lbbd / p.time_limit,
        "alns": tau_alns / p.time_limit,
        "bpc": tau_bpc / p.time_limit,
        "sp": tau_sp / p.time_limit,
    }

    rl_action = rl_ctrl.act(rl_ctx, p.time_limit, alpha_fracs)
    budget_fracs = rl_action["budget_fracs"] or alpha_fracs
    # op_mults      = rl_action["operator_mults"]     # np.ndarray[3]
    action_levels = rl_action["action_levels"]

    # Recompute actual budgets from (possibly RL-adjusted) fractions
    T = p.time_limit
    tau_lbbd = T * budget_fracs.get("lbbd", alpha_fracs["lbbd"])
    tau_alns = T * budget_fracs.get("alns", alpha_fracs["alns"])
    tau_bpc = T * budget_fracs.get("bpc", alpha_fracs["bpc"])
    tau_sp = T * budget_fracs.get("sp", alpha_fracs["sp"])

    # ── Stage 1: LBBD ─────────────────────────────────────────────────────
    logger.info("[LBBDPipeline] Stage 1 LBBD  budget=%.1fs", tau_lbbd)
    lbbd_routes, lbbd_profit, lp_ub = run_lbbd_stage(
        dist_matrix=dm_np,
        wastes={k: v for k, v in wastes_local.items() if k > 0},
        capacity=Q,
        R=R,
        C=C,
        mandatory=mandatory_set,
        n_vehicles=n_vehicles,
        time_limit=tau_lbbd,
        max_iterations=p.lbbd_max_iterations,
        sub_solver=p.lbbd_sub_solver,
        cut_families=p.lbbd_cut_families,
        pareto_eps=p.lbbd_pareto_eps,
        min_cover_ratio=p.lbbd_min_cover_ratio,
        master_time_frac=p.lbbd_master_time_frac,
        sub_time_frac=p.lbbd_sub_time_frac,
        pool=pool,
        seed=p.seed or 42,
        env=env,
        incumbent=best_profit,
    )
    best_profit = max(best_profit, lbbd_profit)
    logger.info("[LBBDPipeline] LBBD profit=%.4f  lp_ub=%.4f  pool=%d", lbbd_profit, lp_ub, len(pool))

    # ── Stage 2: ALNS ─────────────────────────────────────────────────────
    logger.info("[LBBDPipeline] Stage 2 ALNS  budget=%.1fs", tau_alns)
    run_alns_stage = _import_alns_stage()
    ALNSParams = _import_alns_params()

    if run_alns_stage is not None and ALNSParams is not None:
        alns_vals = p.as_alns_values_dict()
        alns_vals["time_limit"] = tau_alns
        # Apply RL operator multipliers to destroy-operator initial weights
        # (they will be further adapted by ALNS's internal roulette mechanism)
        alns_params = ALNSParams.from_config(alns_vals)

        alns_routes, alns_profit = run_alns_stage(
            dist_matrix=dm_np,
            wastes=wastes_local,
            capacity=Q,
            R=R,
            C=C,
            params=alns_params,
            mandatory_nodes=sorted(mandatory_set),
            time_limit=tau_alns,
            initial_routes=lbbd_routes or None,
            pool=pool,
            recorder=recorder,
        )
        best_profit = max(best_profit, alns_profit)
        logger.info("[LBBDPipeline] ALNS profit=%.4f  pool=%d", alns_profit, len(pool))
    else:
        logger.warning("[LBBDPipeline] ALNS unavailable — skipping")

    # ── Stage 3: BPC ──────────────────────────────────────────────────────
    run_bpc_stage = _import_bpc_stage()

    if run_bpc_stage is not None and not p.skip_bpc and tau_bpc > 5.0:
        logger.info("[LBBDPipeline] Stage 3 BPC   budget=%.1fs", tau_bpc)

        # Build a minimal ExactGuidedHeuristicParams-compatible object for the shared stage
        @_dc
        class _BPCProxy:
            bpc_ng_size_min: int = p.bpc_ng_size_min
            bpc_ng_size_max: int = p.bpc_ng_size_max
            bpc_max_bb_nodes_min: int = p.bpc_max_bb_nodes_min
            bpc_max_bb_nodes_max: int = p.bpc_max_bb_nodes_max
            bpc_cutting_planes: str = p.bpc_cutting_planes
            bpc_branching_strategy: str = p.bpc_branching_strategy
            seed: Any = p.seed
            alpha: float = p.alpha

            def bpc_ng_size(self) -> int:
                a = max(0.0, min(1.0, self.alpha))
                return self.bpc_ng_size_min + int(a * (self.bpc_ng_size_max - self.bpc_ng_size_min))

            def bpc_max_bb_nodes(self) -> int:
                a = max(0.0, min(1.0, self.alpha))
                return self.bpc_max_bb_nodes_min + int(a * (self.bpc_max_bb_nodes_max - self.bpc_max_bb_nodes_min))

        _, bpc_profit = run_bpc_stage(
            dist_matrix=dm_np,
            wastes={k: v for k, v in wastes_local.items() if k > 0},
            capacity=Q,
            R=R,
            C=C,
            pipeline_params=_BPCProxy(),
            mandatory_nodes=mandatory_set,
            time_limit=tau_bpc,
            incumbent=best_profit,
            pool=pool,
            vehicle_limit=n_vehicles if n_vehicles > 0 else None,
            env=env,
            recorder=recorder,
        )
        best_profit = max(best_profit, bpc_profit)
        logger.info("[LBBDPipeline] BPC  profit=%.4f  pool=%d", bpc_profit, len(pool))
    else:
        logger.info("[LBBDPipeline] Stage 3 BPC   SKIPPED")

    # ── Stage 4: RL update ────────────────────────────────────────────────
    profit_before_lbbd = 0.0  # base before this call
    delta_profit = max(0.0, best_profit - profit_before_lbbd)
    delta_time = time.perf_counter() - t_start

    if action_levels:
        # Update RL context with post-LBBD+ALNS+BPC information
        rl_ctx_post = rl_ctrl.make_context(
            n_nodes=n_bins,
            fill_levels=fill_arr,
            mandatory_ratio=mand_ratio,
            lp_ub=lp_ub,
            best_profit=best_profit,
            pool_size=len(pool),
            time_remaining=max(0.0, tau_sp - (time.perf_counter() - t_start - delta_time)),
            time_total=p.time_limit,
        )
        rl_ctrl.update(rl_ctx_post, action_levels, delta_profit, delta_time, best_profit)

    # Persist policy if path given
    if p.rl_mode in ("online", "hybrid") and p.rl_policy_path:
        try:
            rl_ctrl.save(p.rl_policy_path)
        except Exception as exc:
            logger.debug("[RL] Could not save policy: %s", exc)

    # ── Stage 5: SP merge ─────────────────────────────────────────────────
    pool.filter_feasible(Q)
    run_sp_stage = _import_sp_stage()

    logger.info("[LBBDPipeline] Stage 5 SP    budget=%.1fs  pool=%d", tau_sp, len(pool))
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

    # ── Assemble output ────────────────────────────────────────────────────
    flat_route: List[int] = [0]
    total_cost: float = 0.0
    for r in selected:
        for local_n in r.nodes:
            flat_route.append(id_map.get(local_n, local_n))
        flat_route.append(0)
        total_cost += r.cost

    elapsed = time.perf_counter() - t_start
    logger.info(
        "[LBBDPipeline] DONE  elapsed=%.1fs  profit=%.4f  cost=%.4f  nodes=%d",
        elapsed,
        best_profit,
        total_cost,
        len(flat_route),
    )
    return flat_route, best_profit, total_cost
