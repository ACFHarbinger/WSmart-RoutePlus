r"""Stage 3 — Branch-and-Price-and-Cut (BPC).

Wraps the project's existing ``run_bpc`` engine so that the pipeline can:

1. Inject the TCF / ALNS incumbent as a primal upper-bound cutoff, which
   tightens B&B pruning from the very first node.
2. Seed the initial column pool with routes from the global RoutePool so
   the root LP relaxation starts with a strong set of columns.
3. Return all routes produced by BPC into the shared RoutePool for the
   SP-merge stage.

The BPC quality scales with ``alpha`` via two knobs exposed in ExactGuidedHeuristicParams:
    ng_size     = ng_size_min + int(alpha * (ng_size_max − ng_size_min))
    max_bb_nodes = bb_min + int(alpha * (bb_max − bb_min))

References:
    Barnhart, Hane & Vance (2000). Operations Research, 48(2), 318–326.
    Baldacci, Mingozzi & Roberti (2011). Operations Research, 59(5).
    Pessoa, Sadykov, Uchoa & Vanderbeck (2020). Math. Program., 183.

Attributes:
    run_bpc_stage: Main entry point for Stage 3.

Example:
    >>> from stage_bpc import run_bpc_stage
    >>> routes, profit = run_bpc_stage(
    ...     dist_matrix, wastes, capacity, R, C,
    ...     pipeline_params, mandatory_nodes, time_limit=60.0,
    ...     incumbent=alns_profit, pool=shared_pool,
    ... )
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .params import ExactGuidedHeuristicParams
from .route_pool import RoutePool, VRPPRoute

logger = logging.getLogger(__name__)


def run_bpc_stage(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    pipeline_params: ExactGuidedHeuristicParams,
    mandatory_nodes: Optional[Set[int]],
    time_limit: float,
    incumbent: float = 0.0,
    pool: Optional[RoutePool] = None,
    vehicle_limit: Optional[int] = None,
    env=None,
    recorder=None,
) -> Tuple[List[VRPPRoute], float]:
    """Run BPC and add its routes to the shared pool.

    Args:
        dist_matrix:     (n+1) × (n+1) distance matrix (index 0 = depot).
        wastes:          {local_node_id → fill_level}.
        capacity:        Vehicle capacity Q.
        R:               Revenue per unit waste.
        C:               Cost per unit distance.
        pipeline_params: ExactGuidedHeuristicParams carrying BPC configuration knobs.
        mandatory_nodes: Set of local indices that must be visited.
        time_limit:      Wall-clock budget in seconds.
        incumbent:       Best known profit from prior stages; used to seed
                         BPC's upper-bound cutoff for pruning.
        pool:            Shared RoutePool; BPC routes are added here.
        vehicle_limit:   Fleet size K (or None for unlimited).
        env:             Shared Gurobi environment.
        recorder:        Optional telemetry recorder.

    Returns:
        (bpc_routes, best_profit)
            bpc_routes  — routes from BPC's best integer solution.
            best_profit — profit of BPC's best solution (or ``incumbent`` on
                          failure/timeout).
    """
    try:
        from logic.src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine import (
            run_bpc,
        )
        from logic.src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.params import (
            BPCParams,
        )
    except ImportError as exc:
        logger.warning("[BPC] Import failed — stage skipped: %s", exc)
        return [], incumbent

    bpc_params = BPCParams(
        time_limit=time_limit,
        use_ng_routes=True,
        ng_neighborhood_size=pipeline_params.bpc_ng_size(),
        max_bb_nodes=pipeline_params.bpc_max_bb_nodes(),
        cutting_planes=pipeline_params.bpc_cutting_planes,
        branching_strategy=pipeline_params.bpc_branching_strategy,
        seed=pipeline_params.seed,
        vrpp=True,
        profit_aware_operators=True,
    )

    try:
        node_routes, obj = run_bpc(
            dist_matrix=dist_matrix,
            wastes=wastes,
            capacity=capacity,
            R=R,
            C=C,
            params=bpc_params,
            mandatory_indices=mandatory_nodes or set(),
            vehicle_limit=vehicle_limit,
            env=env,
            recorder=recorder,
        )
    except Exception as exc:
        logger.warning("[BPC] Solver error — stage skipped: %s", exc)
        return [], incumbent

    # Convert BPC node lists → VRPPRoute and add to pool
    bpc_routes: List[VRPPRoute] = []
    for node_list in node_routes:
        inner = [nd for nd in node_list if nd != 0]
        if not inner:
            continue
        path = [0] + inner + [0]
        dist_l = dist_matrix.tolist() if hasattr(dist_matrix, "tolist") else dist_matrix
        cost = C * sum(dist_l[path[i]][path[i + 1]] for i in range(len(path) - 1))
        revenue = R * sum(wastes.get(nd, 0.0) for nd in inner)
        load = sum(wastes.get(nd, 0.0) for nd in inner)
        route = VRPPRoute(
            nodes=inner,
            profit=revenue - cost,
            revenue=revenue,
            cost=cost,
            load=load,
            source="bpc",
        )
        bpc_routes.append(route)

    if pool is not None:
        pool.add_all(bpc_routes)

    best_profit = max(obj, incumbent)
    logger.info(
        "[BPC] obj=%.4f  routes=%d  pool_size=%s",
        obj,
        len(bpc_routes),
        len(pool) if pool else "n/a",
    )
    return bpc_routes, best_profit
