r"""Stage 2 — Adaptive Large Neighbourhood Search (ALNS).

Wraps the project's existing ``ALNSSolver`` / ``run_alns`` dispatcher so that
the pipeline can:

1. Seed ALNS with the TCF incumbent (or any external initial solution).
2. Collect **every distinct route visited** during the search — not just the
   final best — into the shared ``RoutePool`` for the SP-merge stage.
3. Return control metadata (best profit, wall-clock used) to the orchestrator.

The route-pool harvesting is done by a thin wrapper around ``ALNSSolver``
that overrides ``_update_weights`` to record each candidate solution after
every iteration.  This is the standard "route pool as a by-product of ALNS"
technique described in Subramanian, Uchoa & Ochi (2013, Computers & OR 40).

References:
    Ropke & Pisinger (2006). Transportation Science, 40(4), 455–472.
    Subramanian, Uchoa & Ochi (2013). Computers & Operations Research, 40.

Attributes:
    run_alns_stage: Main entry point for Stage 2.

Example:
    >>> from stage_alns import run_alns_stage
    >>> routes, profit = run_alns_stage(
    ...     dist_matrix, wastes, capacity, R, C,
    ...     params, mandatory_nodes, time_limit=60.0,
    ...     initial_routes=tcf_routes, pool=shared_pool,
    ... )
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns import ALNSSolver
from logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.params import ALNSParams

from .route_pool import RoutePool, VRPPRoute

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pool-harvesting wrapper
# ---------------------------------------------------------------------------


class _PoolHarvestingALNS(ALNSSolver):
    """ALNSSolver subclass that records every candidate into a RoutePool.

    After each iteration's operator application, the candidate solution's
    routes are converted to ``VRPPRoute`` objects and added to the pool.
    Only the ``_update_weights`` hook is overridden — all ALNS logic is
    inherited unchanged from the parent.

    Attributes:
        _harvest_pool: Route pool.
    """

    def __init__(
        self,
        *args,
        pool: RoutePool,
        **kwargs,
    ) -> None:
        """
        Initialize the PoolHarvestingALNS.

        Args:
            pool: Route pool.
            args: Positional arguments to pass to the parent constructor.
            kwargs: Keyword arguments to pass to the parent constructor.

        Returns:
            None
        """
        super().__init__(*args, **kwargs)
        self._harvest_pool = pool

    def _update_weights(self, d_idx: int, r_idx: int, score: float) -> None:
        """Inherited weight update + route harvesting.

        Called after every iteration with the candidate routes still in scope
        via `_select_and_apply_operators`.  We reconstruct the candidate routes
        from the current working state exposed by the parent through the public
        interface, then delegate weight updating to the parent.

        Args:
            d_idx: Index of the destroy operator used.
            r_idx: Index of the repair operator used.
            score: Quality score for this iteration.
        """
        # Delegate first so parent state stays consistent.
        super()._update_weights(d_idx, r_idx, score)

    def _record_routes(self, routes: List[List[int]]) -> None:
        """Convert a route list and add to the shared pool.

        Args:
            routes: List of customer-node lists (local 1-based indices).
        """
        for r in routes:
            if not r:
                continue
            path = [0] + r + [0]
            cost = self.C * sum(self.dist_matrix[path[i]][path[i + 1]] for i in range(len(path) - 1))
            revenue = self.R * sum(self.wastes.get(n, 0.0) for n in r)
            load = sum(self.wastes.get(n, 0.0) for n in r)
            self._harvest_pool.add(
                VRPPRoute(
                    nodes=list(r),
                    profit=revenue - cost,
                    revenue=revenue,
                    cost=cost,
                    load=load,
                    source="alns",
                )
            )

    # Override the operator application to harvest candidate routes
    def _select_and_apply_operators(self, current_routes: List[List[int]]) -> Tuple[List[List[int]], int, int]:
        """Select and apply operators.

        Args:
            current_routes: Current routes.

        Returns:
            Tuple of (new_routes, d_idx, r_idx).
        """
        new_routes, d_idx, r_idx = super()._select_and_apply_operators(current_routes)
        self._record_routes(new_routes)
        return new_routes, d_idx, r_idx


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_alns_stage(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    params: ALNSParams,
    mandatory_nodes: Optional[List[int]],
    time_limit: float,
    initial_routes: Optional[List[VRPPRoute]] = None,
    pool: Optional[RoutePool] = None,
    recorder=None,
) -> Tuple[List[VRPPRoute], float]:
    """Run ALNS with route-pool harvesting.

    The stage constructs a ``_PoolHarvestingALNS`` solver, seeds it with the
    TCF incumbent (converted back to node lists), runs the full ALNS loop, and
    returns both the best VRPPRoute list and the best profit found.

    Args:
        dist_matrix:     (n+1) × (n+1) distance matrix (index 0 = depot).
        wastes:          {local_node_id → fill_level}.
        capacity:        Vehicle capacity Q.
        R:               Revenue per unit waste.
        C:               Cost per unit distance.
        params:          ALNSParams controlling the search.
        mandatory_nodes: Local indices of mandatory customers.
        time_limit:      Wall-clock budget in seconds (injected via params.time_limit
                         override so the existing time-limit logic still works).
        initial_routes:  Optional warm-start routes (e.g., from TCF).
        pool:            Shared RoutePool; harvested routes are added here.
        recorder:        Optional PolicyStateRecorder for telemetry.

    Returns:
        (best_routes, best_profit)
            best_routes — list of VRPPRoute from the best solution found.
            best_profit — net profit of the best solution.
    """
    t0 = time.perf_counter()

    # Patch time_limit into params (avoids mutating the shared ExactGuidedHeuristicParams)
    patched_params = ALNSParams(
        time_limit=time_limit,
        max_iterations=params.max_iterations,
        start_temp=params.start_temp,
        cooling_rate=params.cooling_rate,
        reaction_factor=params.reaction_factor,
        min_removal=params.min_removal,
        start_temp_control=params.start_temp_control,
        xi=params.xi,
        segment_size=params.segment_size,
        noise_factor=params.noise_factor,
        worst_removal_randomness=params.worst_removal_randomness,
        shaw_randomization=params.shaw_randomization,
        max_removal_cap=params.max_removal_cap,
        regret_pool=params.regret_pool,
        sigma_1=params.sigma_1,
        sigma_2=params.sigma_2,
        sigma_3=params.sigma_3,
        vrpp=params.vrpp,
        profit_aware_operators=params.profit_aware_operators,
        extended_operators=params.extended_operators,
        seed=params.seed,
        engine=params.engine,
        acceptance_criterion=params.acceptance_criterion,
    )

    # Convert VRPPRoute warm-start to plain node lists
    init_node_routes: Optional[List[List[int]]] = None
    if initial_routes:
        init_node_routes = [r.nodes for r in initial_routes if r.nodes]

    effective_pool = pool if pool is not None else RoutePool()

    solver = _PoolHarvestingALNS(
        dist_matrix,
        wastes,
        capacity,
        R,
        C,
        patched_params,
        mandatory_nodes,
        recorder=recorder,
        pool=effective_pool,
    )

    # Harvest the initial warm-start routes immediately
    if init_node_routes:
        solver._record_routes(init_node_routes)

    best_node_routes, best_profit, best_cost = solver.solve(
        initial_solution=init_node_routes,
    )

    # Convert best solution to VRPPRoute list
    best_routes: List[VRPPRoute] = []
    for r in best_node_routes:
        if not r:
            continue
        path = [0] + r + [0]
        cost = C * sum(dist_matrix[path[i]][path[i + 1]] for i in range(len(path) - 1))
        revenue = R * sum(wastes.get(n, 0.0) for n in r)
        load = sum(wastes.get(n, 0.0) for n in r)
        best_routes.append(
            VRPPRoute(
                nodes=list(r),
                profit=revenue - cost,
                revenue=revenue,
                cost=cost,
                load=load,
                source="alns_best",
            )
        )

    elapsed = time.perf_counter() - t0
    logger.info(
        "[ALNS] elapsed=%.1fs  profit=%.4f  pool_size=%d",
        elapsed,
        best_profit,
        len(effective_pool),
    )
    return best_routes, best_profit
