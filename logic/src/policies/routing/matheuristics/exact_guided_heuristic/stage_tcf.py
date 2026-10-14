r"""Stage 1 — Two-Commodity Flow MIP (SWC-TCF formulation).

Wraps the existing ``_run_gurobi_optimizer`` TCF implementation and converts
its arc-flow solution into a list of ``VRPPRoute`` objects that are fed into
the shared route pool.  The stage also returns the solved Gurobi model so
callers can read LP-relaxation duals for BPC warm-starting (a future
extension; the model is returned but not consumed by the current pipeline).

References:
    Baldacci, Hadjiconstantinou, Mingozzi (2004).
    "An exact algorithm for the capacitated VRP based on a two-commodity
    network flow formulation." Operations Research, 52(5), 723–738.

Attributes:
    run_tcf_stage: Main entry point for Stage 1.

Example:
    >>> from stage_tcf import run_tcf_stage
    >>> routes, profit, model = run_tcf_stage(bins, dist, env, values,
    ...                                        binsids, mandatory, n_veh,
    ...                                        time_limit=30.0, seed=42)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum
from numpy.typing import NDArray

from logic.src.constants.routing import HEURISTICS_RATIO, MIP_GAP, NODEFILE_START_GB

from .route_pool import RoutePool, VRPPRoute

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_route(
    nodes: List[int],
    dist: List[List[float]],
    S: Dict[int, float],
    R: float,
    C: float,
    source: str = "tcf",
) -> VRPPRoute:
    """Construct a VRPPRoute from a local customer-node list.

    Args:
        nodes: List of nodes.
        dist: Distance matrix.
        S: Dictionary of wastes.
        R: Revenue factor.
        C: Profit factor.
        source: Source string.

    Returns:
        VRPPRoute: Route object.
    """
    path = [0] + nodes + [0]
    cost = C * sum(dist[path[i]][path[i + 1]] for i in range(len(path) - 1))
    revenue = R * sum(S.get(n, 0.0) for n in nodes)
    load = sum(S.get(n, 0.0) for n in nodes)
    return VRPPRoute(nodes=list(nodes), profit=revenue - cost, revenue=revenue, cost=cost, load=load, source=source)


def _decompose_arcs(
    arcs: List[Tuple[int, int]],
    dist: List[List[float]],
    S: Dict[int, float],
    R: float,
    C: float,
) -> List[VRPPRoute]:
    """Decompose an active arc set into depot-rooted tours via greedy DFS.

    Args:
        arcs: List of arcs.
        dist: Distance matrix.
        S: Dictionary of wastes.
        R: Revenue factor.
        C: Profit factor.

    Returns:
        List of VRPPRoute.
    """
    routes: List[VRPPRoute] = []
    visited: set = set()

    while True:
        tour_arcs: List[Tuple[int, int]] = []
        curr = 0
        while True:
            nexts = [j for (i, j) in arcs if i == curr and (i, j) not in visited]
            if not nexts:
                break
            nxt = nexts[0]
            visited.add((curr, nxt))
            tour_arcs.append((curr, nxt))
            curr = nxt
            if curr == 0:
                break
        if not tour_arcs:
            break
        cust = [j for (_, j) in tour_arcs if j != 0]
        if cust:
            routes.append(_build_route(cust, dist, S, R, C, source="tcf"))

    return routes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_tcf_stage(  # noqa: C901
    bins: NDArray[np.float64],
    dist: List[List[float]],
    env: Optional[gp.Env],
    values: Dict,
    binsids: List[int],
    mandatory: List[int],
    n_vehicles: int,
    time_limit: float,
    seed: int = 42,
    dual_values: Optional[Dict[int, float]] = None,
    pool: Optional[RoutePool] = None,
) -> Tuple[List[VRPPRoute], float, Optional[gp.Model]]:
    """Solve the SWC-TCF MIP and extract routes into the shared pool.

    This is a faithful replica of the original ``_run_gurobi_optimizer`` with
    two additions:
    1. Routes are returned as ``VRPPRoute`` objects (not just node lists).
    2. The solved Gurobi model is returned for optional LP dual extraction.

    Args:
        bins:        Array of bin fill levels (local 1-based indexing).
        dist:        Full distance matrix including depot at index 0.
        env:         Shared Gurobi environment (or None for default env).
        values:      Problem parameters dict (Omega, delta, psi, Q, R, B, C, V).
        binsids:     Global bin identifiers (length n_bins or n_bins+1).
        mandatory:   Global IDs of bins that must be collected.
        n_vehicles:  Fleet size K.
        time_limit:  MIP wall-clock limit in seconds.
        seed:        Gurobi random seed.
        dual_values: Optional reduced-cost dual dict for pricing-phase objective.
        pool:        Optional shared RoutePool; if provided, routes are added.

    Returns:
        (routes, obj_val, model)
            routes   — list of VRPPRoute extracted from the MIP solution.
            obj_val  — MIP objective value (0.0 if no solution found).
            model    — solved Gurobi model (or None on error).
    """
    Omega = values["Omega"]
    delta = values["delta"]
    psi = values["psi"]
    Q = values["Q"]
    R = values["R"]
    C = values["C"]

    n_bins = len(bins)
    nodes = list(range(n_bins + 1))
    depot = 0
    nodes_real = [i for i in nodes if i != depot]

    enchimentos = np.insert(bins, 0, 0.0)
    S_dict = {i: float(enchimentos[i]) for i in nodes}

    pure_binsids = binsids[1:] if len(binsids) == n_bins + 1 else binsids
    criticos: Dict[int, bool] = {0: False}
    for idx, bid in enumerate(pure_binsids, 1):
        criticos[idx] = bid in mandatory

    max_d = 6_000_000.0
    pairs = [(i, j) for i in nodes for j in nodes if i != j and dist[i][j] <= max_d]

    try:
        mdl = gp.Model("TCF_Stage", env=env) if env else gp.Model("TCF_Stage")
    except gp.GurobiError as exc:
        logger.error("[TCF] Could not create Gurobi model: %s", exc)
        return [], 0.0, None

    mdl.Params.LogToConsole = 0
    mdl.Params.Seed = seed

    x = mdl.addVars(pairs, vtype=GRB.BINARY, name="x")
    g = mdl.addVars(nodes, vtype=GRB.BINARY, name="g")
    f = mdl.addVars(pairs, vtype=GRB.CONTINUOUS, lb=0, name="f")
    h = mdl.addVars(pairs, vtype=GRB.CONTINUOUS, lb=0, name="h")
    k_var = mdl.addVar(lb=0, vtype=GRB.INTEGER, name="k_var")

    # ── Constraints (identical to original SWC-TCF) ─────────────────────
    for i, j in pairs:
        mdl.addConstr(f[i, j] + h[i, j] == Q * x[i, j])

    for i in nodes_real:
        mdl.addConstr(
            quicksum(f[i, j] for j in nodes if (i, j) in f) - quicksum(f[j, i] for j in nodes if (j, i) in f)
            == S_dict[i] * g[i]
        )
        mdl.addConstr(
            quicksum(h[j, i] for j in nodes if (j, i) in h) - quicksum(h[i, j] for j in nodes if (i, j) in h)
            == S_dict[i] * g[i]
        )

    mdl.addConstr(
        quicksum(f[i, 0] for i in nodes_real if (i, 0) in f) == quicksum(S_dict[i] * g[i] for i in nodes_real)
    )
    mdl.addConstr(quicksum(h[0, j] for j in nodes_real if (0, j) in h) == Q * k_var)
    mdl.addConstr(quicksum(f[0, j] for j in nodes_real if (0, j) in f) == 0)

    n_veh = n_vehicles if n_vehicles > 0 else len(binsids)
    mdl.addConstr(k_var <= n_veh)
    mdl.addConstr(quicksum(x[depot, j] for j in nodes_real if (depot, j) in x) == k_var)
    mdl.addConstr(quicksum(x[j, depot] for j in nodes_real if (j, depot) in x) == k_var)
    for j in nodes_real:
        if (depot, j) in x:
            mdl.addConstr(x[depot, j] <= g[j])
        if (j, depot) in x:
            mdl.addConstr(x[j, depot] <= g[j])

    n_crit = len([i for i in nodes_real if criticos[i]])
    if n_crit > 0:
        mdl.addConstr(quicksum(g[i] for i in nodes_real if criticos[i]) >= n_crit - len(nodes_real) * delta)
    for i in nodes_real:
        if criticos[i] or enchimentos[i] >= psi * 100:
            mdl.addConstr(g[i] == 1)

    for j in nodes_real:
        mdl.addConstr(quicksum(x[i, j] for i in nodes if (i, j) in x) == g[j])
        mdl.addConstr(quicksum(x[j, k] for k in nodes if (j, k) in x) == g[j])

    # ── Objective ────────────────────────────────────────────────────────
    if dual_values:
        pi_0 = dual_values.get(depot, 0.0)
        mdl.setObjective(
            quicksum((R * S_dict[i] - dual_values.get(i, 0.0)) * g[i] for i in nodes_real)
            - 0.5 * C * quicksum(x[i, j] * dist[i][j] for i, j in pairs)
            - pi_0 * k_var,
            GRB.MAXIMIZE,
        )
    else:
        mdl.setObjective(
            R * quicksum(S_dict[i] * g[i] for i in nodes_real)
            - 0.5 * C * quicksum(x[i, j] * dist[i][j] for i, j in pairs)
            - Omega * k_var,
            GRB.MAXIMIZE,
        )

    # ── Solver parameters (mirror original SWC-TCF) ──────────────────────
    mdl.Params.MIPFocus = 1
    mdl.Params.Heuristics = HEURISTICS_RATIO
    mdl.Params.Threads = 0
    mdl.Params.Cuts = 3
    mdl.Params.CliqueCuts = 2
    mdl.Params.CoverCuts = 2
    mdl.Params.FlowCoverCuts = 2
    mdl.Params.GUBCoverCuts = 2
    mdl.Params.Presolve = 1
    mdl.Params.NodefileStart = NODEFILE_START_GB
    mdl.Params.OutputFlag = 0
    mdl.setParam("MIPGap", MIP_GAP)
    if time_limit > 0:
        mdl.Params.TimeLimit = time_limit

    mdl.optimize()

    routes: List[VRPPRoute] = []
    obj_val = 0.0

    if mdl.SolCount > 0:
        obj_val = float(mdl.ObjVal)
        arcs = [(i, j) for (i, j) in x.keys() if x[i, j].X > 0.5 and i != j]
        routes = _decompose_arcs(arcs, dist, S_dict, R, C)
        if pool is not None:
            pool.add_all(routes)
        logger.info(
            "[TCF] obj=%.4f  routes=%d  pool_size=%s",
            obj_val,
            len(routes),
            len(pool) if pool else "n/a",
        )
    else:
        logger.warning("[TCF] No feasible solution found within %.1fs", time_limit)

    return routes, obj_val, mdl
