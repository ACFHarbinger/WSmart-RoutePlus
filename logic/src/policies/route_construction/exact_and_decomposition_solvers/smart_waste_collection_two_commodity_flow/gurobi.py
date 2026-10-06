r"""Gurobi Solver for VRPP.

Attributes:
    _run_gurobi_optimizer: Solves the VRPP using a native Gurobi TCF formulation.

Example:
    >>> res = _run_gurobi_optimizer(bins, dist, env, values, ids, mandatory)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum
from numpy.typing import NDArray

from logic.src.constants.routing import HEURISTICS_RATIO, MIP_GAP, NODEFILE_START_GB

from ._route_extraction import extract_depot_delimited_route
from ._tcf_data import TCFData, build_tcf_data
from .params import SWCTCFParams


def _clarke_wright_trips(d: TCFData, distance_matrix: List[List[float]], visit: List[int]) -> Optional[List[List[int]]]:
    """Build Clarke-Wright trips over ``visit`` that are feasible in the TCF model.

    Args:
        d: Shared TCF model data (local node indices, depot 0).
        distance_matrix: Full kilometre matrix in local indexing.
        visit: Local indices of the bins to collect.

    Returns:
        Trips as lists of local node indices, or None when the trips would not
        be feasible in the model (a bin above capacity, a missing arc, or more
        trips than the fleet bound).
    """
    from logic.src.policies.route_construction.other_algorithms.capacitated_vehicle_routing_problem.clark_wright import (
        clarke_wright_solve,
    )

    if not visit or any(d.S_dict[i] > d.Q for i in visit):
        return None
    tour = clarke_wright_solve(np.asarray(distance_matrix, dtype=float), {i: d.S_dict[i] for i in visit}, d.Q, visit)
    trips: List[List[int]] = []
    current: List[int] = []
    for node in tour[1:]:
        if node == 0:
            if current:
                trips.append(current)
            current = []
        else:
            current.append(node)
    arcs = set(d.valid_arcs)
    for trip in trips:
        path = [0, *trip, 0]
        if any((a, b) not in arcs for a, b in zip(path, path[1:], strict=False)):
            return None
    if len(trips) > d.max_trucks:
        return None
    return trips


def _warm_starts(d: TCFData, distance_matrix: List[List[float]], forced_nodes: List[int]) -> List[List[List[int]]]:
    """Feasible MIP starts: Clarke-Wright trips over the forced bins and over every bin with waste.

    Used only when ``warm_start`` is enabled; the published model (Ramos et al.,
    2018) has no MIP start. At 350 bins and a 60 s limit Gurobi can end without
    any incumbent when only part of the bins is forced, and the day then
    collected nothing (#41). A forced-only start alone fixes that but anchors
    the search on a small plan; the collect-everything start supplies the
    multi-trip plan Gurobi finds on its good days. Gurobi keeps the better one
    and improves it.

    Args:
        d: Shared TCF model data.
        distance_matrix: Full kilometre matrix in local indexing.
        forced_nodes: Local indices whose visit is forced.

    Returns:
        Distinct feasible starts, each a list of trips.
    """
    starts: List[List[List[int]]] = []
    candidates = [forced_nodes, [i for i in d.nodes_real if d.S_dict[i] > 0]]
    for visit in candidates:
        if visit and all(i in visit for i in forced_nodes):
            trips = _clarke_wright_trips(d, distance_matrix, visit)
            if trips is not None and trips not in starts:
                starts.append(trips)
    return starts


def _plan_objective(d: TCFData, distance_matrix: List[List[float]], trips: List[List[int]]) -> Tuple[float, float]:
    """Return (profit, travel cost) of a trip plan under the standard objective."""
    cost = sum(distance_matrix[a][b] for trip in trips for a, b in zip([0, *trip], [*trip, 0], strict=False))
    profit = d.R * sum(d.S_dict[i] for trip in trips for i in trip) - d.C * cost - d.Omega * len(trips)
    return profit, cost


def _flag(value: object) -> bool:
    """Read a yaml/OmegaConf boolean that may arrive as a string."""
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _run_gurobi_optimizer(  # noqa: C901
    bins: NDArray[np.float64],
    distance_matrix: List[List[float]],
    env: Optional[gp.Env],
    values: Dict[str, float],
    binsids: List[int],
    mandatory: List[int],
    number_vehicles: int = 1,
    time_limit: int = 60,
    seed: int = 42,
    dual_values: Optional[Dict[int, float]] = None,
) -> Tuple[List[int], float, float]:
    """Solve the Vehicle Routing Problem with Profits using Two-Commodity Flow.

    Args:
        bins (NDArray[np.float64]): Array of bin fill levels.
        distance_matrix (List[List[float]]): Distance matrix between nodes.
        env (Optional[gp.Env]): Gurobi environment.
        values (Dict[str, float]): Problem parameters (Omega, psi, Q, R, B, C, V).
        binsids (List[int]): Global identifiers for bins.
        mandatory (List[int]): IDs of bins that must be collected.
        number_vehicles (int): Maximum number of vehicles.
        time_limit (int): Solver time limit in seconds.
        seed (int): Random seed for reproducibility.
        dual_values (Optional[Dict[int, float]]): Dual values for pricing subproblems.

    Returns:
        Tuple[List[int], float, float]: A tuple containing:
            - route: Sequence of node IDs starting and ending at depot (0).
            - profit: The total profit of the solution.
            - cost: The total travel cost of the solution.
    """
    # Validate before allocating a native model. sim.cpu_cores controls worker
    # processes, not Gurobi's per-solve thread count.
    threads = float(values.get("gurobi_threads", SWCTCFParams.gurobi_threads))
    memory_gb = float(values.get("gurobi_soft_mem_limit_gb", SWCTCFParams.gurobi_soft_mem_limit_gb))
    if not math.isfinite(threads) or threads < 1 or not threads.is_integer():
        raise ValueError("gurobi_threads must be a positive integer")
    if not math.isfinite(memory_gb) or memory_gb <= 0:
        raise ValueError("gurobi_soft_mem_limit_gb must be positive and finite")

    # Shared preparation (identical across all three backends).
    d = build_tcf_data(bins, distance_matrix, values, binsids, mandatory, number_vehicles)
    Omega, psi = d.Omega, d.psi
    Q, R, C = d.Q, d.R, d.C
    nodes, nodes_real = d.nodes, d.nodes_real
    S_dict, criticos_dict = d.S_dict, d.criticos_dict
    pares_viaveis = d.valid_arcs

    mdl = gp.Model("VRPP", env=env) if env else gp.Model("VRPP")
    try:
        # Do not loosen limits on a caller-owned environment. SoftMemLimit allows
        # solution extraction on MEM_LIMIT, unlike a hard allocation failure.
        inherited_threads = mdl.Params.Threads
        mdl.Params.Threads = min(int(threads), inherited_threads) if inherited_threads > 0 else int(threads)
        mdl.Params.SoftMemLimit = min(memory_gb, mdl.Params.SoftMemLimit)
        print(f"[INFO][VRPP-Gurobi] Threads={mdl.Params.Threads}, SoftMemLimit={mdl.Params.SoftMemLimit} GB")
        mdl.Params.Seed = seed
        x = mdl.addVars(pares_viaveis, vtype=GRB.BINARY, name="x")
        g = mdl.addVars(nodes, vtype=GRB.BINARY, name="g")

        # Two-Commodity Flow Variables
        # f_ij: flow of waste (commodity 1)
        # h_ij: flow of empty capacity (commodity 2)
        f = mdl.addVars(pares_viaveis, vtype=GRB.CONTINUOUS, lb=0, name="f")
        h = mdl.addVars(pares_viaveis, vtype=GRB.CONTINUOUS, lb=0, name="h")

        k_var = mdl.addVar(lb=0, vtype=GRB.INTEGER, name="k_var")

        # 1. Capacity constraints on arcs
        for i, j in pares_viaveis:
            mdl.addConstr(f[i, j] + h[i, j] == Q * x[i, j])

        # 2. Flow balance for waste (Commodity 1)
        # At each visited node i, outflow - inflow = waste[i]
        for i in nodes_real:
            mdl.addConstr(
                quicksum(f[i, j] for j in nodes if (i, j) in f) - quicksum(f[j, i] for j in nodes if (j, i) in f)
                == S_dict[i] * g[i]
            )

        # 3. Flow balance for empty capacity (Commodity 2)
        # At each visited node i, inflow - outflow = waste[i]
        for i in nodes_real:
            mdl.addConstr(
                quicksum(h[j, i] for j in nodes if (j, i) in h) - quicksum(h[i, j] for j in nodes if (i, j) in h)
                == S_dict[i] * g[i]
            )

        # 4. Depot balance
        # Total waste collected = inflow to depot
        mdl.addConstr(
            quicksum(f[i, 0] for i in nodes_real if (i, 0) in f) == quicksum(S_dict[i] * g[i] for i in nodes_real)
        )
        # Total empty capacity = outflow from depot (full capacity Q per vehicle)
        mdl.addConstr(quicksum(h[0, j] for j in nodes_real if (0, j) in h) == Q * k_var)
        # Load leaving depot is 0
        mdl.addConstr(quicksum(f[0, j] for j in nodes_real if (0, j) in f) == 0)

        mdl.addConstr(k_var <= d.max_trucks)

        mdl.addConstr(quicksum(x[0, j] for j in nodes_real if (0, j) in x) == k_var)
        mdl.addConstr(quicksum(x[j, 0] for j in nodes_real if (j, 0) in x) == k_var)

        for j in nodes_real:
            if (0, j) in x:
                mdl.addConstr(x[0, j] <= g[j])
            if (j, 0) in x:
                mdl.addConstr(x[j, 0] <= g[j])

        forced = [
            mdl.addConstr(g[i] == 1, name=f"forced_{i}")
            for i in nodes_real
            if criticos_dict[i] or S_dict[i] >= psi * 100
        ]

        for j in nodes_real:
            mdl.addConstr(quicksum(x[i, j] for i in nodes if (i, j) in x) == g[j])
            mdl.addConstr(quicksum(x[j, k] for k in nodes if (j, k) in x) == g[j])

        forced_nodes = [i for i in nodes_real if criticos_dict[i] or S_dict[i] >= psi * 100]
        # Optional, off by default: not part of the published model (#41).
        starts = _warm_starts(d, distance_matrix, forced_nodes) if _flag(values.get("warm_start", False)) else []
        if starts:
            # Complete MIP starts: every variable gets a value, so Gurobi checks
            # each start directly instead of repairing a partial assignment.
            all_vars = [*x.values(), *g.values(), *f.values(), *h.values()]
            mdl.NumStart = len(starts)
            for number, trips in enumerate(starts):
                mdl.Params.StartNumber = number
                mdl.setAttr("Start", all_vars, [0.0] * len(all_vars))
                start_vars, start_vals = [k_var], [float(len(trips))]
                for trip in trips:
                    load = 0.0
                    path = [0, *trip, 0]
                    for a, b in zip(path, path[1:], strict=False):
                        start_vars += [x[a, b], f[a, b], h[a, b]]
                        start_vals += [1.0, load, Q - load]
                        if b != 0:
                            start_vars.append(g[b])
                            start_vals.append(1.0)
                            load += S_dict[b]
                mdl.setAttr("Start", start_vars, start_vals)

        # Two-commodity flow handles subtour elimination and capacity automatically.
        # No need for the old 'f' based commodity flow here.

        if dual_values:
            # VRPP Pricing Phase: Maximize Reduced Cost = Profit - sum(π_i * g_i) - π_0 * k_var
            # Depot dual is usually at index 0 (representing the fleet limit constraint)
            pi_0 = dual_values.get(0, 0.0)
            mdl.setObjective(
                quicksum((R * S_dict[i] - dual_values.get(i, 0.0)) * g[i] for i in nodes_real)
                - C * quicksum(x[i, j] * distance_matrix[i][j] for i, j in pares_viaveis)
                - pi_0 * k_var,
                GRB.MAXIMIZE,
            )
        else:
            # Standard Objective
            mdl.setObjective(
                R * quicksum(S_dict[i] * g[i] for i in nodes_real)
                - C * quicksum(x[i, j] * distance_matrix[i][j] for i, j in pares_viaveis)
                - Omega * k_var,
                GRB.MAXIMIZE,
            )

        if not env:
            mdl.Params.MIPFocus = 1
            mdl.Params.Heuristics = HEURISTICS_RATIO
            mdl.Params.Cuts = 3
            mdl.Params.CliqueCuts = 2
            mdl.Params.CoverCuts = 2
            mdl.Params.FlowCoverCuts = 2
            mdl.Params.GUBCoverCuts = 2
            mdl.Params.Presolve = 1
            mdl.Params.NodefileStart = NODEFILE_START_GB
            mdl.Params.LogToConsole = 0
            mdl.Params.OutputFlag = 0
            mdl.setParam("MIPGap", MIP_GAP)

        if time_limit > 0:
            mdl.Params.TimeLimit = float(time_limit)

        profit = 0.0
        cost = 0.0
        from logic.src.pipeline.simulations.solver_status import format_backend_status, note_solver_status

        mdl.optimize()
        note_solver_status(format_backend_status("gurobi", mdl.Status))
        if mdl.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD) and forced:
            # Forcing every mandatory / over-psi bin can exceed the fleet's capacity.
            # Collect what is feasible instead of returning an empty day.
            print(
                f"[WARN][VRPP-Gurobi] {len(forced)} forced visits are infeasible together; re-solving without forcing."
            )
            mdl.remove(forced)
            mdl.optimize()
            note_solver_status(format_backend_status("gurobi", mdl.Status), append=True)
        if mdl.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            raise RuntimeError(f"SWC-TCF model is infeasible (Gurobi status {mdl.Status}).")
        if mdl.SolCount == 0:
            best = max(starts, key=lambda t: _plan_objective(d, distance_matrix, t)[0]) if starts else None
            # Without forced bins the empty plan (profit 0) is feasible and wins unless a start beats it.
            if (
                best is not None
                and not dual_values
                and (forced_nodes or _plan_objective(d, distance_matrix, best)[0] > 0)
            ):
                # The starts are feasible by construction; execute the best one
                # rather than skip the forced bins.
                trips = best
                profit, cost = _plan_objective(d, distance_matrix, trips)
                arcs = [(a, b) for trip in trips for a, b in zip([0, *trip], [*trip, 0], strict=False)]
                route = extract_depot_delimited_route(arcs, d.id_map)
                print(
                    f"[WARN][VRPP-Gurobi] No solution found (status {mdl.Status}); "
                    f"executing the best Clarke-Wright start ({len(trips)} trips)."
                )
                note_solver_status("fallback:clarke_wright", append=True)
                return route, profit, cost
            print(f"[WARN][VRPP-Gurobi] No solution found (status {mdl.Status}); the day collects nothing.")
            return [0, 0], 0.0, 0.0
        if mdl.SolCount > 0:
            arcos_ativos = [(i, j) for (i, j) in x.keys() if i != j and x[i, j].X > 0.5]
            route = extract_depot_delimited_route(arcos_ativos, d.id_map)

            if route == [0, 0]:
                # All-zero incumbent (or nothing collected): the shared empty-day shape.
                return [0, 0], 0.0, 0.0

            profit = mdl.ObjVal
            cost = sum([x[i, j].X * distance_matrix[i][j] for i, j in pares_viaveis])
            print(
                f"[INFO][VRPP-Gurobi] Profit: {profit}, Cost: {cost}, MIPGap: {mdl.Params.MIPGap}, Collected: {sum(1 for n in route if n != 0)}"
            )
            return route, profit, cost

        return [0, 0], 0.0, 0.0
    finally:
        # Native solver memory must be released before the next simulated day,
        # including build/optimization errors and early no-incumbent returns.
        mdl.dispose()
