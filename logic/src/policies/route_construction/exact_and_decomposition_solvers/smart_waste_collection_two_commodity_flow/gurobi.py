r"""Gurobi Solver for VRPP.

Attributes:
    _run_gurobi_optimizer: Solves the VRPP using a native Gurobi TCF formulation.

Example:
    >>> res = _run_gurobi_optimizer(bins, dist, env, values, ids, mandatory)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum
from numpy.typing import NDArray

from logic.src.constants.routing import HEURISTICS_RATIO, MIP_GAP, NODEFILE_START_GB

from ._route_extraction import extract_depot_delimited_route
from .params import MAX_ARC_DISTANCE_KM


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
    Omega, psi = values["Omega"], values["psi"]
    Q, R, _B, C, _V = values["Q"], values["R"], values["B"], values["C"], values["V"]

    n_bins = len(bins)
    nodes = list(range(n_bins + 1))
    idx_deposito = 0
    nodes_real = [i for i in nodes if i != idx_deposito]

    enchimentos = np.insert(bins, 0, 0.0)
    # Use percent fill levels directly. Revenue R is already scaled to Euro per 1% fill
    # by BaseRoutingPolicy, so (R * percent) gives Euro.
    S_dict = {i: float(enchimentos[i]) for i in nodes}

    # Normalize binsids to only include bin IDs (exclude depot if present)
    pure_binsids = binsids[1:] if len(binsids) == n_bins + 1 else binsids
    criticos_dict = {0: False}
    for i, bin_id in enumerate(pure_binsids, 1):
        criticos_dict[i] = bin_id in mandatory

    pares_viaveis = [
        (i, j) for i in nodes for j in nodes if i != j and distance_matrix[i][j] <= MAX_ARC_DISTANCE_KM
    ]

    mdl = gp.Model("VRPP", env=env) if env else gp.Model("VRPP")
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

    if number_vehicles == 0:
        number_vehicles = n_bins

    MAX_TRUCKS = number_vehicles
    mdl.addConstr(k_var <= MAX_TRUCKS)

    mdl.addConstr(quicksum(x[idx_deposito, j] for j in nodes_real if (idx_deposito, j) in x) == k_var)
    mdl.addConstr(quicksum(x[j, idx_deposito] for j in nodes_real if (j, idx_deposito) in x) == k_var)

    for j in nodes_real:
        if (idx_deposito, j) in x:
            mdl.addConstr(x[idx_deposito, j] <= g[j])
        if (j, idx_deposito) in x:
            mdl.addConstr(x[j, idx_deposito] <= g[j])

    forced = [
        mdl.addConstr(g[i] == 1, name=f"forced_{i}")
        for i in nodes_real
        if criticos_dict[i] or enchimentos[i] >= psi * 100
    ]

    for j in nodes_real:
        mdl.addConstr(quicksum(x[i, j] for i in nodes if (i, j) in x) == g[j])
        mdl.addConstr(quicksum(x[j, k] for k in nodes if (j, k) in x) == g[j])

    # Two-commodity flow handles subtour elimination and capacity automatically.
    # No need for the old 'f' based commodity flow here.

    if dual_values:
        # VRPP Pricing Phase: Maximize Reduced Cost = Profit - sum(π_i * g_i) - π_0 * k_var
        # Depot dual is usually at index 0 (representing the fleet limit constraint)
        pi_0 = dual_values.get(idx_deposito, 0.0)
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
        mdl.Params.Threads = 0
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
    mdl.optimize()
    if mdl.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD) and forced:
        # Forcing every mandatory / over-psi bin can exceed the fleet's capacity.
        # Collect what is feasible instead of returning an empty day.
        print(f"[WARN][VRPP-Gurobi] {len(forced)} forced visits are infeasible together; re-solving without forcing.")
        mdl.remove(forced)
        mdl.optimize()
    if mdl.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        raise RuntimeError(f"SWC-TCF model is infeasible (Gurobi status {mdl.Status}).")
    if mdl.SolCount == 0:
        print(f"[WARN][VRPP-Gurobi] No solution found (status {mdl.Status}); the day collects nothing.")
        return [0, 0], 0.0, 0.0
    if mdl.SolCount > 0:
        id_map = {0: 0}
        for i, bin_id in enumerate(pure_binsids, 1):
            id_map[i] = bin_id
        arcos_ativos = [(i, j) for (i, j) in x.keys() if i != j and x[i, j].X > 0.5]
        route = extract_depot_delimited_route(arcos_ativos, id_map)

        if route == [0, 0]:
            # All-zero incumbent (or nothing collected): the shared empty-day shape.
            return [0, 0], 0.0, 0.0

        profit = mdl.ObjVal
        cost = sum([x[i, j].X * distance_matrix[i][j] for i, j in pares_viaveis])
        print(f"[INFO][VRPP-Gurobi] Profit: {profit}, Cost: {cost}, MIPGap: {mdl.Params.MIPGap}, Collected: {len(route) - 2}")
        return route, profit, cost

    return [0, 0], 0.0, 0.0
