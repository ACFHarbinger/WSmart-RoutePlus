r"""Google OR-Tools (MPSolver) implementation of the SWC-TCF algorithm.

Supports Gurobi, SCIP, CPLEX, and HiGHS backends natively.

Attributes:
    _run_ortools_tcf_optimizer: Solves TCF using OR-Tools MPSolver.

Example:
    >>> res = _run_ortools_tcf_optimizer(bins, dist, values, ids, mandatory)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from ortools.linear_solver import pywraplp

from logic.src.constants.routing import MIP_GAP

from ._route_extraction import extract_depot_delimited_route
from ._tcf_data import build_tcf_data


def _run_ortools_tcf_optimizer(  # noqa: C901
    bins: NDArray[np.float64],
    distance_matrix: List[List[float]],
    values: Dict[str, float],
    binsids: List[int],
    mandatory_nodes: List[int],
    number_vehicles: int = 1,
    time_limit: int = 60,
    solver_id: str = "SCIP",  # Can be 'GUROBI', 'SCIP', 'HIGHS', 'CPLEX'
    seed: int = 42,
    dual_values: Optional[Dict[int, float]] = None,
) -> Tuple[List[int], float, float]:
    """Solves the SWC-TCF using Google OR-Tools MPSolver.

    Args:
        bins (NDArray[np.float64]): Array of bin fill levels.
        distance_matrix (List[List[float]]): Distance matrix between nodes.
        values (Dict[str, float]): Problem parameters (Omega, psi, Q, R, C; percent fill units).
        binsids (List[int]): Global identifiers for bins.
        mandatory_nodes (List[int]): IDs of bins that must be collected.
        number_vehicles (int): Maximum number of vehicles.
        time_limit (int): Solver time limit in seconds.
        solver_id (str): MPSolver backend ID ('SCIP', 'GUROBI', etc.).
        seed (int): Random seed for reproducibility.
        dual_values (Optional[Dict[int, float]]): Dual values for pricing subproblems.

    Returns:
        Tuple[List[int], float, float]: (route, profit, cost)
    """
    # Initialize the requested solver backend
    solver = pywraplp.Solver.CreateSolver(solver_id)
    if not solver:
        print(f"[ERROR] Could not create OR-Tools solver with backend: {solver_id}")
        return [0, 0], 0.0, 0.0

    # 1. Gurobi Backend
    if solver_id == "GUROBI":
        # Gurobi parameter strings are usually "ParamName Value"
        solver.SetSolverSpecificParametersAsString(f"Seed {seed}")

    # 2. SCIP Backend
    elif solver_id == "SCIP":
        # SCIP parameter strings are usually "path/to/param = value"
        solver.SetSolverSpecificParametersAsString(f"randomization/randomseedshift = {seed}")

    # 3. HiGHS Backend (Not natively accessible via string params in older OR-Tools)
    # If using HiGHS, you may need to rely on the C++ API or check the specific
    # OR-Tools version documentation for HiGHS parameter routing.

    # One gap policy across backends (best-effort string routing).
    try:
        if solver_id == "GUROBI":
            solver.SetSolverSpecificParametersAsString(f"MIPGap {MIP_GAP}")
        elif solver_id == "SCIP":
            solver.SetSolverSpecificParametersAsString(f"limits/gap = {MIP_GAP}")
        elif solver_id == "HIGHS":
            solver.SetSolverSpecificParametersAsString(f"mip_rel_gap = {MIP_GAP}")
    except Exception as exc:  # backend rejected the parameter string
        print(f"[WARN][VRPP-OR-Tools] could not set MIP gap on {solver_id}: {exc}")

    solver.SetTimeLimit(int(float(time_limit) * 1000))  # OR-Tools expects milliseconds

    # 1. Parameter Extraction
    # Shared preparation (identical across all three backends).
    d = build_tcf_data(bins, distance_matrix, values, binsids, mandatory_nodes, number_vehicles)
    Omega, psi = d.Omega, d.psi
    Q, R, C = d.Q, d.R, d.C
    nodes, nodes_real = d.nodes, d.nodes_real
    S_dict, criticos_dict = d.S_dict, d.criticos_dict
    valid_arcs = d.valid_arcs

    # 2. Variable Definitions
    x = {}  # Arc selection (binary)
    f = {}  # Waste flow (continuous)
    h = {}  # Empty capacity flow (continuous)

    for i, j in valid_arcs:
        x[i, j] = solver.BoolVar(f"x_{i}_{j}")
        f[i, j] = solver.NumVar(0.0, solver.infinity(), f"f_{i}_{j}")
        h[i, j] = solver.NumVar(0.0, solver.infinity(), f"h_{i}_{j}")

    g = {i: solver.BoolVar(f"g_{i}") for i in nodes}

    k_var = solver.IntVar(0, d.max_trucks, "k_var")

    # 3. Constraints
    # Capacity constraints on arcs
    for i, j in valid_arcs:
        solver.Add(f[i, j] + h[i, j] == Q * x[i, j])

    # Flow balance for Waste & Empty Capacity
    for i in nodes_real:
        # Waste
        inflow_f = solver.Sum(f[j, i] for j in nodes if (j, i) in valid_arcs)
        outflow_f = solver.Sum(f[i, j] for j in nodes if (i, j) in valid_arcs)
        solver.Add(outflow_f - inflow_f == S_dict[i] * g[i])

        # Empty Capacity
        inflow_h = solver.Sum(h[j, i] for j in nodes if (j, i) in valid_arcs)
        outflow_h = solver.Sum(h[i, j] for j in nodes if (i, j) in valid_arcs)
        solver.Add(inflow_h - outflow_h == S_dict[i] * g[i])

    # Depot balance
    solver.Add(
        solver.Sum(f[i, 0] for i in nodes_real if (i, 0) in valid_arcs)
        == solver.Sum(S_dict[i] * g[i] for i in nodes_real)
    )
    solver.Add(solver.Sum(h[0, j] for j in nodes_real if (0, j) in valid_arcs) == Q * k_var)
    solver.Add(solver.Sum(f[0, j] for j in nodes_real if (0, j) in valid_arcs) == 0)

    # Route Continuity & Vehicle Count
    solver.Add(solver.Sum(x[0, j] for j in nodes_real if (0, j) in valid_arcs) == k_var)

    for j in nodes_real:
        # In-degree equals out-degree equals g[j]
        solver.Add(solver.Sum(x[i, j] for i in nodes if (i, j) in valid_arcs) == g[j])
        solver.Add(solver.Sum(x[j, k] for k in nodes if (j, k) in valid_arcs) == g[j])

    # Mandatory & Pre-assignments
    forced = [solver.Add(g[i] == 1) for i in nodes_real if criticos_dict[i] or S_dict[i] >= psi * 100]

    # 4. Objective Function
    objective = solver.Objective()
    if dual_values:
        # Reduced cost optimization for B&P integration
        pi_0 = dual_values.get(0, 0.0)
        for i in nodes_real:
            objective.SetCoefficient(g[i], R * S_dict[i] - dual_values.get(i, 0.0))
        for i, j in valid_arcs:
            objective.SetCoefficient(x[i, j], -C * distance_matrix[i][j])
        objective.SetCoefficient(k_var, -pi_0)
    else:
        # Standard objective
        for i in nodes_real:
            objective.SetCoefficient(g[i], R * S_dict[i])
        for i, j in valid_arcs:
            objective.SetCoefficient(x[i, j], -C * distance_matrix[i][j])
        objective.SetCoefficient(k_var, -Omega)

    objective.SetMaximization()

    # 5. Optimization & Parsing
    from logic.src.pipeline.simulations.solver_status import format_backend_status, note_solver_status

    status = solver.Solve()
    note_solver_status(format_backend_status("ortools", status))
    if status == pywraplp.Solver.INFEASIBLE and forced:
        print(f"[WARN] OR-Tools TCF: {len(forced)} forced visits are infeasible together; re-solving without forcing.")
        for ct in forced:
            ct.SetBounds(-solver.infinity(), solver.infinity())
        status = solver.Solve()
        note_solver_status(format_backend_status("ortools", status), append=True)
    if status == pywraplp.Solver.INFEASIBLE:
        raise RuntimeError("SWC-TCF model is infeasible (OR-Tools).")

    if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:

        arcos_ativos = [(i, j) for (i, j) in valid_arcs if x[i, j].solution_value() > 0.5]
        route = extract_depot_delimited_route(arcos_ativos, d.id_map)

        profit = objective.Value()
        cost = sum([x[i, j].solution_value() * distance_matrix[i][j] for i, j in valid_arcs])

        return route, profit, cost

    else:
        print("[WARN] OR-Tools TCF could not find a feasible solution.")
        return [0, 0], 0.0, 0.0
