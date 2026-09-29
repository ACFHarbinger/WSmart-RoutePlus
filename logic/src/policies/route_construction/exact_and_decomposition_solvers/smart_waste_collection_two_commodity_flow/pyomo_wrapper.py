r"""Pyomo implementation of the SWC-TCF algorithm.

Serves as the high-level Algebraic Modeling Language (AML) baseline.

Attributes:
    _run_pyomo_tcf_optimizer: Solves TCF using Pyomo.

Example:
    >>> res = _run_pyomo_tcf_optimizer(bins, dist, values, ids, mandatory)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pyomo.environ as pyo
from numpy.typing import NDArray

from logic.src.constants.routing import MIP_GAP

from ._route_extraction import extract_depot_delimited_route
from .params import MAX_ARC_DISTANCE_KM


def _run_pyomo_tcf_optimizer(  # noqa: C901
    bins: NDArray[np.float64],
    distance_matrix: List[List[float]],
    values: Dict[str, float],
    binsids: List[int],
    mandatory_nodes: List[int],
    number_vehicles: int = 1,
    time_limit: int = 60,
    solver_id: str = "scip",
    seed: int = 42,
    dual_values: Optional[Dict[int, float]] = None,
) -> Tuple[List[int], float, float]:
    """Builds and solves the SWC-TCF using Pyomo.

    Args:
        bins (NDArray[np.float64]): Array of bin fill levels.
        distance_matrix (List[List[float]]): Distance matrix between nodes.
        values (Dict[str, float]): Problem parameters (Omega, psi, Q, R, C; percent fill units).
        binsids (List[int]): Global identifiers for bins.
        mandatory_nodes (List[int]): IDs of bins that must be collected.
        number_vehicles (int): Maximum number of vehicles.
        time_limit (int): Solver time limit in seconds.
        solver_id (str): Pyomo solver backend ID ('scip', 'gurobi', etc.).
        seed (int): Random seed for reproducibility.
        dual_values (Optional[Dict[int, float]]): Dual values for pricing subproblems.

    Returns:
        Tuple[List[int], float, float]: (route, profit, cost)
    """
    # 1. Parameter Extraction
    Omega, psi = values["Omega"], values["psi"]
    Q, R, C = values["Q"], values["R"], values["C"]

    n_bins = len(bins)
    nodes = list(range(n_bins + 1))
    idx_deposito = 0
    nodes_real = [i for i in nodes if i != idx_deposito]

    enchimentos = np.insert(bins, 0, 0.0)
    # Percent fill, as in gurobi.py: the adapter passes Q in percent points and R in
    # EUR per percent point, so a kg conversion here mixed unit systems.
    S_dict = {i: float(enchimentos[i]) for i in nodes}

    pure_binsids = binsids[1:] if len(binsids) == n_bins + 1 else binsids
    criticos_dict = {0: False}
    for i, bin_id in enumerate(pure_binsids, 1):
        criticos_dict[i] = bin_id in mandatory_nodes

    max_dist = MAX_ARC_DISTANCE_KM

    # 2. Pyomo Model Initialization
    model = pyo.ConcreteModel(name="SWC_TCF_Pyomo")
    model.V = pyo.Set(initialize=nodes)
    model.V_real = pyo.Set(initialize=nodes_real)

    def valid_arcs_rule(m, i, j):
        """Rule to filter out invalid or excessively long arcs.

        Args:
            m (pyo.ConcreteModel): The Pyomo model instance.
            i (int): Tail node index.
            j (int): Head node index.

        Returns:
            bool: True if the arc is valid and within max_dist.
        """
        return i != j and distance_matrix[i][j] <= max_dist

    # A filter-only set is never enumerated (it stayed empty); give it the candidates.
    model.A = pyo.Set(within=model.V * model.V, initialize=[(i, j) for i in nodes for j in nodes], filter=valid_arcs_rule)

    # Variables
    model.x = pyo.Var(model.A, within=pyo.Binary)
    model.f = pyo.Var(model.A, within=pyo.NonNegativeReals)
    model.h = pyo.Var(model.A, within=pyo.NonNegativeReals)
    model.g = pyo.Var(model.V, within=pyo.Binary)

    max_trucks = number_vehicles if number_vehicles > 0 else n_bins
    model.k_var = pyo.Var(within=pyo.Integers, bounds=(0, max_trucks))

    # 3. Constraints
    def cap_match_rule(m, i, j):
        """Ensures the total flow (waste + empty) matches vehicle capacity if used.

        Args:
            m (pyo.ConcreteModel): The Pyomo model instance.
            i (int): Tail node index.
            j (int): Head node index.

        Returns:
            pyo.Expression: Equality constraint expression.
        """
        return m.f[i, j] + m.h[i, j] == Q * m.x[i, j]

    model.cap_match = pyo.Constraint(model.A, rule=cap_match_rule)

    def flow_waste_rule(m, i):
        """Conservation rule for the waste commodity (commodity 1).

        Args:
            m (pyo.ConcreteModel): The Pyomo model instance.
            i (int): Node index.

        Returns:
            pyo.Expression: Conservation constraint expression.
        """
        inflow = sum(m.f[j, i] for j in m.V if (j, i) in m.A)
        outflow = sum(m.f[i, j] for j in m.V if (i, j) in m.A)
        return outflow - inflow == S_dict[i] * m.g[i]

    model.flow_waste = pyo.Constraint(model.V_real, rule=flow_waste_rule)

    def flow_empty_rule(m, i):
        """Conservation rule for the empty space commodity (commodity 2).

        Args:
            m (pyo.ConcreteModel): The Pyomo model instance.
            i (int): Node index.

        Returns:
            pyo.Expression: Conservation constraint expression.
        """
        inflow = sum(m.h[j, i] for j in m.V if (j, i) in m.A)
        outflow = sum(m.h[i, j] for j in m.V if (i, j) in m.A)
        return inflow - outflow == S_dict[i] * m.g[i]

    model.flow_empty = pyo.Constraint(model.V_real, rule=flow_empty_rule)

    model.depot_waste_in = pyo.Constraint(
        expr=sum(model.f[i, 0] for i in model.V_real if (i, 0) in model.A)
        == sum(S_dict[i] * model.g[i] for i in model.V_real)
    )
    model.depot_empty_out = pyo.Constraint(
        expr=sum(model.h[0, j] for j in model.V_real if (0, j) in model.A) == Q * model.k_var
    )
    depot_out_arcs = [j for j in model.V_real if (0, j) in model.A]
    if depot_out_arcs:
        model.depot_waste_out = pyo.Constraint(expr=sum(model.f[0, j] for j in depot_out_arcs) == 0)
    # With no depot arcs (all cut by MAX_ARC_DISTANCE_KM) the balance is 0 == 0, which pyomo rejects as a
    # trivial Boolean; skipping it leaves the model infeasible or empty, like the native backends.

    model.vehicle_count = pyo.Constraint(
        expr=sum(model.x[0, j] for j in model.V_real if (0, j) in model.A) == model.k_var
    )

    def route_in_rule(m, j):
        """Ensures that if a node is collected, it has exactly one incoming arc.

        Args:
            m (pyo.ConcreteModel): The Pyomo model instance.
            j (int): Node index.

        Returns:
            pyo.Expression: Degree constraint expression.
        """
        return sum(m.x[i, j] for i in m.V if (i, j) in m.A) == m.g[j]

    model.route_in = pyo.Constraint(model.V_real, rule=route_in_rule)

    def route_out_rule(m, j):
        """Ensures that if a node is collected, it has exactly one outgoing arc.

        Args:
            m (pyo.ConcreteModel): The Pyomo model instance.
            j (int): Node index.

        Returns:
            pyo.Expression: Degree constraint expression.
        """
        return sum(m.x[j, k] for k in m.V if (j, k) in m.A) == m.g[j]

    model.route_out = pyo.Constraint(model.V_real, rule=route_out_rule)

    # Mandatory & Pre-assignments
    model.forced_visits = pyo.ConstraintList()
    for i in nodes_real:
        if criticos_dict[i] or enchimentos[i] >= psi * 100:
            model.forced_visits.add(model.g[i] == 1)

    # 4. Objective Function
    def obj_rule(m):
        """Calculates the objective value (profit - routing cost - vehicle cost).

        Args:
            m (pyo.ConcreteModel): The Pyomo model instance.

        Returns:
            pyo.Expression: Maximization objective expression.
        """
        if dual_values:
            pi_0 = dual_values.get(0, 0.0)
            profit = sum((R * S_dict[i] - dual_values.get(i, 0.0)) * m.g[i] for i in m.V_real)
            cost = C * sum(m.x[i, j] * distance_matrix[i][j] for i, j in m.A)
            return profit - cost - (pi_0 * m.k_var)
        else:
            profit = R * sum(S_dict[i] * m.g[i] for i in m.V_real)
            cost = C * sum(m.x[i, j] * distance_matrix[i][j] for i, j in m.A)
            return profit - cost - (Omega * m.k_var)

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    # 5. Optimization
    opt = pyo.SolverFactory(solver_id)
    if solver_id == "gurobi":
        opt.options["Seed"] = seed
        opt.options["MIPGap"] = MIP_GAP
    elif solver_id == "scip":
        # SCIP uses randomseedshift to offset its default internal seed
        opt.options["randomization/randomseedshift"] = seed
        opt.options["limits/gap"] = MIP_GAP
    elif solver_id in ["appsi_highs", "highs"]:
        opt.options["random_seed"] = seed
        opt.options["mip_rel_gap"] = MIP_GAP
        opt.options["time_limit"] = float(time_limit)

    if solver_id == "gurobi":
        opt.options["TimeLimit"] = time_limit
    elif solver_id == "scip":
        opt.options["limits/time"] = time_limit

    def _has_solution(res) -> bool:
        tc = res.solver.termination_condition
        status_ok = pyo.check_optimal_termination(res) or (
            tc in (pyo.TerminationCondition.maxTimeLimit, pyo.TerminationCondition.feasible)
            and res.solver.status != pyo.SolverStatus.error
        )
        # A time limit hit before any incumbent reports maxTimeLimit with no solution to load.
        return status_ok and len(getattr(res, "solution", [])) > 0

    from logic.src.pipeline.simulations.solver_status import format_backend_status, note_solver_status

    results = opt.solve(model, tee=False, load_solutions=False)
    note_solver_status(format_backend_status("pyomo", results.solver.termination_condition))
    if results.solver.termination_condition == pyo.TerminationCondition.infeasible and len(model.forced_visits) > 0:
        print(f"[WARN] Pyomo TCF: {len(model.forced_visits)} forced visits are infeasible together; re-solving without forcing.")
        model.forced_visits.deactivate()
        results = opt.solve(model, tee=False, load_solutions=False)
        note_solver_status(format_backend_status("pyomo", results.solver.termination_condition), append=True)
    if results.solver.termination_condition == pyo.TerminationCondition.infeasible:
        raise RuntimeError(f"SWC-TCF model is infeasible (Pyomo/{solver_id}).")

    # 6. Parse Results
    loaded = False
    if _has_solution(results):
        try:
            model.solutions.load_from(results)
            loaded = True
        except ValueError as exc:  # e.g. "Cannot load a SolverResults object with bad status: aborted"
            print(f"[WARN] Pyomo TCF ({solver_id}): no loadable incumbent ({exc}).")
    if loaded:
        id_map = {0: 0}
        for i, bin_id in enumerate(pure_binsids, 1):
            id_map[i] = bin_id

        arcos_ativos = [(i, j) for (i, j) in model.A if pyo.value(model.x[i, j]) > 0.5]
        route = extract_depot_delimited_route(arcos_ativos, id_map)

        profit = pyo.value(model.obj)
        cost = sum([pyo.value(model.x[i, j]) * distance_matrix[i][j] for i, j in model.A])
        return route, profit, cost

    print(f"[WARN] Pyomo TCF ({solver_id}) could not find a feasible solution.")
    return [0, 0], 0.0, 0.0
