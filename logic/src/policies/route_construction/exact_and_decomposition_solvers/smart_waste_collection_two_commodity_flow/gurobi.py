r"""Gurobi solver for the Smart Waste Collection Routing model (SWC-TCF).

The default formulation is the published model: Ramos, Morais & Barbosa-Póvoa
(2018), "The smart waste collection routing problem: Alternative operational
management approaches", Expert Systems with Applications 103, Section 3.2.2,
eqs. (6), (8), (11)-(15) and (17)-(21), as implemented in
``notebooks/collectioncompare.ipynb`` (``policy_gurobi``): an undirected
two-commodity flow with a copy depot n+1.

Everything that is not in those sources is optional and off by default:

- ``formulation: directed``: the earlier directed two-commodity flow (separate
  waste / empty-space flows on directed arcs, no copy depot).
- ``depot_inflow: le``: the notebook's ``sum y_i0 <= Q k`` instead of eq. (15).
- ``solver_tuning``: MIPFocus/heuristics/cut/presolve settings and a 1% MIP gap.
- ``relax_forced_on_infeasible``: re-solve without forced visits when infeasible.
- ``link_depot_arcs``: ``x[0, j] <= g[j]``.
- ``max_arc_distance_km``: drop arcs longer than this.
- ``warm_start``: Clarke-Wright MIP starts and a no-incumbent fallback (#41).

Attributes:
    _run_gurobi_optimizer: Solves one day's SWC-TCF model with native Gurobi.

Example:
    >>> res = _run_gurobi_optimizer(bins, dist, env, values, ids, mandatory)
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    from logic.src.utils.routing.clark_wright import clarke_wright_solve

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
    cost = float(sum(distance_matrix[a][b] for trip in trips for a, b in zip([0, *trip], [*trip, 0], strict=False)))
    profit = d.R * sum(d.S_dict[i] for trip in trips for i in trip) - d.C * cost - d.Omega * len(trips)
    return profit, cost


def _flag(value: object) -> bool:
    """Read a yaml/OmegaConf boolean that may arrive as a string."""
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _optional_float(value: object) -> Optional[float]:
    """Read an optional numeric option (None, '', 'null' or 'none' mean unset)."""
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "null", "none")):
        return None
    return float(value)  # type: ignore[arg-type]


class _Built:
    """What a formulation builder hands back to the shared solve driver."""

    def __init__(
        self,
        x: Any,
        g: Any,
        k_var: Any,
        forced: List[Any],
        set_starts: Callable[[List[List[List[int]]]], None],
        extract_route: Callable[[], List[int]],
    ) -> None:
        self.x = x
        self.g = g
        self.k_var = k_var
        self.forced = forced
        self.set_starts = set_starts
        self.extract_route = extract_route


def _build_paper(  # noqa: C901
    mdl: gp.Model,
    d: TCFData,
    distance_matrix: List[List[float]],
    forced_nodes: List[int],
    number_vehicles: int,
    dual_values: Optional[Dict[int, float]],
    options: Dict[str, Any],
) -> _Built:
    """Published SWCR model (Ramos et al., 2018, eqs. 6, 8, 11-15, 17-21; collectioncompare ``policy_gurobi``).

    Node set ``I = {0, 1..n, n+1}``: the real depot 0, the bins and the copy depot n+1.
    Distances to and from the copy depot are the depot's row, as in the notebook
    (``d[i][n+1] = d[0][i]``, ``d[n+1][j] = d[0][j]``).
    """
    Q, R, C, Omega = d.Q, d.R, d.C, d.Omega
    S = d.S_dict
    bins = d.nodes_real
    copy = d.n_bins + 1
    nodes = [0, *bins, copy]

    def dist(i: int, j: int) -> float:
        if i == copy and j == copy:
            return float(distance_matrix[0][0])
        if j == copy:
            return float(distance_matrix[0][i])
        if i == copy:
            return float(distance_matrix[0][j])
        return float(distance_matrix[i][j])

    cutoff = options["max_arc_distance_km"]
    pairs = [
        (i, j)
        for i in nodes
        for j in nodes
        if i != j and (cutoff is None or (dist(i, j) <= cutoff and dist(j, i) <= cutoff))
    ]

    x = mdl.addVars(pairs, vtype=GRB.BINARY, name="x")
    y = mdl.addVars(pairs, vtype=GRB.CONTINUOUS, lb=0.0, name="y")
    g = mdl.addVars(bins, vtype=GRB.BINARY, name="g")
    k_var = mdl.addVar(lb=0, vtype=GRB.INTEGER, name="k")

    collected = quicksum(S[i] * g[i] for i in bins)
    # (12) outflow minus inflow at a visited bin is twice its waste
    for i in bins:
        mdl.addConstr(quicksum(y[i, j] - y[j, i] for j in nodes if (i, j) in y) == 2 * S[i] * g[i], name=f"flow_{i}")
    # (13) inflow of the copy depot is the waste collected
    mdl.addConstr(quicksum(y[i, copy] for i in bins if (i, copy) in y) == collected, name="copy_in")
    # (14) outflow of the copy depot is the fleet's residual capacity
    mdl.addConstr(quicksum(y[copy, j] for j in bins if (copy, j) in y) == Q * k_var - collected, name="copy_out")
    # (15) inflow of the real depot is the fleet capacity (notebook: <=)
    depot_in = quicksum(y[i, 0] for i in bins if (i, 0) in y)
    if options["depot_inflow"] == "le":
        mdl.addConstr(depot_in <= Q * k_var, name="depot_in")
    else:
        mdl.addConstr(depot_in == Q * k_var, name="depot_in")
    # (6) the real depot sends no load
    mdl.addConstr(quicksum(y[0, j] for j in bins if (0, j) in y) == 0, name="depot_out")
    # (18) two edges incident to each visited bin
    for j in bins:
        mdl.addConstr(quicksum(x[i, j] for i in nodes if (i, j) in x) == 2 * g[j], name=f"degree_{j}")
    # (8) the two flows on a used edge add up to the vehicle capacity
    for i, j in pairs:
        mdl.addConstr(y[i, j] + y[j, i] == Q * x[i, j], name=f"edge_{i}_{j}")
    if number_vehicles > 0:
        mdl.addConstr(k_var <= number_vehicles, name="fleet")
    if options["link_depot_arcs"]:
        for j in bins:
            if (0, j) in x:
                mdl.addConstr(x[0, j] <= g[j])
    # (17) and the selection's mandatory bins
    forced = [mdl.addConstr(g[i] == 1, name=f"forced_{i}") for i in forced_nodes]

    # (11) profit: revenue minus half the doubly counted distance minus the vehicle penalty
    travel = 0.5 * C * quicksum(x[i, j] * dist(i, j) for i, j in pairs)
    if dual_values:
        pi_0 = dual_values.get(0, 0.0)
        mdl.setObjective(
            quicksum((d.R * S[i] - dual_values.get(i, 0.0)) * g[i] for i in bins) - travel - pi_0 * k_var,
            GRB.MAXIMIZE,
        )
    else:
        mdl.setObjective(R * collected - travel - Omega * k_var, GRB.MAXIMIZE)

    def set_starts(starts: List[List[List[int]]]) -> None:
        all_vars = [*x.values(), *y.values(), *g.values()]
        mdl.NumStart = len(starts)
        for number, trips in enumerate(starts):
            mdl.Params.StartNumber = number
            mdl.setAttr("Start", all_vars, [0.0] * len(all_vars))
            start_vars, start_vals = [k_var], [float(len(trips))]
            for trip in trips:
                load = 0.0
                path = [0, *trip, copy]
                for a, b in zip(path, path[1:], strict=False):
                    if a != 0:
                        load += S[a]
                        start_vars.append(g[a])
                        start_vals.append(1.0)
                    start_vars += [x[a, b], x[b, a], y[a, b], y[b, a]]
                    start_vals += [1.0, 1.0, load, Q - load]
            mdl.setAttr("Start", start_vars, start_vals)

    def extract_route() -> List[int]:
        adjacent: Dict[int, List[int]] = {i: [] for i in nodes}
        for i, j in pairs:
            if i < j and x[i, j].X > 0.5:
                adjacent[i].append(j)
                adjacent[j].append(i)
        route = [0]
        for first in adjacent[0]:
            if first == copy:
                continue  # an unused vehicle: the zero-length depot edge
            prev, cur, steps = 0, first, 0
            while cur != copy and steps <= len(nodes):
                route.append(d.id_map[cur])
                nxt = [n for n in adjacent[cur] if n != prev]
                if not nxt:
                    break
                prev, cur, steps = cur, nxt[0], steps + 1
            route.append(0)
        return route if len(route) > 1 else [0, 0]

    return _Built(x, g, k_var, forced, set_starts, extract_route)


def _build_directed(  # noqa: C901
    mdl: gp.Model,
    d: TCFData,
    distance_matrix: List[List[float]],
    forced_nodes: List[int],
    number_vehicles: int,
    dual_values: Optional[Dict[int, float]],
    options: Dict[str, Any],
) -> _Built:
    """Directed two-commodity flow (``formulation: directed``; not the published model)."""
    Q, R, C, Omega = d.Q, d.R, d.C, d.Omega
    S = d.S_dict
    nodes, nodes_real = d.nodes, d.nodes_real
    cutoff = options["max_arc_distance_km"]
    pairs = [(i, j) for i in nodes for j in nodes if i != j and (cutoff is None or distance_matrix[i][j] <= cutoff)]

    x = mdl.addVars(pairs, vtype=GRB.BINARY, name="x")
    g = mdl.addVars(nodes, vtype=GRB.BINARY, name="g")
    # f_ij: flow of waste (commodity 1); h_ij: flow of empty capacity (commodity 2)
    f = mdl.addVars(pairs, vtype=GRB.CONTINUOUS, lb=0, name="f")
    h = mdl.addVars(pairs, vtype=GRB.CONTINUOUS, lb=0, name="h")
    k_var = mdl.addVar(lb=0, vtype=GRB.INTEGER, name="k_var")

    for i, j in pairs:
        mdl.addConstr(f[i, j] + h[i, j] == Q * x[i, j])
    for i in nodes_real:
        mdl.addConstr(
            quicksum(f[i, j] for j in nodes if (i, j) in f) - quicksum(f[j, i] for j in nodes if (j, i) in f)
            == S[i] * g[i]
        )
    for i in nodes_real:
        mdl.addConstr(
            quicksum(h[j, i] for j in nodes if (j, i) in h) - quicksum(h[i, j] for j in nodes if (i, j) in h)
            == S[i] * g[i]
        )
    mdl.addConstr(quicksum(f[i, 0] for i in nodes_real if (i, 0) in f) == quicksum(S[i] * g[i] for i in nodes_real))
    mdl.addConstr(quicksum(h[0, j] for j in nodes_real if (0, j) in h) == Q * k_var)
    mdl.addConstr(quicksum(f[0, j] for j in nodes_real if (0, j) in f) == 0)
    if number_vehicles > 0:
        mdl.addConstr(k_var <= number_vehicles)
    mdl.addConstr(quicksum(x[0, j] for j in nodes_real if (0, j) in x) == k_var)
    mdl.addConstr(quicksum(x[j, 0] for j in nodes_real if (j, 0) in x) == k_var)
    if options["link_depot_arcs"]:
        for j in nodes_real:
            if (0, j) in x:
                mdl.addConstr(x[0, j] <= g[j])
            if (j, 0) in x:
                mdl.addConstr(x[j, 0] <= g[j])
    forced = [mdl.addConstr(g[i] == 1, name=f"forced_{i}") for i in forced_nodes]
    for j in nodes_real:
        mdl.addConstr(quicksum(x[i, j] for i in nodes if (i, j) in x) == g[j])
        mdl.addConstr(quicksum(x[j, k] for k in nodes if (j, k) in x) == g[j])

    travel = C * quicksum(x[i, j] * distance_matrix[i][j] for i, j in pairs)
    if dual_values:
        pi_0 = dual_values.get(0, 0.0)
        mdl.setObjective(
            quicksum((R * S[i] - dual_values.get(i, 0.0)) * g[i] for i in nodes_real) - travel - pi_0 * k_var,
            GRB.MAXIMIZE,
        )
    else:
        mdl.setObjective(R * quicksum(S[i] * g[i] for i in nodes_real) - travel - Omega * k_var, GRB.MAXIMIZE)

    def set_starts(starts: List[List[List[int]]]) -> None:
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
                        load += S[b]
            mdl.setAttr("Start", start_vars, start_vals)

    def extract_route() -> List[int]:
        return extract_depot_delimited_route([(i, j) for (i, j) in pairs if x[i, j].X > 0.5], d.id_map)

    return _Built(x, g, k_var, forced, set_starts, extract_route)


def _options(values: Dict[str, Any]) -> Dict[str, Any]:
    """Read and validate the optional, off-by-default model switches."""
    formulation = str(values.get("formulation", SWCTCFParams.formulation)).strip().lower()
    if formulation not in ("paper", "directed"):
        raise ValueError("formulation must be 'paper' or 'directed'")
    depot_inflow = str(values.get("depot_inflow", SWCTCFParams.depot_inflow)).strip().lower()
    if depot_inflow not in ("equal", "le"):
        raise ValueError("depot_inflow must be 'equal' (eq. 15) or 'le' (notebook)")
    cutoff = _optional_float(values.get("max_arc_distance_km", SWCTCFParams.max_arc_distance_km))
    if cutoff is not None and not (math.isfinite(cutoff) and cutoff > 0):
        raise ValueError("max_arc_distance_km must be positive and finite, or null")
    return {
        "formulation": formulation,
        "depot_inflow": depot_inflow,
        "max_arc_distance_km": cutoff,
        "solver_tuning": _flag(values.get("solver_tuning", SWCTCFParams.solver_tuning)),
        "relax_forced_on_infeasible": _flag(
            values.get("relax_forced_on_infeasible", SWCTCFParams.relax_forced_on_infeasible)
        ),
        "link_depot_arcs": _flag(values.get("link_depot_arcs", SWCTCFParams.link_depot_arcs)),
        "warm_start": _flag(values.get("warm_start", SWCTCFParams.warm_start)),
    }


def _run_gurobi_optimizer(  # noqa: C901
    bins: NDArray[np.float64],
    distance_matrix: List[List[float]],
    env: Optional[gp.Env],
    values: Dict[str, Any],
    binsids: List[int],
    mandatory: List[int],
    number_vehicles: int = 1,
    time_limit: int = 60,
    seed: int = 42,
    dual_values: Optional[Dict[int, float]] = None,
) -> Tuple[List[int], float, float]:
    """Solve one day of the Smart Waste Collection Routing model.

    Args:
        bins (NDArray[np.float64]): Bin fill levels in percent (depot excluded).
        distance_matrix (List[List[float]]): Distance matrix between nodes (km).
        env (Optional[gp.Env]): Gurobi environment.
        values (Dict[str, Any]): Problem parameters (Omega, psi, Q, R, C, ...) and the
            optional switches described in the module docstring.
        binsids (List[int]): Global identifiers for bins.
        mandatory (List[int]): IDs of bins that must be collected.
        number_vehicles (int): Maximum number of vehicles; non-positive means unbounded.
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
    options = _options(values)

    # Shared preparation (identical across all three backends).
    d = build_tcf_data(bins, distance_matrix, values, binsids, mandatory, number_vehicles)
    forced_nodes = [i for i in d.nodes_real if d.criticos_dict[i] or d.S_dict[i] >= d.psi * 100]

    mdl = gp.Model("VRPP", env=env) if env else gp.Model("VRPP")
    try:
        # Do not loosen limits on a caller-owned environment. SoftMemLimit allows
        # solution extraction on MEM_LIMIT, unlike a hard allocation failure.
        inherited_threads = mdl.Params.Threads
        mdl.Params.Threads = min(int(threads), inherited_threads) if inherited_threads > 0 else int(threads)
        mdl.Params.SoftMemLimit = min(memory_gb, mdl.Params.SoftMemLimit)
        print(f"[INFO][VRPP-Gurobi] Threads={mdl.Params.Threads}, SoftMemLimit={mdl.Params.SoftMemLimit} GB")
        mdl.Params.Seed = seed

        build = _build_paper if options["formulation"] == "paper" else _build_directed
        built = build(mdl, d, distance_matrix, forced_nodes, number_vehicles, dual_values, options)

        # Optional, off by default: not part of the published model (#41).
        starts = _warm_starts(d, distance_matrix, forced_nodes) if options["warm_start"] else []
        if starts:
            built.set_starts(starts)

        if not env:
            mdl.Params.LogToConsole = 0
            mdl.Params.OutputFlag = 0
            if options["formulation"] == "paper":
                mdl.Params.FeasibilityTol = 1e-09  # as in collectioncompare's policy_gurobi
            if options["solver_tuning"]:
                mdl.Params.MIPFocus = 1
                mdl.Params.Heuristics = HEURISTICS_RATIO
                mdl.Params.Cuts = 3
                mdl.Params.CliqueCuts = 2
                mdl.Params.CoverCuts = 2
                mdl.Params.FlowCoverCuts = 2
                mdl.Params.GUBCoverCuts = 2
                mdl.Params.Presolve = 1
                mdl.Params.NodefileStart = NODEFILE_START_GB
                mdl.setParam("MIPGap", MIP_GAP)

        if time_limit > 0:
            mdl.Params.TimeLimit = float(time_limit)

        from logic.src.pipeline.simulations.solver_status import format_backend_status, note_solver_status

        mdl.optimize()
        note_solver_status(format_backend_status("gurobi", mdl.Status))
        if mdl.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD) and built.forced and options["relax_forced_on_infeasible"]:
            # Forcing every mandatory / over-psi bin can exceed the fleet's capacity.
            print(
                f"[WARN][VRPP-Gurobi] {len(built.forced)} forced visits are infeasible together; "
                "re-solving without forcing."
            )
            mdl.remove(built.forced)
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
                profit, cost = _plan_objective(d, distance_matrix, best)
                arcs = [(a, b) for trip in best for a, b in zip([0, *trip], [*trip, 0], strict=False)]
                route = extract_depot_delimited_route(arcs, d.id_map)
                print(
                    f"[WARN][VRPP-Gurobi] No solution found (status {mdl.Status}); "
                    f"executing the best Clarke-Wright start ({len(best)} trips)."
                )
                note_solver_status("fallback:clarke_wright", append=True)
                return route, profit, cost
            print(f"[WARN][VRPP-Gurobi] No solution found (status {mdl.Status}); the day collects nothing.")
            return [0, 0], 0.0, 0.0

        route = built.extract_route()
        if route == [0, 0]:
            # All-zero incumbent (or nothing collected): the shared empty-day shape.
            return [0, 0], 0.0, 0.0
        profit = float(mdl.ObjVal)
        local = {gid: i for i, gid in d.id_map.items()}
        tour = [local[n] for n in route]
        # Travel cost of the executed tour on the (possibly asymmetric) matrix.
        cost = float(sum(distance_matrix[a][b] for a, b in zip(tour, tour[1:], strict=False)))
        print(
            f"[INFO][VRPP-Gurobi] Profit: {profit}, Cost: {cost}, MIPGap: {mdl.Params.MIPGap}, "
            f"Collected: {sum(1 for n in route if n != 0)}"
        )
        return route, profit, cost
    finally:
        # Native solver memory must be released before the next simulated day,
        # including build/optimization errors and early no-incumbent returns.
        mdl.dispose()
