"""Gurobi formulations of SWC-TCF: the paper's two-commodity flow model and its asymmetric variant.

Attributes:
    _build_paper: The published symmetric model.
    _build_asymmetric: Arc-based variant for asymmetric matrices.

Example:
    >>> built = _build_paper(model, data, cfg)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import gurobipy as gp
from gurobipy import GRB, quicksum

from ._tcf_data import TCFData


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


def _copy_depot_distance(distance_matrix: List[List[float]], copy: int, i: int, j: int) -> float:
    """Distance (km) from i to j where the copy depot ``copy`` is a second copy of the real depot 0."""
    if i == copy and j == copy:
        return float(distance_matrix[0][0])
    if j == copy:
        return float(distance_matrix[0][i])
    if i == copy:
        return float(distance_matrix[0][j])
    return float(distance_matrix[i][j])


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
        return _copy_depot_distance(distance_matrix, copy, i, j)

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


def _build_asymmetric(  # noqa: C901
    mdl: gp.Model,
    d: TCFData,
    distance_matrix: List[List[float]],
    forced_nodes: List[int],
    number_vehicles: int,
    dual_values: Optional[Dict[int, float]],
    options: Dict[str, Any],
) -> _Built:
    """Asymmetric version of the published model (``asymmetric: true``; not the published model).

    Same node set and copy depot as ``_build_paper``, but every arc is directed and costs
    ``distance_matrix[i][j]`` in its own direction. A route is a directed path
    ``0 -> bins -> copy``. ``load[i, j]`` is the waste still on board when the vehicle
    traverses the arc, bounded by Q on used arcs; the copy depot receives the collected
    waste. The objective charges each traversed arc once (no factor one half).
    """
    Q, R, C, Omega = d.Q, d.R, d.C, d.Omega
    S = d.S_dict
    bins = d.nodes_real
    copy = d.n_bins + 1

    def dist(i: int, j: int) -> float:
        # Return legs use the bin's own distance back to the depot; the copy depot only
        # receives arcs, so the asymmetric matrix is never read in the reverse direction.
        if j == copy:
            return float(distance_matrix[i][0])
        return float(distance_matrix[i][j])

    origins = [0, *bins]
    targets = [*bins, copy]
    cutoff = options["max_arc_distance_km"]
    pairs = [(i, j) for i in origins for j in targets if i != j and (cutoff is None or dist(i, j) <= cutoff)]

    x = mdl.addVars(pairs, vtype=GRB.BINARY, name="x")
    load = mdl.addVars(pairs, vtype=GRB.CONTINUOUS, lb=0.0, name="load")
    g = mdl.addVars(bins, vtype=GRB.BINARY, name="g")
    k_var = mdl.addVar(lb=0, vtype=GRB.INTEGER, name="k")

    collected = quicksum(S[i] * g[i] for i in bins)
    # each visited bin has one successor and one predecessor
    for i in bins:
        mdl.addConstr(quicksum(x[i, j] for j in targets if (i, j) in x) == g[i], name=f"out_{i}")
        mdl.addConstr(quicksum(x[j, i] for j in origins if (j, i) in x) == g[i], name=f"in_{i}")
    # each route leaves the real depot and enters the copy depot once
    mdl.addConstr(quicksum(x[0, j] for j in targets if (0, j) in x) == k_var, name="depot_out")
    mdl.addConstr(quicksum(x[i, copy] for i in origins if (i, copy) in x) == k_var, name="copy_in")
    if number_vehicles > 0:
        mdl.addConstr(k_var <= number_vehicles, name="fleet")
    # load on a used arc is at most the vehicle capacity; an unused arc carries nothing
    for i, j in pairs:
        mdl.addConstr(load[i, j] <= Q * x[i, j], name=f"cap_{i}_{j}")
    # the vehicle leaves the depot empty and picks up each visited bin's waste
    for i in bins:
        mdl.addConstr(
            quicksum(load[i, j] for j in targets if (i, j) in load)
            - quicksum(load[j, i] for j in origins if (j, i) in load)
            == S[i] * g[i],
            name=f"load_{i}",
        )
    mdl.addConstr(quicksum(load[0, j] for j in targets if (0, j) in load) == 0, name="load_out")
    mdl.addConstr(quicksum(load[i, copy] for i in origins if (i, copy) in load) == collected, name="load_in")
    if options["link_depot_arcs"]:
        for j in bins:
            if (0, j) in x:
                mdl.addConstr(x[0, j] <= g[j])
    forced = [mdl.addConstr(g[i] == 1, name=f"forced_{i}") for i in forced_nodes]

    travel = C * quicksum(x[i, j] * dist(i, j) for i, j in pairs)
    if dual_values:
        pi_0 = dual_values.get(0, 0.0)
        mdl.setObjective(
            quicksum((R * S[i] - dual_values.get(i, 0.0)) * g[i] for i in bins) - travel - pi_0 * k_var,
            GRB.MAXIMIZE,
        )
    else:
        mdl.setObjective(R * collected - travel - Omega * k_var, GRB.MAXIMIZE)

    def set_starts(starts: List[List[List[int]]]) -> None:
        raise NotImplementedError("Clarke-Wright warm starts are not defined for the asymmetric model.")

    def extract_route() -> List[int]:
        succ: Dict[int, List[int]] = {i: [] for i in [0, *bins, copy]}
        for i, j in pairs:
            if x[i, j].X > 0.5:
                succ[i].append(j)
        route = [0]
        for first in succ[0]:
            if first == copy:
                continue  # an unused vehicle: the zero-length depot arc
            cur, steps = first, 0
            while cur != copy and steps <= len(bins) + 1:
                route.append(d.id_map[cur])
                if not succ[cur]:
                    break
                cur, steps = succ[cur][0], steps + 1
            route.append(0)
        return route if len(route) > 1 else [0, 0]

    return _Built(x, g, k_var, forced, set_starts, extract_route)
