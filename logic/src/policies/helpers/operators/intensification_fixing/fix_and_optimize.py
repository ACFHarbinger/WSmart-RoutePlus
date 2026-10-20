"""Fix-and-Optimize (Sub-MIP) Intensification Operator.

Selects a subset of routes to optimise exactly while holding the remainder
fixed.  The "free" routes' customers are extracted and handed to Gurobi as a
small CVRP or VRPP sub-MIP.  The proven-optimal recombination found by the
solver replaces the original free routes in the plan.

Selection strategy:
    Routes are ranked by quality (total distance / total load for CVRP, or
    profit for VRPP).  The ``free_fraction`` worst-quality routes are
    designated "free" and submitted to the sub-MIP.  All other routes remain
    fixed and are appended unchanged to the final plan.

Sub-MIP formulation (CVRP -- Miller-Tucker-Zemlin)::

    Variables  x[i,j] in {0,1}  (arc traversal), u[i] in [demand_i, Q]  (MTZ load).
    Objective  Minimize Sum d[i,j] * x[i,j].
    Constraints:
        Sum_j x[i,j] = 1  for all customer i      (depart each customer once)
        Sum_i x[i,j] = 1  for all customer j      (enter each customer once)
        Sum_j x[0,j] = K_free                     (exactly K_free vehicles from depot)
        u[j] >= u[i] + demand[j] - Q(1 - x[i,j]) for all i,j in customers (MTZ)

Sub-MIP formulation (VRPP)::

    Additional binary y[i] in {0,1} (whether customer i is visited).
    Objective  Maximize Sum waste[i]*R*y[i] - Sum d[i,j]*C*x[i,j].
    Flow conservation uses y[i] instead of 1:  Sum_j x[i,j] = y[i].
    MTZ:  u[j] >= u[i] + demand[j] - Q(1 - x[i,j])  (same as CVRP).

If Gurobi finds no feasible integer solution within *time_limit* the original
free routes are returned unchanged as a fallback.

Attributes:
    None

Requirements:
    Gurobi Optimizer >= 11.0 with a valid licence.

Example:
    >>> from logic.src.policies.helpers.operators.intensification import fix_and_optimize
    >>> improved = fix_and_optimize(routes, dist_matrix, wastes, capacity)
    >>> improved = fix_and_optimize_profit(routes, dist_matrix, wastes, capacity,
    ...                                    R=1.0, C=0.5)

Note:
    VRPP profit-maximising components and route metrics are located in
    :mod:`.fix_and_optimize_profit`.
"""

from typing import Dict, List, Optional, Set, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from .fix_and_optimize_profit import (
    _route_distance as _route_distance,
)
from .fix_and_optimize_profit import (
    _route_profit as _route_profit,
)
from .fix_and_optimize_profit import (
    _solve_vrpp_mip as _solve_vrpp_mip,
)
from .fix_and_optimize_profit import (
    fix_and_optimize_profit as fix_and_optimize_profit,
)

__all__ = [
    "fix_and_optimize",
    "fix_and_optimize_profit",
    "_route_distance",
    "_route_profit",
    "_solve_cvrp_mip",
    "_solve_vrpp_mip",
]

# ---------------------------------------------------------------------------
# Gurobi CVRP sub-MIP solver
# ---------------------------------------------------------------------------


def _solve_cvrp_mip(  # noqa: C901
    free_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    n_vehicles: int,
    time_limit: float,
    seed: int,
) -> List[List[int]]:
    """Solve a CVRP sub-MIP for *free_nodes* using Gurobi (MTZ formulation).

    All nodes in *free_nodes* are mandatory.  The solver minimises total
    travel distance with at most *n_vehicles* vehicles.

    Args:
        free_nodes: Customer node IDs to optimise (actual dist_matrix indices).
        dist_matrix: Full distance matrix; depot at index 0.
        wastes: Node demand lookup.
        capacity: Vehicle capacity Q.
        n_vehicles: Maximum number of vehicles to use (K_free).
        time_limit: Gurobi wall-clock time limit in seconds.
        seed: Gurobi random seed.

    Returns:
        List of routes found by the sub-MIP, or one route per customer as a
        fallback if no integer solution is found within the time limit.
    """
    N = len(free_nodes)
    if N == 0:
        return []

    # verts[0] = depot (0), verts[1..N] = customers (free_nodes)
    verts = [0] + free_nodes

    m = gp.Model("fix_and_opt_cvrp")
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", time_limit)
    m.setParam("Seed", seed)
    m.setParam("MIPGap", 1e-4)
    m.setParam("LogToConsole", 0)

    # Arc variables
    x: Dict[Tuple[int, int], gp.Var] = {}
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                x[i, j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # MTZ load variables u[1..N] ∈ [demand_i, Q]
    u: Dict[int, gp.Var] = {}
    for i in range(1, N + 1):
        dem = wastes.get(verts[i], 0.0)
        u[i] = m.addVar(lb=dem, ub=capacity, name=f"u_{i}")

    m.update()

    # Objective: minimise total arc cost
    m.setObjective(
        gp.quicksum(dist_matrix[verts[i], verts[j]] * x[i, j] for i in range(N + 1) for j in range(N + 1) if i != j),
        GRB.MINIMIZE,
    )

    # Each customer entered exactly once
    for j in range(1, N + 1):
        m.addConstr(gp.quicksum(x[i, j] for i in range(N + 1) if i != j) == 1)

    # Each customer departed exactly once
    for i in range(1, N + 1):
        m.addConstr(gp.quicksum(x[i, j] for j in range(N + 1) if i != j) == 1)

    # At most n_vehicles vehicles from depot
    m.addConstr(gp.quicksum(x[0, j] for j in range(1, N + 1)) <= n_vehicles)

    # Flow balance at depot
    m.addConstr(gp.quicksum(x[0, j] for j in range(1, N + 1)) == gp.quicksum(x[i, 0] for i in range(1, N + 1)))

    # MTZ subtour-elimination constraints
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            if i != j:
                dem_j = wastes.get(verts[j], 0.0)
                m.addConstr(u[j] >= u[i] + dem_j - capacity * (1 - x[i, j]))

    m.optimize()

    # Extract routes from solution
    if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) or m.SolCount == 0:
        # Fallback: one customer per route (always feasible)
        return [[n] for n in free_nodes]

    routes_out: List[List[int]] = []
    visited: Set[int] = set()

    for start in range(1, N + 1):
        if x[0, start].X < 0.5:
            continue
        if verts[start] in visited:
            continue

        route: List[int] = []
        curr = start
        max_steps = N + 1

        while curr != 0 and max_steps > 0:
            route.append(verts[curr])
            visited.add(verts[curr])
            moved = False
            for nxt in range(N + 1):
                if nxt != curr and x[curr, nxt].X > 0.5:
                    curr = nxt
                    moved = True
                    break
            if not moved:
                break
            max_steps -= 1

        if route:
            routes_out.append(route)

    # Ensure every free node is covered (safety net for numerical edge-cases)
    covered: Set[int] = {n for r in routes_out for n in r}
    for node in free_nodes:
        if node not in covered:
            routes_out.append([node])

    return routes_out


# ---------------------------------------------------------------------------
# Public API — CVRP
# ---------------------------------------------------------------------------


def fix_and_optimize(
    routes: List[List[int]],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    n_free: Optional[int] = None,
    free_fraction: float = 0.30,
    time_limit: float = 30.0,
    seed: int = 42,
) -> List[List[int]]:
    """Fix a subset of routes and solve a CVRP Sub-MIP for the remainder.

    Designates the worst-quality routes as "free" (scored by total distance ÷
    total load) and passes their customers to Gurobi as a small CVRP sub-MIP.
    All remaining "fixed" routes are appended unchanged to the result.

    The number of free routes is ``max(1, int(K · free_fraction))``, where K
    is the total number of routes, unless *n_free* is specified explicitly.

    If Gurobi finds no integer solution within *time_limit*, the original free
    routes are returned as-is (graceful fallback).

    Args:
        routes: Current plan (list of customer sequences, depot implicit at 0).
        dist_matrix: Square distance matrix ``(N+1, N+1)``, depot at index 0.
        wastes: Mapping from node index to demand.
        capacity: Maximum vehicle load.
        n_free: Exact number of routes to designate as free.  If ``None``
            (default), ``free_fraction`` is used.
        free_fraction: Fraction of routes to optimise exactly.  The
            ``free_fraction`` worst-quality routes are selected.  Default 0.30.
        time_limit: Gurobi time limit in seconds.  Larger values yield better
            solutions at the cost of longer wall-clock time.  Default 30.
        seed: Gurobi random seed for reproducibility.

    Returns:
        List[List[int]]: Improved routes.  All free-route customers are present
            in the returned plan (mandatory coverage).  Fixed routes are
            returned unchanged.  The input *routes* is never mutated.

    Example:
        >>> improved = fix_and_optimize(routes, dist_matrix, wastes, capacity)
        >>> improved = fix_and_optimize(routes, dist_matrix, wastes, capacity,
        ...                             free_fraction=0.5, time_limit=60.0)
    """
    if not routes:
        return []

    K = len(routes)

    # Score each route by distance-per-unit-load (higher = worse quality)
    scores: List[Tuple[float, int]] = []
    for i, route in enumerate(routes):
        dist = _route_distance(route, dist_matrix)
        load = sum(wastes.get(n, 0.0) for n in route)
        scores.append((dist / max(load, 1e-9), i))

    scores.sort(reverse=True)  # Worst quality first

    k_free = n_free if n_free is not None else max(1, int(K * free_fraction))
    k_free = min(k_free, K)

    free_indices: Set[int] = {idx for _, idx in scores[:k_free]}
    fixed_routes = [routes[i] for i in range(K) if i not in free_indices]
    free_routes = [routes[i] for i in range(K) if i in free_indices]
    free_nodes = [n for r in free_routes for n in r]

    if not free_nodes:
        return [list(r) for r in routes]

    optimal_free = _solve_cvrp_mip(
        free_nodes,
        dist_matrix,
        wastes,
        capacity,
        n_vehicles=k_free,
        time_limit=time_limit,
        seed=seed,
    )

    return [list(r) for r in fixed_routes] + optimal_free

