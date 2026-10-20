"""Model construction and warm-start helpers for the Kernel Search matheuristic.

Holds the base VRPP model builder and the greedy MIP warm start split out of
``solver``.
"""

import random
from typing import Dict, List, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum

from logic.src.policies.helpers.operators.solution_initialization.greedy_si import build_greedy_routes


def _setup_ks_model(
    model: gp.Model,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: List[int],
    use_binary_vars: bool = False,
) -> Tuple[Dict[Tuple[int, int], gp.Var], Dict[int, gp.Var]]:
    r"""
    Setup the base mathematical parameters, variables, and constraints for the VRPP model.

    **CRITICAL CHANGE**: This function now uses a Branch-and-Cut architecture with
    Dantzig-Fulkerson-Johnson (DFJ) lazy constraints instead of Miller-Tucker-Zemlin (MTZ).

    Mathematical Justification:
        MTZ uses auxiliary continuous variables u[i] and Big-M constraints:
            u[j] >= u[i] + d[j] - Q(1 - x[i,j])

        The continuous LP relaxation of MTZ is notoriously weak because:
        1. The Big-M coefficient Q creates a large feasible region
        2. Fractional x[i,j] values allow u[i] to "float" arbitrarily
        3. The polyhedral bounds are loose, leading to poor dual bounds

        DFJ, in contrast, directly enforces combinatorial structure:
            sum_{i,j in S} x[i,j] <= |S| - 1  for all S ⊂ V \ {0}

        This yields a dramatically tighter LP relaxation because:
        1. Constraints are purely in the x-space (no auxiliary variables)
        2. Fractional solutions that "look like" subtours are immediately cut off
        3. The LP bound converges much faster to the integer optimum

        For instances with n > 40, MTZ typically fails to find integer solutions within
        reasonable time limits, while DFJ scales effectively to n = 100-200.

    Args:
        model: The Gurobi model instance.
        dist_matrix: Cost matrix between all nodes (0 is depot).
        wastes: Waste level at each node.
        capacity: Capacity of the vehicle.
        R: Revenue multiplier for collected waste.
        C: Cost multiplier for distance traveled.
        mandatory_nodes: Nodes that MUST have their `y` variable set to 1.
        use_binary_vars: If True, create BINARY variables directly. If False (default),
            create CONTINUOUS variables for LP relaxation (Kernel Search workflow).

    Returns:
        Tuple: A 2-tuple containing:
            - `x` (Dict[Tuple[int, int], gp.Var]): binary/continuous flow variables x_{i,j}.
            - `y` (Dict[int, gp.Var]): binary/continuous selection variables y_i.
    """
    num_nodes = len(dist_matrix)
    nodes = list(range(num_nodes))
    customers = list(range(1, num_nodes))
    m_set = set(mandatory_nodes)

    # Determine variable type based on usage context
    var_type = GRB.BINARY if use_binary_vars else GRB.CONTINUOUS

    # 1. Decision Variables
    # Flow variables: x[i,j] = 1 if the vehicle travels from i to j
    x = {}
    for i in nodes:
        for j in nodes:
            if i != j:
                x[i, j] = model.addVar(lb=0, ub=1, vtype=var_type, name=f"x_{i}_{j}")

    # Selection variables: y[i] = 1 if customer i is visited
    y = {}
    for i in customers:
        y[i] = model.addVar(lb=0, ub=1, vtype=var_type, name=f"y_{i}")

    # 2. Objective Function: Maximize (Revenue - Cost)
    travel_cost = quicksum(dist_matrix[i][j] * C * x[i, j] for i in nodes for j in nodes if i != j)
    revenue = quicksum(wastes.get(i, 0) * R * y[i] for i in customers)
    model.setObjective(revenue - travel_cost, GRB.MAXIMIZE)

    # 3. Flow Balance and Selection Constraints
    for i in customers:
        # If node i is visited, exactly one edge enters and one edge leaves
        model.addConstr(quicksum(x[i, j] for j in nodes if i != j) == y[i])
        model.addConstr(quicksum(x[j, i] for j in nodes if i != j) == y[i])

    # Depot usage: at most one trip leaves the depot
    model.addConstr(quicksum(x[0, j] for j in customers) <= 1)
    # Conservation of flow at the depot
    model.addConstr(quicksum(x[j, 0] for j in customers) == quicksum(x[0, j] for j in customers))

    # 4. Capacity Constraint (Simplified - no MTZ variables needed)
    # We enforce capacity via the greedy construction and lazy cuts
    # The DFJ callback will handle subtour elimination dynamically

    # 5. Mandatory Visits
    for i in m_set:
        model.addConstr(y[i] == 1)

    # 6. Enable Lazy Constraint Mode
    # This MUST be set BEFORE calling optimize() with a callback
    model.Params.LazyConstraints = 1

    # Store metadata in model for callback access
    model._x_vars = x
    model._num_nodes = num_nodes
    model._wastes = wastes
    model._capacity = capacity

    return x, y


def _set_mip_start(
    model: gp.Model,
    x: Dict[Tuple[int, int], gp.Var],
    y: Dict[int, gp.Var],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: List[int],
):
    """
    Warm-start the MIP solver with a greedy heuristic solution.

    This function computes a feasible solution using the greedy construction heuristic
    and injects it into the Gurobi model via the .Start attribute. This ensures that
    the solver begins with SolCount = 1, which is critical for matheuristics like
    Local Branching that require an incumbent to define the k-neighborhood.

    Mathematical Rationale:
        Warm-starting with a feasible solution provides:
        1. An immediate upper bound (for maximization problems)
        2. A starting point for the branch-and-bound tree
        3. A baseline for local search neighborhoods (LB, LB-VNS)
        4. Improved primal-dual gap closure rates

    Args:
        model: The Gurobi model instance.
        x: Edge flow variables.
        y: Node selection variables.
        dist_matrix: Distance matrix.
        wastes: Waste dictionary.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        mandatory_nodes: Nodes that must be visited.
    """
    rng = random.Random()
    heuristic_routes = build_greedy_routes(
        dist_matrix=dist_matrix, wastes=wastes, capacity=capacity, R=R, C=C, mandatory_nodes=mandatory_nodes, rng=rng
    )

    # Initialize all variables to 0
    for var in x.values():
        var.Start = 0.0
    for var in y.values():
        var.Start = 0.0

    # Set variables based on heuristic routes
    visited_nodes = set()
    for route in heuristic_routes:
        if not route:
            continue

        # First edge: depot to first customer
        if (0, route[0]) in x:
            x[(0, route[0])].Start = 1.0

        # Interior edges
        for i in range(len(route) - 1):
            if (route[i], route[i + 1]) in x:
                x[(route[i], route[i + 1])].Start = 1.0
            visited_nodes.add(route[i])

        # Last edge: last customer to depot
        if len(route) > 0:
            if (route[-1], 0) in x:
                x[(route[-1], 0)].Start = 1.0
            visited_nodes.add(route[-1])

    # Set y variables for visited nodes
    for node in visited_nodes:
        if node in y:
            y[node].Start = 1.0
