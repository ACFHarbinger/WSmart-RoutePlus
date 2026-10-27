"""
Kernel Search matheuristic solver for VRPP.

This module implements the core Gurobi-based Kernel Search algorithm.
The framework generalizes decomposition techniques by restricting binary variables
to a promising subset (the Kernel) and iteratively expanding it with additional sets (Buckets).

Reference:
    Angelelli, R., Dell'Amico, M., & Martello, S. (2010). "A Kernel Search Algorithm for the
    Vehicle Routing Problem with Pickups and Deliveries". Networks, 56(4), 297–307.
    Angelelli, R., & Mansini, R. (2010). "Kernel Search: a new heuristic framework for
    portfolio selection".

The Gurobi cut callbacks moved to :mod:`cut_callbacks` and the model construction
and warm-start helpers moved to :mod:`model_setup`; both are re-imported here so
every existing ``from kernel_search.solver import ...`` keeps working. This module
keeps the variable partitioning, the iterative bucket solves, the tour
reconstruction and the main entry point.

Attributes:
    run_kernel_search_gurobi: Main entry point for the Kernel Search solver.

Example:
    >>> import numpy as np
    >>> from logic.src.policies.routing.matheuristics.kernel_search.solver import run_kernel_search_gurobi
    >>> dm = np.zeros((3, 3))
    >>> tour, obj, cost = run_kernel_search_gurobi(dm, {1: 0.5}, 10.0, 5.0, 1.0, [])
"""

import random
from typing import Any, Dict, List, Optional, Set, Tuple, cast

import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum

from logic.src.policies.helpers.operators.solution_initialization.greedy_si import build_greedy_routes
from logic.src.policies.routing.matheuristics.kernel_search.cut_callbacks import (
    _dfj_subtour_elimination_callback,
    _root_node_callback,
)
from logic.src.policies.routing.matheuristics.kernel_search.model_setup import (
    _set_mip_start,
    _setup_ks_model,
)
from logic.src.tracking.viz_mixin import PolicyStateRecorder


def _get_partitioned_vars(
    model: gp.Model,
    x: Dict[Tuple[int, int], gp.Var],
    y: Dict[int, gp.Var],
    initial_kernel_size: int,
    bucket_size: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: List[int],
) -> Tuple[List[gp.Var], List[gp.Var], List[List[gp.Var]]]:
    """
    Solve the root node relaxation and partition discrete variables into Kernel and Buckets.
    Ranking is based on subtour-aware relaxation values.

    Note: A unified list is used for all variables (x and y) because this VRPP
    formulation is a pure 0-1 MIP. While Guastaroba et al. (2017) suggests
    maintaining separate lists for binary and general integer variables, they
    are merged here as all discrete variables in this model are binary.

    Args:
        model: Gurobi model to optimize.
        x: Edge decision variables keyed by (i, j) tuples.
        y: Node selection variables keyed by node index.
        initial_kernel_size: Minimum number of variables in the initial kernel.
        bucket_size: Number of variables per bucket.
        dist_matrix: Distance matrix of shape (n_nodes, n_nodes).
        wastes: Waste amounts keyed by node index.
        capacity: Vehicle capacity.
        R: Revenue per unit of waste collected.
        C: Cost per unit of distance traveled.
        mandatory_nodes: Node indices that must be included in the solution.

    Returns:
        Tuple[List[gp.Var], List[gp.Var], List[List[gp.Var]]]: Kernel variables,
            remaining variables, and bucketed remaining variables.
    """
    # Ensure Lazy Constraints are enabled for any solve with callbacks
    model.Params.LazyConstraints = 1

    # Prepare for root node relaxation harvesting
    all_vars_list = list(x.values()) + list(y.values())
    model._all_vars_list = all_vars_list
    model._node_rel = [0.0] * len(all_vars_list)
    model._y_vars = y  # Needed for separation logic

    # variables already binary from _setup_ks_model(use_binary_vars=True) logic in run call

    # Optimize with the root node callback
    model.optimize(_root_node_callback)

    # 3. Extract relaxation values
    var_values = {var: val for var, val in zip(all_vars_list, model._node_rel, strict=False)}

    # 4. Compute a totally feasible heuristic route for robustness
    rng = random.Random()
    heuristic_routes = build_greedy_routes(
        dist_matrix=dist_matrix, wastes=wastes, capacity=capacity, R=R, C=C, mandatory_nodes=mandatory_nodes, rng=rng
    )

    heuristic_edges = set()
    heuristic_nodes = set()
    for route in heuristic_routes:
        if not route:
            continue
        heuristic_edges.add((0, route[0]))
        for i in range(len(route) - 1):
            heuristic_edges.add((route[i], route[i + 1]))
            heuristic_nodes.add(route[i])
        heuristic_edges.add((route[-1], 0))
        heuristic_nodes.add(route[-1])

    # 5. Extract variables and their harvested relaxation values
    all_vars = []
    for (i, j), var in x.items():
        val = var_values.get(var, 0.0)
        all_vars.append((var, val, "x", (i, j)))
    for i, var in y.items():
        val = var_values.get(var, 0.0)
        all_vars.append((var, val, "y", i))  # type: ignore[arg-type]

    # Sort logic: Root node relaxation value descending
    all_vars.sort(key=lambda item: item[1], reverse=True)

    # 6. Partition into sets
    # Include heuristic variables and mandatory nodes to guarantee ILP feasibility
    lp_support_size = sum(1 for _, val, _, _ in all_vars if val > 1e-4)
    actual_kernel_size = max(initial_kernel_size, lp_support_size)

    mandatory_set = set(mandatory_nodes)
    kernel_vars_list = []
    kernel_vars_set = set()
    remaining_vars = []

    # First pass: actual_kernel_size of LP variables
    for v_obj, _, _, _ in all_vars[:actual_kernel_size]:
        kernel_vars_list.append(v_obj)
        kernel_vars_set.add(v_obj)

    # Second pass: ensure ALL heuristic and mandatory variables are present
    for v_obj, _, vtype, idx in all_vars:
        if v_obj in kernel_vars_set:
            continue

        is_heuristic = (vtype == "x" and idx in heuristic_edges) or (vtype == "y" and idx in heuristic_nodes)
        is_mandatory = vtype == "y" and idx in mandatory_set

        if is_heuristic or is_mandatory:
            kernel_vars_list.append(v_obj)
            kernel_vars_set.add(v_obj)
        else:
            remaining_vars.append(v_obj)

    kernel_vars = kernel_vars_list

    buckets = [remaining_vars[i : i + bucket_size] for i in range(0, len(remaining_vars), bucket_size)]

    return kernel_vars, remaining_vars, buckets


def _solve_ks_iterations(
    model: gp.Model,
    kernel_vars: List[gp.Var],
    buckets: List[List[gp.Var]],
    remaining_vars: List[gp.Var],
    time_limit: float,
    mip_limit_nodes: int,
) -> Set[gp.Var]:
    """Perform the iterative improvement phase by solving sub-problems with DFJ cuts.

    Each iteration adds one bucket of variables and a budget constraint.

    Args:
        model: Gurobi model to optimize.
        kernel_vars: Variables in the initial kernel.
        buckets: List of variable buckets to iterate over.
        remaining_vars: All variables not in the kernel, used to fix upper bounds.
        time_limit: Total time budget in seconds.
        mip_limit_nodes: Node limit for each bucket MIP solve.

    Returns:
        Set[gp.Var]: Variables that took positive values in the best solution found.
    """
    # Fix everything to zero initially
    for v in remaining_vars:
        v.UB = 0

    best_obj = -float("inf")
    used_vars = set()
    start_time = model.Runtime

    # 1. Solve INITIAL KERNEL with DFJ callback
    model.setParam("TimeLimit", max(2.0, (time_limit / (len(buckets) + 1))))
    model.setParam("NodeLimit", mip_limit_nodes)
    model.optimize(_dfj_subtour_elimination_callback)

    if model.SolCount > 0:
        best_obj = model.ObjVal
        used_vars = {v for v in kernel_vars if v.X > 0.5}

    # 2. Iterative BUCKET refinement
    for bucket in buckets:
        # Check time budget
        if (model.Runtime - start_time) > time_limit:
            break

        # Dynamically unfix the current bucket
        for v in bucket:
            v.UB = 1

        # KS Bucket Constraint: sum_{j in B_i} x_j >= 1
        bucket_constr = model.addConstr(quicksum(bucket) >= 1, name="KS_Bucket_Constraint")

        # Step 2: Objective Cutoff (Guastaroba 2017, Step 3.1)
        # Solutions worse than this will be discarded.
        model.Params.Cutoff = best_obj + 1e-4

        model.optimize(_dfj_subtour_elimination_callback)

        if model.SolCount > 0:
            if model.ObjVal > best_obj:
                best_obj = model.ObjVal
                # If improvement, capture NEW useful variables
                for v in bucket:
                    if v.X > 0.5:
                        used_vars.add(v)

            # Fix unused variables in the bucket back to zero to keep sub-MIP small
            for v in bucket:
                if v.X < 0.5:
                    v.UB = 0
        else:
            # Revert unfixing if no better solution found
            for v in bucket:
                v.UB = 0

        # RIGOR: Explicitly remove the bucket constraint after the solve
        model.remove(bucket_constr)

    return used_vars


def _reconstruct_tour(
    num_nodes: int, x: Dict[Tuple[int, int], Any], dist_matrix: np.ndarray
) -> Tuple[List[int], float]:
    """
    Convert the resulting binary flow variables into a sequence of nodes (a tour).

    Args:
        num_nodes: Total number of nodes in graph.
        x: Flow variables from the Gurobi model (either gp.Var with .X or floats).
        dist_matrix: Matrix used for distance calculation.

    Returns:
        Tuple: (tour list, total routing cost)
    """
    # 1. Extract activated edges
    active_edges = []
    for (i, j), var in x.items():
        val = cast(float, getattr(var, "X", var))
        if val > 0.5:
            active_edges.append((i, j))

    if not active_edges:
        return [0, 0], 0.0

    # 2. Build adjacency map for path reconstruction
    adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
    for i, j in active_edges:
        adj[i].append(j)

    # 3. Traverse the path starting at depot 0
    full_tour = [0]
    current = 0
    visited_nodes = {0}
    while True:
        if not adj[current]:
            break
        nx_node = adj[current].pop(0)
        # Handle potential cycle/convergence to zero
        if nx_node in visited_nodes and nx_node != 0:
            break
        full_tour.append(nx_node)
        visited_nodes.add(nx_node)
        current = nx_node
        if current == 0:
            break

    # 4. Ensure it closes at the depot
    if full_tour[-1] != 0:
        full_tour.append(0)

    # 5. Calculate final cost
    routing_cost = sum(dist_matrix[i][j] for i, j in active_edges)
    return full_tour, float(routing_cost)


def run_kernel_search_gurobi(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: List[int],
    initial_kernel_size: int = 50,
    bucket_size: int = 20,
    max_buckets: int = 10,
    time_limit: float = 300.0,
    mip_limit_nodes: int = 5000,
    mip_gap: float = 0.01,
    seed: int = 42,
    env: Optional[gp.Env] = None,
    recorder: Optional[PolicyStateRecorder] = None,
) -> Tuple[List[int], float, float]:
    """
    Entry point to solve the VRPP via Kernel Search matheuristic.

    Orchestrates the model setup, variable partitioning, iterative improvement,
    and final tour reconstruction.

    Args:
        dist_matrix: Pairwise distance matrix.
        wastes: Node demands/waste levels.
        capacity: Vehicle capacity.
        R: Revenue multiplier for collected waste.
        C: Cost multiplier for distance.
        mandatory_nodes: Nodes that must be visited.
        initial_kernel_size: Number of variables in the primary search space.
        bucket_size: Number of variables explored per iteration.
        max_buckets: Maximum iterations (bucket solves).
        time_limit: Total wall-clock runtime limit (seconds).
        mip_limit_nodes: Maximum branch-and-bound nodes per sub-MIP solve.
        mip_gap: Targeted optimality gap for each sub-problem.
        seed: Random seed for solver reproducibility.
        env: Optional pre-configured Gurobi environment.
        recorder: Optional utility for tracking search progress/statistics.

    Returns:
        Tour, MILP Objective value, and physical Routing Cost.
    """
    # 1. Initialize Model
    model = gp.Model("KS_VRPP", env=env) if env else gp.Model("KS_VRPP")
    model.setParam("OutputFlag", 0)  # Deactivate verbose output
    model.setParam("LogToConsole", 0)
    model.setParam("Seed", seed)
    model.setParam("MIPGap", mip_gap)

    # Phase 1: Structural Setup (ensure continuous variables for root relaxation)
    x, y = _setup_ks_model(model, dist_matrix, wastes, capacity, R, C, mandatory_nodes, use_binary_vars=False)

    # Phase 2: LP Relaxation & Variable Partitioning
    # This phase creates the ranking of variables for the Kernel and Buckets.
    kernel_vars, remaining_vars, buckets = _get_partitioned_vars(
        model, x, y, initial_kernel_size, bucket_size, dist_matrix, wastes, capacity, R, C, mandatory_nodes
    )
    if not kernel_vars:
        # Fallback if LP solve failed to find even partial feasibility
        # print(f"DEBUG: KS Partitioning failed. LP Status: {model.Status}")
        return [0, 0], 0.0, 0.0

    # print(f"DEBUG: KS Kernel size: {len(kernel_vars)}. Mandatory in kernel: {[v.VarName for v in kernel_vars if 'y_2' in v.VarName]}")

    # Promote to BINARY for the iterative MIP search phase
    for v in model.getVars():
        v.VType = GRB.BINARY
    model.update()

    # Limit search space by max_buckets parameter
    buckets = buckets[:max_buckets]

    # Warm-start with greedy heuristic to ensure feasibility
    _set_mip_start(model, x, y, dist_matrix, wastes, capacity, R, C, mandatory_nodes)

    # Phase 3: Iterative Search (Incremental MIPs)
    # The kernel grows and unused variables are pruned in each step.
    # print(f"DEBUG: Starting KS Iterations. MIP Start SolCount: {model.SolCount}")
    used_vars = _solve_ks_iterations(model, kernel_vars, buckets, remaining_vars, time_limit, mip_limit_nodes)
    # print(f"DEBUG: KS Iterations complete. SolCount: {model.SolCount}")

    # Phase 4: Final extraction and tour reconstruction
    if not used_vars:
        return [0, 0], 0.0, 0.0

    # Unfix all variables identified as useful for a final "polishing" solve
    for v in used_vars:
        v.UB = 1

    model.optimize(_dfj_subtour_elimination_callback)

    if model.SolCount == 0:
        return [0, 0], 0.0, 0.0

    tour, cost = _reconstruct_tour(len(dist_matrix), x, dist_matrix)

    # 5. Analytics tracking
    if recorder:
        recorder.record(engine="kernel_search", obj_val=model.ObjVal, cost=cost, solved=1)

    return tour, float(model.ObjVal), float(cost)
