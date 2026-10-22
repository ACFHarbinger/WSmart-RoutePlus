"""Branch-and-Bound (BB) solver core engine.

Standard Best-Bound-First Branch-and-Bound algorithm based on LP relaxations.
While Land & Doig (1960) established the foundational concept of Branch-and-Bound
and spatial branching, this modern implementation utilizes cold/warm-start LP
resolves at each node (via Gurobi) rather than the original parametric systematic
hyperplane shifts.

The solver manages a search tree to ensure global optimality for VRPP/CWC VRP problems.

**REFACTORED**: Uses the Miller-Tucker-Zemlin (MTZ) compact formulation to ensure
subtour elimination strictly within the LP relaxations at each node.

**PERFORMANCE OPTIMIZATIONS**:
1. Persistent Gurobi model (no rebuilding at each node)
2. Maximization objective (direct profit maximization)
3. True Strong Branching (LP-based variable selection)

Reference:
    Land, A. H., & Doig, A. (1960). "An automatic method for solving discrete
    programming problems". Econometrica, 28(3), 497-520.

Attributes:
    BBSolver (class): Core Branch-and-Bound solver engine.
    run_bb_mtz (function): Entry point for MTZ-formulation BB.

Example:
    >>> from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_bound.mtz import run_bb_mtz
    >>> routes, obj = run_bb_mtz(dist_matrix, wastes, capacity, R, C)

Parts are relocated to mtz_branching.
"""

import heapq
import itertools
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum

from logic.src.configs.policies.routing.exact_and_decomposition_solvers.bb import BBConfig
from logic.src.policies.helpers.solvers_and_matheuristics.common.node import Node
from logic.src.tracking.viz_mixin import PolicyStateRecorder

from .mtz_branching import MTZBranchingMixin


class BBSolver(MTZBranchingMixin):
    """
    Exact Branch-and-Bound engine for Vehicle Routing Problems.

    This engine is designed to solve the Profit-Maximizing or Waste-Collecting
    Vehicle Routing Problem with high precision, utilizing automated
    variable selection and node pruning based on deterministic bounds.

    The solver maintains an 'incumbent' solution (the best integer solution
    found so far) and stops when the search tree is exhausted or the
    specified MIP gap is reached.

    **ARCHITECTURAL JUSTIFICATION**:
    This custom solver is engineered specifically for **full observability of the
    search state**. Exposing the B&B tree entirely in Python is a necessary
    architectural choice for integrating machine learning models (e.g., injecting
    Graph Neural Networks for state evaluation or imitation learning for variable
    selection heuristics), which is heavily restricted by standard commercial
    solver callback systems.

    **REFACTORED DESIGN**:
    - Persistent Gurobi model stored in self.model (initialized once)
    - Fast node evaluation via variable bound updates (no model rebuilding)
    - Maximization objective (profit maximization, not cost minimization)
    - True Strong Branching using LP solves

    Attributes:
        dist_matrix (np.ndarray): Symmetric matrix representing edge costs.
        wastes (Dict[int, float]): Mapping customer IDs to their respective fill levels.
        capacity (float): Physical payload limit for the routing model.
        R (float): Revenue coefficient per unit of collected waste.
        C (float): Cost coefficient per unit of distance traveled.
        time_limit (float): Maximum allowed runtime in seconds.
        mip_gap (float): Relative optimality gap threshold.
        branching_strategy (str): Strategy for selecting branching variables.
        strong_branching_limit (int): Number of candidates for strong branching.
        mandatory_indices (Set[int]): Local indices of nodes that MUST be visited.
        env (gp.Env): Gurobi environment for resource management.
        recorder (PolicyStateRecorder): Telemetry recorder for state tracking.
        num_nodes (int): Total number of nodes in the problem.
        nodes_range (range): Range object for node indices.
        customers (List[int]): List of customer node indices (excluding depot).
        incumbent_obj (float): Objective value of the best integer solution found.
        incumbent_routes (List[List[int]]): Route sequences of the incumbent solution.
        start_time (float): Start time of the solve operation.
        model (gp.Model): Persistent Gurobi LP model.
        x (Dict): Dictionary of edge decision variables.
        y (Dict): Dictionary of node visit decision variables.
        u (Dict): Dictionary of MTZ load variables.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        time_limit: float = 60.0,
        mip_gap: float = 0.01,
        branching_strategy: str = "strong",
        strong_branching_limit: int = 5,
        mandatory_indices: Optional[Set[int]] = None,
        env: Optional[gp.Env] = None,
        recorder: Optional[PolicyStateRecorder] = None,
    ):
        """
        Args:
            dist_matrix: Symmetric matrix representing edge costs.
            wastes: Mapping customer IDs to their respective fill levels.
            capacity: Physical payload limit for the routing model.
            R: Revenue coefficient per unit of collected waste.
            C: Cost coefficient per unit of distance traveled.
            time_limit: Maximum allowed runtime in seconds.
            mip_gap: Relative optimality gap threshold.
            branching_strategy: Strategy for selecting branching variables.
            strong_branching_limit: Number of candidates for strong branching.
            mandatory_indices: Set of customer nodes that MUST be included in routes.
            env: Optional Gurobi environment for resource management.
            recorder: Optional telemetry recorder for state tracking.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.branching_strategy = branching_strategy
        self.strong_branching_limit = strong_branching_limit
        self.mandatory_indices = mandatory_indices or set()
        self.env = env
        self.recorder = recorder

        self.num_nodes = len(dist_matrix)
        self.nodes_range = range(self.num_nodes)
        self.customers = list(range(1, self.num_nodes))

        # CHANGE: Maximization problem, so incumbent starts at -inf
        self.incumbent_obj = -float("inf")
        self.incumbent_routes: List[List[int]] = []
        self.start_time = 0.0

        # NEW: Persistent model and variable dictionaries
        self.model: Optional[gp.Model] = None
        self.x: Dict[Tuple[int, int], gp.Var] = {}
        self.y: Dict[int, gp.Var] = {}
        self.u: Dict[int, gp.Var] = {}

        # Initialize the base model once
        self._initialize_base_model()

    def _initialize_base_model(self) -> None:
        """
        Construct the persistent Linear Programming model with MTZ formulation.

        This method is called once during __init__ to create the base Gurobi model.
        All variables are defined as CONTINUOUS with bounds [0.0, 1.0].
        The model is then reused throughout the search by updating variable bounds.

        Uses Miller-Tucker-Zemlin (MTZ) load-based formulation for subtour elimination.
        MTZ provides a compact formulation suitable for LP relaxations without callbacks.
        The load variables (u[i]) track accumulated waste, naturally preventing subtours.

        **OBJECTIVE**: Maximize (Revenue - Travel Cost)
        """
        self.model = gp.Model("BB_Persistent", env=self.env) if self.env else gp.Model("BB_Persistent")
        self.model.setParam("OutputFlag", 0)  # Silence Gurobi output for sub-solves
        self.model.setParam("LogToConsole", 0)
        self.model.setParam("Threads", 1)  # Single-threaded sub-solves for better control

        # --- Decision Variables (CONTINUOUS for LP relaxation) ---

        # x[i,j]: Edge usage (CONTINUOUS [0,1] for LP relaxation)
        for i in self.nodes_range:
            for j in self.nodes_range:
                if i != j:
                    self.x[i, j] = self.model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"x_{i}_{j}")

        # y[i]: Node visit (CONTINUOUS [0,1] for LP relaxation)
        for i in self.customers:
            lb = 1.0 if i in self.mandatory_indices else 0.0
            self.y[i] = self.model.addVar(lb=lb, ub=1.0, vtype=GRB.CONTINUOUS, name=f"y_{i}")

        # u[i]: MTZ load variables for subtour elimination
        for i in self.customers:
            self.u[i] = self.model.addVar(
                lb=self.wastes.get(i, 0.0), ub=self.capacity, vtype=GRB.CONTINUOUS, name=f"u_{i}"
            )

        # --- Objective: Maximize (Revenue - Travel Cost) ---
        revenue = quicksum(self.y[i] * self.wastes.get(i, 0) * self.R for i in self.customers)
        travel_cost = quicksum(self.dist_matrix[i][j] * self.C * self.x[i, j] for (i, j) in self.x)
        self.model.setObjective(revenue - travel_cost, GRB.MAXIMIZE)

        # --- Base Constraints (Flow and Integrity) ---
        for i in self.customers:
            self.model.addConstr(
                quicksum(self.x[i, j] for j in self.nodes_range if (i, j) in self.x) == self.y[i],
                name=f"out_{i}",
            )
            self.model.addConstr(
                quicksum(self.x[j, i] for j in self.nodes_range if (j, i) in self.x) == self.y[i],
                name=f"in_{i}",
            )

        # Depot flow constraint
        self.model.addConstr(quicksum(self.x[0, j] for j in self.customers if (0, j) in self.x) <= 1, name="depot_out")
        self.model.addConstr(
            quicksum(self.x[j, 0] for j in self.customers if (j, 0) in self.x)
            == quicksum(self.x[0, j] for j in self.customers if (0, j) in self.x),
            name="depot_balance",
        )

        # Capacity constraint
        self.model.addConstr(
            quicksum(self.wastes.get(i, 0) * self.y[i] for i in self.customers) <= self.capacity,
            name="capacity",
        )

        # --- MTZ Subtour Elimination Constraints (Load-Based) ---
        # u[i] + waste[j] <= u[j] + Capacity * (1 - x[i,j])
        # This prevents any closed loops that do not originate from the depot
        for i in self.customers:
            for j in self.customers:
                if i != j and (i, j) in self.x:
                    self.model.addConstr(
                        self.u[i] + self.wastes.get(j, 0.0) <= self.u[j] + self.capacity * (1 - self.x[i, j]),
                        name=f"mtz_{i}_{j}",
                    )





    def solve(self) -> Tuple[List[List[int]], float]:
        """
        Perform the Branch-and-Bound search to find the optimal integer routes.

        Implements the core manual B&B loop using LP-based relaxations (Best-Bound-First):
        1. Initialize priority queue with root node
        2. Loop while queue is not empty:
           - Pop node with best bound
           - Prune by bound if node cannot improve incumbent
           - Evaluate LP relaxation with fixed variables
           - Prune by infeasibility or bound
           - Check integrality:
             * If integer and valid → update incumbent
             * If fractional → branch

        **MAXIMIZATION LOGIC**:
        - Priority queue uses negative bounds for max-heap behavior
        - Prune if node.bound <= incumbent_obj * (1 + mip_gap)

        Returns:
            A tuple of (Optimal Integer Routes, Total Objective Value).
        """
        self.start_time = time.perf_counter()

        # Initialize tie-breaker counter to ensure deterministic heap ordering
        counter = itertools.count()

        # Step 1: Initialize priority queue with root node
        # Root node has no fixed variables
        root_node = Node(bound=float("inf"))  # Upper bound unknown until LP is solved
        queue = [(-float("inf"), next(counter), root_node)]  # Priority uses -inf until first LP

        nodes_explored = 0

        # Step 2: Main B&B loop
        while queue:
            # Check time limit
            if time.perf_counter() - self.start_time >= self.time_limit:
                break

            # Pop node with best bound (for maximization, we negate the priority)
            _, _, current_node = heapq.heappop(queue)
            nodes_explored += 1

            # Pruning by Bound (before LP solve)
            # For maximization: if node.bound <= incumbent_obj * (1 + mip_gap), prune
            gap_threshold = self.incumbent_obj + abs(self.incumbent_obj) * self.mip_gap
            if self.incumbent_obj > -float("inf") and current_node.bound <= gap_threshold:
                continue

            # Evaluate LP relaxation
            current_obj = self._evaluate_node(current_node)

            # Pruning by Infeasibility
            if current_obj == -float("inf"):
                continue

            # Pruning by Bound (after LP solve)
            # For maximization: if current_obj <= incumbent_obj * (1 + mip_gap), prune
            gap_threshold = self.incumbent_obj + abs(self.incumbent_obj) * self.mip_gap
            if self.incumbent_obj > -float("inf") and current_obj <= gap_threshold:
                continue

            # Integrality Check
            branch_var = self._get_branching_variable()

            if branch_var is None:
                # Solution is integer-feasible
                # Verify no subtours exist (should be prevented by MTZ)
                routes = self._extract_routes()

                # Update incumbent (for maximization, keep the maximum)
                if routes and self._is_valid_solution(routes) and current_obj > self.incumbent_obj:
                    self.incumbent_obj = current_obj
                    self.incumbent_routes = routes
            else:
                # Branching - Variable is fractional
                var_type, var_idx = branch_var

                # Create two child nodes

                # Left Child (Exclude: variable = 0)
                left_child = Node(
                    bound=current_obj,
                    fixed_x=current_node.fixed_x.copy(),
                    fixed_y=current_node.fixed_y.copy(),
                    depth=current_node.depth + 1,
                )
                if var_type == "x":
                    left_child.fixed_x[var_idx] = 0
                else:
                    left_child.fixed_y[var_idx] = 0

                # For max-heap, push with negative priority and tie-breaker
                heapq.heappush(queue, (-current_obj, next(counter), left_child))

                # Right Child (Enforce: variable = 1)
                right_child = Node(
                    bound=current_obj,
                    fixed_x=current_node.fixed_x.copy(),
                    fixed_y=current_node.fixed_y.copy(),
                    depth=current_node.depth + 1,
                )
                if var_type == "x":
                    right_child.fixed_x[var_idx] = 1
                else:
                    right_child.fixed_y[var_idx] = 1

                # For max-heap, push with negative priority and tie-breaker
                heapq.heappush(queue, (-current_obj, next(counter), right_child))

        # Telemetry updates
        if self.recorder:
            self.recorder.record(
                engine="bb_lp_relaxation",
                nodes_explored=nodes_explored,
                obj_val=self.incumbent_obj,
                time=time.perf_counter() - self.start_time,
            )

        return self.incumbent_routes, self.incumbent_obj

    def _is_valid_solution(self, routes: List[List[int]]) -> bool:
        """
        Validate that a solution is feasible (no subtours, respects capacity).

        Args:
            routes: List of routes to validate.

        Returns:
            True if solution is valid, False otherwise.
        """
        # Check capacity constraint
        for route in routes:
            load = sum(self.wastes.get(i, 0) for i in route)
            if load > self.capacity:
                return False

        # Check that all nodes are visited at most once
        visited = set()
        for route in routes:
            for node in route:
                if node in visited:
                    return False  # Node visited multiple times
                visited.add(node)

        # Check mandatory nodes
        return all(node in visited for node in self.mandatory_indices)

    def _extract_routes(self) -> List[List[int]]:
        """
        Convert sparse binary edge variables into ordered route sequences.

        Traverses the solution adjacency graph starting from the depot (0).

        Returns:
            List of routes (each route is a list of customer node IDs).
        """
        edges = [e for e, v in self.x.items() if v.X > 0.5]
        adj: Dict[int, List[int]] = {i: [] for i in self.nodes_range}
        for i, j in edges:
            adj[i].append(j)

        routes = []
        for start_node in adj[0]:
            route = [start_node]
            curr = start_node
            visited = {0, start_node}
            while curr != 0 and adj[curr]:
                next_node = adj[curr][0]
                if next_node in visited and next_node != 0:
                    break  # Cycle detected, stop
                if next_node != 0:
                    route.append(next_node)
                    visited.add(next_node)
                curr = next_node
                if curr == 0:
                    break
            if route:
                routes.append(route)
        return routes


def run_bb_mtz(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    params: Optional[BBConfig] = None,
    mandatory_indices: Optional[Set[int]] = None,
    env: Optional[gp.Env] = None,
    recorder: Optional[PolicyStateRecorder] = None,
    **kwargs: Any,
) -> Tuple[List[List[int]], float]:
    """
    Dispatcher entry point for the MTZ-formulation Branch-and-Bound solver.

    Uses Miller-Tucker-Zemlin (MTZ) compact formulation with load variables
    for subtour elimination in LP relaxations.

    Args:
        dist_matrix: Symmetric distance matrix (n x n).
        wastes: Mapping of customer IDs to fill levels/profits.
        capacity: Vehicle payload capacity.
        R: Revenue coefficient per unit collected.
        C: Cost coefficient per unit distance.
        params: Standardized BB parameters.
        mandatory_indices: Set of mandatory customer nodes.
        env: Optional Gurobi environment for resource management.
        recorder: Optional telemetry recorder.
        kwargs: Additional arguments.

    Returns:
        Tuple of (routes, objective_value).
    """
    if params is None:
        params = BBConfig()

    solver = BBSolver(
        dist_matrix=dist_matrix,
        wastes=wastes,
        capacity=capacity,
        R=R,
        C=C,
        time_limit=params.time_limit,
        mip_gap=params.mip_gap,
        branching_strategy=params.branching_strategy,
        strong_branching_limit=params.strong_branching_limit,
        mandatory_indices=mandatory_indices,
        env=env,
        recorder=recorder,
    )
    return solver.solve()
