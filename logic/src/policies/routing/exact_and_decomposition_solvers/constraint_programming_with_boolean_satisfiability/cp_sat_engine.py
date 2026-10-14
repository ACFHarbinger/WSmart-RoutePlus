"""CP-SAT solver engine using OR-Tools.

Provides the underlying integration between the stochastic problem definition
and the Google OR-Tools CP-SAT logic solver.

Attributes:
    CPSATEngine (class): Core solver engine for CP-SAT logic.

Example:
    >>> engine = CPSATEngine(config, dist_matrix, wastes)
    >>> route, profit = engine.solve()
"""

from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from ortools.sat.python import cp_model

    OR_TOOLS_AVAILABLE = True
except ImportError:
    OR_TOOLS_AVAILABLE = False

from logic.src.configs.policies.cp_sat import CPSATConfig


class CPSATEngine:
    """CP-SAT multi-day routing solver using Google OR-Tools.

    Models the problem as an integrated inventory-routing problem over a fixed horizon.

    Attributes:
        config (CPSATConfig): Configuration parameters.
        distance_matrix (np.ndarray): Distance matrix between nodes.
        initial_wastes (Dict[int, float]): Initial waste levels at customer nodes.
        capacity (float): Vehicle capacity.
        num_nodes (int): Total number of nodes (including depot).
        num_customers (int): Total number of customer nodes.
        customers (List[int]): List of customer node indices.
        nodes (List[int]): List of all node indices.
        sf (int): Scaling factor for integer conversion.
        stats (Dict[str, Any]): Execution statistics.
    """

    def __init__(
        self,
        config: CPSATConfig,
        distance_matrix: np.ndarray,
        initial_wastes: Dict[int, float],
        capacity: float = 1.0,
    ):
        """
        Initializes the CP-SAT engine.

        Args:
            config: Configuration parameters.
            distance_matrix: Distance matrix between nodes.
            initial_wastes: Initial waste levels at customer nodes.
            capacity: Vehicle capacity.
        """
        self.config = config
        self.distance_matrix = distance_matrix
        self.initial_wastes = initial_wastes
        self.capacity = capacity

        self.num_nodes = distance_matrix.shape[0]
        self.num_customers = self.num_nodes - 1
        self.customers = list(range(1, self.num_nodes))
        self.nodes = list(range(self.num_nodes))

        self.sf = config.scaling_factor
        self.stats: Dict[str, Any] = {}

    def solve(self) -> Tuple[List[int], float]:
        """Solves the CP-SAT model for the multi-day horizon.

        Returns:
            Tuple[List[int], float]: A tuple containing:
                - List[int]: The optimized route for the first day.
                - float: The expected profit over the horizon.

        Raises:
            ImportError: If OR-Tools is not installed.
        """
        if not OR_TOOLS_AVAILABLE:
            raise ImportError("OR-Tools (ortools.sat.python) is required for CP-SAT policy.")

        model = cp_model.CpModel()

        # 1. Variables
        vars_dict = self._init_variables(model)

        # 2. Constraints
        self._add_routing_constraints(model, vars_dict)
        self._add_inventory_constraints(model, vars_dict)

        # 3. Objective
        self._add_objective(model, vars_dict)

        # 4. Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.config.time_limit
        solver.parameters.num_search_workers = self.config.search_workers
        if self.config.seed is not None:
            solver.parameters.random_seed = self.config.seed

        status = solver.Solve(model)

        self.stats = {
            "status": solver.StatusName(status),
            "user_time": solver.UserTime(),
            "wall_time": solver.WallTime(),
            "branches": solver.NumBranches(),
            "conflicts": solver.NumConflicts(),
        }

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            route = self._extract_route(solver, vars_dict["x"])
            expected_profit = solver.ObjectiveValue() / self.sf
            return route, expected_profit

        return [0, 0], 0.0

    def _init_variables(self, model: cp_model.CpModel) -> Dict[str, Dict]:
        """Initialises all CP-SAT variables.

        Args:
            model: The CP model to populate with variables.

        Returns:
            Dict[str, Dict]: A dictionary mapping variable names ('x', 'visit', etc.)
                to their respective OR-Tools variable objects.
        """
        x = {}
        visit = {}
        collected = {}
        waste = {}
        overflow = {}

        for d in range(1, self.config.num_days + 1):
            for i in self.nodes:
                visit[d, i] = model.NewBoolVar(f"v_{d}_{i}")
                collected[d, i] = model.NewIntVar(0, self.sf, f"q_{d}_{i}")
                if i == 0:
                    model.Add(collected[d, i] == 0)
                    model.Add(visit[d, i] == 1)

                waste[d, i] = model.NewIntVar(0, self.sf * 10, f"w_{d}_{i}")
                overflow[d, i] = model.NewIntVar(0, self.sf * 10, f"of_{d}_{i}")

                for j in self.nodes:
                    x[d, i, j] = model.NewBoolVar(f"x_{d}_{i}_{j}")

        return {"x": x, "visit": visit, "collected": collected, "waste": waste, "overflow": overflow}

    def _add_routing_constraints(self, model: cp_model.CpModel, vars_dict: Dict[str, Dict]):
        """Adds routing constraints (circuit and capacity) per day.

        Args:
            model: The CP model to add constraints to.
            vars_dict: Dictionary containing the variables.
        """
        x = vars_dict["x"]
        visit = vars_dict["visit"]
        collected = vars_dict["collected"]
        waste = vars_dict["waste"]

        for d in range(1, self.config.num_days + 1):
            circuit_arcs = []
            for i in self.nodes:
                for j in self.nodes:
                    circuit_arcs.append((i, j, x[d, i, j]))
                model.Add(x[d, i, i] == visit[d, i].Not())

            model.AddCircuit(circuit_arcs)
            model.Add(sum(collected[d, i] for i in self.customers) <= int(self.capacity * self.sf))

            for i in self.customers:
                model.Add(collected[d, i] <= waste[d, i])
                model.Add(collected[d, i] <= self.sf).OnlyEnforceIf(visit[d, i])
                model.Add(collected[d, i] == 0).OnlyEnforceIf(visit[d, i].Not())

    def _add_inventory_constraints(self, model: cp_model.CpModel, vars_dict: Dict[str, Dict]):
        """Adds inventory flow and overflow constraints across days.

        Args:
            model: The CP model to add constraints to.
            vars_dict: Dictionary containing the variables.
        """
        collected = vars_dict["collected"]
        waste = vars_dict["waste"]
        overflow = vars_dict["overflow"]

        increment_scaled = int(self.config.mean_increment * self.sf)
        for d in range(1, self.config.num_days + 1):
            for i in self.customers:
                if d == 1:
                    initial_w = int(self.initial_wastes.get(i, 0.0) * self.sf)
                    model.Add(waste[d, i] == initial_w)
                else:
                    model.Add(waste[d, i] == waste[d - 1, i] - collected[d - 1, i] + increment_scaled)

                model.Add(overflow[d, i] >= waste[d, i] - self.sf)
                model.Add(overflow[d, i] >= 0)

    def _add_objective(self, model: cp_model.CpModel, vars_dict: Dict[str, Dict]):
        """Defines the objective function.

        Args:
            model: The CP model to set the objective for.
            vars_dict: Dictionary containing the variables.
        """
        x = vars_dict["x"]
        collected = vars_dict["collected"]
        overflow = vars_dict["overflow"]

        cost_scaled = []
        for d in range(1, self.config.num_days + 1):
            for i in self.nodes:
                for j in self.nodes:
                    if i != j:
                        dist_val = int(self.distance_matrix[i][j] * self.sf)
                        cost_scaled.append(x[d, i, j] * dist_val)

        revenue_total = sum(collected[d, i] for d in range(1, self.config.num_days + 1) for i in self.customers)
        cost_total = sum(cost_scaled)
        penalty_total = sum(overflow[d, i] for d in range(1, self.config.num_days + 1) for i in self.customers)

        model.Maximize(
            int(self.config.waste_weight) * revenue_total
            - int(self.config.cost_weight) * cost_total
            - int(self.config.overflow_penalty) * penalty_total
        )

    def _extract_route(self, solver: cp_model.CpSolver, x: Dict) -> List[int]:
        """Extracts the day 1 route from the solution.

        Args:
            solver: The CP solver containing the solution.
            x: Dictionary of arc traversal variables.

        Returns:
            List[int]: The ordered sequence of node visits for day 1.
        """
        current = 0
        route = [0]
        while True:
            next_node = -1
            for j in self.nodes:
                if j != current and solver.Value(x[1, current, j]) == 1:
                    next_node = j
                    break
            if next_node == -1 or next_node == 0:
                break
            route.append(next_node)
            current = next_node
        route.append(0)
        return route
