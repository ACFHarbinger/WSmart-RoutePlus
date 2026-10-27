"""
ILS-RVND-SP solver.

Combines the Iterated Local Search (ILS) metaheuristic with Randomized
Variable Neighborhood Descent (RVND) and an exact Set Partitioning (SP) formulation.

Reference:
    Subramanian et al. "A hybrid algorithm for a class of vehicle routing problems",
    Computers & Operations Research, 2013.

The search-phase methods (perturbation, ILS-RVND loop, strategies A/B and the
RVND operator factory) moved to :mod:`ils_rvnd_sp_search` and are mixed back in
via ``ILSRVNDSPSearchMixin``; this module keeps initialization, cost/profit
accounting, route-pool management, the Set Partitioning refinement and the
main ``solve`` entry point.

Attributes:
    ILSRVNDSPSolver: Main solver class combining ILS, RVND, and set partitioning.

Example:
    >>> import numpy as np
    >>> from logic.src.policies.routing.matheuristics.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp import ILSRVNDSPSolver
    >>> from logic.src.configs.policies.routing.matheuristics.ils_rvnd_sp import ILSRVNDSPConfig
    >>> dm = np.zeros((3, 3))
    >>> solver = ILSRVNDSPSolver(dm, {1: 0.5}, 10.0, 5.0, 1.0, ILSRVNDSPConfig())
"""

import logging
import time
from random import Random
from typing import Dict, List, Optional, Set, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from logic.src.configs.policies.routing.matheuristics.ils_rvnd_sp import ILSRVNDSPConfig
from logic.src.configs.policies.routing.meta_heuristics.aco_ks import KSparseACOConfig
from logic.src.policies.helpers.local_search.local_search_general import (
    GeneralLocalSearch,
)
from logic.src.policies.helpers.operators import (
    build_greedy_routes,
)
from logic.src.policies.routing.matheuristics.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp_search import (
    ILSRVNDSPSearchMixin,
)
from logic.src.policies.routing.matheuristics.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.rvnd import (
    RVND,
)

logger = logging.getLogger(__name__)


class ILSRVNDSPSolver(ILSRVNDSPSearchMixin):
    """
    ILS-RVND-SP solver executing ILS with RVND and resolving via Set Partitioning.

    Attributes:
        dist_matrix: Distance matrix of shape (n_nodes+1, n_nodes+1) including depot.
        wastes: Waste amounts keyed by node index.
        capacity: Vehicle capacity.
        R: Revenue multiplier per unit of waste.
        C: Cost multiplier per unit of distance.
        params: Runtime parameters controlling the solver.
        mandatory_nodes: Node indices that must be included in the solution.
        rng: NumPy random generator for stochastic operations.
        random: Python Random instance for sampling.
        n_nodes: Number of customer nodes (excluding depot).
        nodes: List of all customer node indices.
        route_pool: Set of unique canonical routes discovered during search.
        ls_manager: ACO-based local search manager.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: ILSRVNDSPConfig,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize the ILS-RVND-SP solver.

        Args:
            dist_matrix: NxN distance matrix.
            wastes: Dictionary of node wastes (demands/profits).
            capacity: Vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            params: Parameters for the algorithm.
            mandatory_nodes: List of mandatory nodes.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes or []
        self.rng = np.random.default_rng(params.seed) if params.seed is not None else np.random.default_rng()
        self.random = Random(params.seed) if params.seed is not None else Random()

        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))

        # Keep track of unique valid routes explored
        # A route is mapped to a tuple for correct hashing
        self.route_pool: Set[Tuple[int, ...]] = set()

        self.ls_manager = GeneralLocalSearch(
            dist_matrix=self.dist_matrix,
            waste=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            params=KSparseACOConfig(
                local_search_iterations=self.params.local_search_iterations,
                profit_aware_operators=self.params.profit_aware_operators,
                seed=self.params.seed,
            ),
        )

    def calculate_cost(self, routes: List[List[int]]) -> float:
        """Calculate total routing cost.

        Args:
            routes: List of routes, each a list of node indices.

        Returns:
            float: Total travel cost scaled by the cost multiplier C.
        """
        total_dist = 0.0
        for route in routes:
            if not route:
                continue
            dist = self.dist_matrix[0][route[0]]
            for i in range(len(route) - 1):
                dist += self.dist_matrix[route[i]][route[i + 1]]
            dist += self.dist_matrix[route[-1]][0]
            total_dist += dist
        return total_dist * self.C

    def calculate_profit(self, routes: List[List[int]]) -> float:
        """Calculate network profit (revenue - cost).

        Args:
            routes: List of routes, each a list of node indices.

        Returns:
            float: Net profit (revenue minus travel cost).
        """
        cost = self.calculate_cost(routes)
        revenue = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        return revenue - cost

    def _add_to_pool(self, routes: List[List[int]], pool: Optional[Set[Tuple[int, ...]]] = None):
        """Standardize and hash routes to add to the unique pool with canonical representation.

        Routes [1, 2, 3] and [3, 2, 1] are geographically identical (same tour, opposite direction).
        To prevent symmetric duplicates from bloating the MIP pool, we enforce a canonical
        orientation: the route endpoint with the smaller node ID is always placed first.
        This reduces the MIP pool size by ~50% and significantly speeds up Gurobi solving.

        Args:
            routes: Routes to add; each is a list of node indices.
            pool: Target pool to add to. If None, uses the solver's route_pool.
        """
        target_pool = pool if pool is not None else self.route_pool
        for route in routes:
            if not route:
                continue

            # Enforce canonical representation based on endpoint node IDs
            # Orient route so the smaller endpoint ID is always first (or keep single-node route as-is)
            canonical_route = tuple(route if route[0] < route[-1] else route[::-1]) if len(route) > 1 else tuple(route)

            target_pool.add(canonical_route)

    def build_initial_solution(self) -> List[List[int]]:
        """Construct initial solution using standard greedy heuristic.

        Returns:
            List[List[int]]: Initial feasible routes.
        """
        return build_greedy_routes(
            dist_matrix=self.dist_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
        )

    def solve_set_partitioning(
        self, pool: Optional[Set[Tuple[int, ...]]] = None
    ) -> Tuple[List[List[int]], float, float]:
        """
        Solve the Set Packing Problem exactly using Gurobi over the route pool.

        For VRPP (Vehicle Routing Problem with Profits):
        - Mandatory nodes must be visited exactly once (equality constraint)
        - Optional nodes can be visited at most once (inequality constraint <= 1)
        - Unvisited nodes contribute 0 to profit (no penalty)
        - Objective: maximize total profit (revenue - routing cost)

        This is a Set Packing formulation adapted for VRPP, not strict Set Partitioning.

        Args:
            pool: Route pool to solve over. If None, uses the solver's route_pool.

        Returns:
            Tuple[List[List[int]], float, float]: Best routes, profit, and travel cost.
        """
        target_pool = list(pool) if pool is not None else list(self.route_pool)
        n_routes = len(target_pool)

        if n_routes == 0:
            return [], 0.0, 0.0

        logger.info(f"ILS-RVND-SP solving Set Partitioning over {n_routes} unique routes.")

        # Precompute costs and node incidence
        route_costs = np.zeros(n_routes)
        route_profits = np.zeros(n_routes)

        # Map node to the indices of routes that contain it
        node_to_routes: Dict[int, List[int]] = {n: [] for n in self.nodes}
        for i, route in enumerate(target_pool):
            cost = self.dist_matrix[0][route[0]]
            for j in range(len(route) - 1):
                cost += self.dist_matrix[route[j]][route[j + 1]]
            cost += self.dist_matrix[route[-1]][0]
            route_costs[i] = cost * self.C

            revenue = sum(self.wastes.get(n, 0.0) * self.R for n in route)
            route_profits[i] = revenue - route_costs[i]

            for node in route:
                node_to_routes[node].append(i)

        # Build Model
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("LogToConsole", 0)
        env.start()
        model = gp.Model("ILSRVNDSP_SetPartitioning", env=env)

        model.setParam("TimeLimit", getattr(self.params, "mip_time_limit", 60.0))
        model.setParam("MIPGap", getattr(self.params, "sp_mip_gap", 0.01))

        # Variables: x[i] = 1 if route i is selected
        x = model.addVars(n_routes, vtype=GRB.BINARY, name="x")

        # Maximize Profit (Revenue - Cost)
        model.setObjective(gp.quicksum(route_profits[i] * x[i] for i in range(n_routes)), GRB.MAXIMIZE)

        # Constraints: Mandatory nodes visited exactly once
        mandatory_set = set(self.mandatory_nodes)
        for node in mandatory_set:
            if node_to_routes[node]:
                model.addConstr(gp.quicksum(x[i] for i in node_to_routes[node]) == 1, name=f"Mandatory_{node}")

        # Constraints: Optional nodes visited at most once
        optional_set = set(self.nodes) - mandatory_set
        for node in optional_set:
            if node_to_routes[node]:
                model.addConstr(gp.quicksum(x[i] for i in node_to_routes[node]) <= 1, name=f"Optional_{node}")

        # Constraints: Vehicle fleet limit (Subramanian 2013)
        max_v = getattr(self.params, "max_vehicles", 0)
        if max_v > 0:
            model.addConstr(gp.quicksum(x[i] for i in range(n_routes)) <= max_v, name="FleetLimit")

        model.optimize()

        if model.Status in {GRB.OPTIMAL, GRB.TIME_LIMIT} and model.SolCount > 0:
            best_routes = []
            for i in range(n_routes):
                if x[i].X > 0.5:
                    best_routes.append(list(target_pool[i]))

            best_cost = self.calculate_cost(best_routes)
            best_profit = self.calculate_profit(best_routes)
            return best_routes, best_profit, best_cost

        logger.warning("SP failed to find a valid combination. Falling back to best ILS solution.")
        return [], -float("inf"), float("inf")

    def solve(self, initial_solution: Optional[List[List[int]]] = None) -> Tuple[List[List[int]], float, float]:
        """Main ILS-RVND-SP execution loop with dual strategy handling.

        Args:
            initial_solution: Optional pre-built starting routes. Built greedily if None.

        Returns:
            Tuple[List[List[int]], float, float]: Best routes, profit, and travel cost.
        """
        start_time = time.perf_counter()

        # Create RVND with local search operators
        operators = self._create_rvnd_operators()
        rvnd = RVND(operators=operators, rng=self.random)

        n_limit = getattr(self.params, "N", 150)

        if self.n_nodes <= n_limit:
            global_best_routes, global_best_profit = self._run_strategy_a(initial_solution, start_time, rvnd)
        else:
            global_best_routes, global_best_profit = self._run_strategy_b(initial_solution, start_time, rvnd)

        best_cost = self.calculate_cost(global_best_routes)
        return global_best_routes, global_best_profit, best_cost
