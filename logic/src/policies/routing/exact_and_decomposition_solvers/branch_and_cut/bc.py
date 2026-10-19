r"""Branch-and-Cut Solver for VRPP using Gurobi.

This solver adapts the mathematical infrastructure of Fischetti et al. (1997)
for the Symmetric Generalized TSP to the Single-Vehicle VRPP. Nodes are treated
as Singleton Clusters, and SECs are implemented as Generalized Subtour Elimination
Constraints (GSECs) which reduce to Prize-Collecting SECs (PC-SECs).

References:
    Fischetti, M., Lodi, A., & Toth, P. (1997). "A Branch-and-Cut Algorithm for the Symmetric
    Generalized Traveling Salesman Problem". Operations Research, 45(2), 326-349.

    Lysgaard, J., Letchford, A. N., & Eglese, R. W. (2004). "A new branch-and-cut algorithm
    for the capacitated vehicle routing problem". Mathematical Programming, 100(2), 423-445.

Attributes:
    BranchAndCutSolver (class): The core optimization engine using Gurobi.
    GUROBI_AVAILABLE (bool): Whether the gurobipy package is installed.

Example:
    >>> from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_cut.bc import BranchAndCutSolver
    >>> solver = BranchAndCutSolver(model, params)
    >>> routes, profit, stats = solver.solve()
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from logic.src.configs.policies.bc import BCConfig
from logic.src.pipeline.learning.envs.common.expand_pool import current_expand_pool
from logic.src.policies.helpers.solvers_and_matheuristics import (
    SeparationEngine,
)
from logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model import VRPPModel
from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_cut.cut_callbacks import BCCutSeparationMixin
from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_cut.heuristics import (
    construct_initial_solution,
    construct_nn_solution,
    farthest_insertion,
)
from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_cut.lagrangian_bound import BCLagrangianMixin

try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    gp: Any = None  # type: ignore[assignment,no-redef]
    GRB: Any = None  # type: ignore[assignment,misc,no-redef]


class BranchAndCutSolver(BCLagrangianMixin, BCCutSeparationMixin):
    """Branch-and-Cut solver for VRPP.

    Follows the cutting plane algorithm described in Section 5.1 of Fischetti et al. (1997):
    1. Solve LP relaxation
    2. Separate violated inequalities
    3. Add cuts and resolve
    4. Branch if fractional
    5. Use primal heuristics for upper bounds

    Attributes:
        model (VRPPModel): Problem definition and data.
        params (BCConfig): Solver configuration parameters.
        scenarios (Optional[List[Dict[int, float]]]): SAA demand scenarios.
        separator (SeparationEngine): Logic for cut detection.
        stats (Dict[str, Any]): Internal solve statistics (cuts, nodes, etc.).
        gurobi_model (Optional[gp.Model]): Underlying Gurobi optimization model.
        x_vars (Dict[Tuple[int, int], gp.Var]): Edge selection variables.
        y_vars (Dict[int, gp.Var]): Node selection variables.
    """

    def __init__(
        self,
        model: VRPPModel,
        params: Optional[BCConfig] = None,
        scenarios: Optional[List[Dict[int, float]]] = None,
        **kwargs: Any,
    ):
        """
        Initialize Branch-and-Cut solver.

        Args:
            model: VRPPModel instance defining the problem.
            params: Standardized BC configuration.
            scenarios: Optional list of mappings from node index to stochastic demand for SAA scenarios.
            kwargs: Legacy configuration parameters (for backward compatibility).
        """
        if not GUROBI_AVAILABLE:
            raise ImportError("Gurobi is required for Branch-and-Cut solver")

        if params is None:
            # Create a defaults-aware params object from explicit arguments or framework defaults
            params = BCConfig(
                time_limit=kwargs.get("time_limit", 300.0),
                mip_gap=kwargs.get("mip_gap", 0.01),
                max_cuts_per_round=kwargs.get("max_cuts_per_round", 50),
                use_heuristics=kwargs.get("use_heuristics", True),
                verbose=kwargs.get("verbose", False),
                profit_aware_operators=kwargs.get("profit_aware_operators", False),
                enable_fractional_capacity_cuts=kwargs.get("enable_fractional_capacity_cuts", True),
            )

        self.model = model
        self.params = params

        # SAA Scenario Handling Scaffold
        self.scenarios = scenarios
        if self.params.use_saa:
            if self.params.verbose:
                print("============================================================")
                print("⚠ WARNING: Solving massive deterministic equivalent MILP   ")
                print("  for SAA. LP relaxations for compact stochastic models ")
                print("  are notoriously weak and solver may choke on symmetry.")
                print("============================================================")
            if not self.scenarios:
                if self.params.verbose:
                    print(f"Generating {self.params.num_scenarios} mock scenarios for SAA scaffold.")
                self.scenarios = []
                for _ in range(self.params.num_scenarios):
                    scen = {}
                    for idx in self.model.customers:
                        base_demand = self.model.get_node_demand(idx)
                        scen[idx] = max(0.0, np.random.normal(base_demand, max(0.01, 0.2 * base_demand)))
                    self.scenarios.append(scen)

        # Separation engine with adaptive capacity cut toggling
        # Auto-disable for large instances (n > 75) to prevent O(V⁴) bottleneck
        if model.n_nodes > 75 and self.params.enable_fractional_capacity_cuts:
            if self.params.verbose:
                print(f"⚠ Large instance detected (n={model.n_nodes}). Disabling exact fractional capacity cuts.")
            self.params.enable_fractional_capacity_cuts = False

        self.separator = SeparationEngine(
            model=self.model,
            enable_heuristic_rcc_separation=self.params.enable_heuristic_rcc_separation,
            enable_comb_cuts=getattr(self.params, "enable_comb_cuts", False),
        )

        # Statistics
        self.stats = {
            "total_cuts": 0,
            "sec_cuts": 0,
            "capacity_cuts": 0,
            "nodes_explored": 0,
            "lp_iterations": 0,  # counts integer solutions (MIPSOL events)
            "fractional_iterations": 0.0,  # counts fractional LP nodes (MIPNODE events)
        }

        # Cut pool: stores compact representations of previously found violated cuts.
        # Paper (Section 5.1): "All violated constraints found (except fan inequalities)
        # are permanently stored in compact form in a global data structure called the pool."
        # At each new B&B node, pool cuts are re-evaluated first before running any
        # separation algorithm, avoiding redundant rediscovery of ancestor cuts.
        # NOTE: Fan inequality separation (Equation 2.12) is NOT implemented.
        # NOTE: Pool re-evaluation is active via _evaluate_pool_cuts(), called at
        #       the start of both _add_integer_cuts() and _add_fractional_cuts().
        self._cut_pool: List[Any] = []
        self._cut_signatures: Set[Tuple[Any, ...]] = set()

        # Gurobi model
        self.gurobi_model: Optional[gp.Model] = None
        self.x_vars: Dict[Tuple[int, int], gp.Var] = {}
        self.y_vars: Dict[int, gp.Var] = {}

    def solve(self) -> Tuple[List[List[int]], float, Dict[str, Any]]:
        """Solve the VRPP instance using Branch-and-Cut.

        Returns:
            A 3-tuple containing:
                - routes: Optimized collection routes (list-of-lists).
                - profit: Total net profit (Revenue - Cost).
                - stats: Dictionary of solver performance metrics.
        """
        if self.params.verbose:
            print("=" * 60)
            print("Branch-and-Cut Solver for VRPP")
            print("=" * 60)
            print(f"Nodes: {self.model.n_nodes}")
            print(f"Capacity: {self.model.capacity}")
            print(f"Mandatory nodes: {len(self.model.mandatory_nodes)}")
            print("=" * 60)

        # Step 1: Build initial Gurobi model
        self._build_initial_model()

        # Step 2: Root node strengthening via Lagrangian relaxation.
        # Return value intentionally discarded — used only for VarHintVal hints.
        self._pre_optimize_lagrangian()

        # Step 3: REMOVED (Lagrangian GSECs are now separated in the first LP callback)

        # Step 4: Get initial primal solution (heuristic)
        if self.params.use_heuristics:
            best_tour = []
            best_profit = -float("inf")

            # 1. Greedy Profit Initialization (Best for VRPP Knapsack constraints)
            tour_greedy, profit_greedy = construct_initial_solution(self.model)
            if profit_greedy > best_profit:
                best_profit, best_tour = profit_greedy, tour_greedy

            # 2. Nearest Neighbor Initialization (Good for tight spatial clustering)
            tour_nn, profit_nn = construct_nn_solution(self.model)
            if profit_nn > best_profit:
                best_profit, best_tour = profit_nn, tour_nn

            # 3. Farthest Insertion (Good for exploring the convex hull, as per Fischetti 1997)
            tour_farthest, profit_farthest = farthest_insertion(
                self.model,
                profit_aware_operators=self.params.profit_aware_operators,
                expand_pool=current_expand_pool(),
            )
            if profit_farthest > best_profit:
                best_profit, best_tour = profit_farthest, tour_farthest

            if best_tour:
                if self.params.verbose:
                    print(f"Warm start heuristic selected with profit: {best_profit:.2f}")
                self._set_start_solution(best_tour)

        # Step 3: Enable lazy constraint callback for cutting planes
        # CRITICAL: LazyConstraints=1 is MANDATORY for correctness.
        # Without MTZ variables, the base model CANNOT enforce subtours or capacity on its own.
        # The SeparationEngine dynamically detects and adds violated SEC/RCC cuts via callbacks.
        # If callbacks fail or are disabled, the solver will return invalid routes.
        assert self.gurobi_model is not None
        self.gurobi_model.Params.LazyConstraints = 1
        self.gurobi_model.Params.PreCrush = 1  # type: ignore[union-attr]
        self.gurobi_model.optimize(self._lazy_constraint_callback)  # type: ignore[union-attr]

        # Step 4: Extract solution
        routes, profit = self._extract_solution()

        self.stats["obj_value"] = profit
        assert self.gurobi_model is not None
        self.stats["solve_time"] = self.gurobi_model.Runtime
        self.stats["mip_gap"] = self.gurobi_model.MIPGap if self.gurobi_model.SolCount > 0 else 1.0
        self.stats["nodes_explored"] = int(self.gurobi_model.NodeCount)

        return routes, profit, self.stats  # type: ignore[return-value]

    def _build_initial_model(self):
        """Build the initial Gurobi model with basic degree and capacity constraints."""
        self.gurobi_model = gp.Model("VRPP_BranchAndCut")
        self.gurobi_model.Params.TimeLimit = self.params.time_limit
        self.gurobi_model.Params.MIPGap = self.params.mip_gap
        self.gurobi_model.Params.OutputFlag = 1 if self.params.verbose else 0
        self.gurobi_model.Params.LogToConsole = 1 if self.params.verbose else 0

        # Decision variables
        # x[i,j]: Edge (i,j) is in the tour
        for i, j in self.model.edges:
            # Special case for depot edges: allow value 2 for [0, i, 0] trips
            if i == self.model.depot or j == self.model.depot:
                self.x_vars[(i, j)] = self.gurobi_model.addVar(lb=0, ub=2, vtype=GRB.INTEGER, name=f"x_{i}_{j}")
            else:
                self.x_vars[(i, j)] = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

        # y[i]: Node i is visited
        for i in self.model.customers:
            self.y_vars[i] = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"y_{i}")

        # Objective: Maximize profit = waste collected - travel cost
        # Gurobi minimizes by default, so negate
        waste_collected = gp.quicksum(self.model.get_node_profit(i) * self.y_vars[i] for i in self.model.customers)

        travel_cost = gp.quicksum(self.model.get_edge_cost(i, j) * self.x_vars[(i, j)] for i, j in self.model.edges)

        if self.params.use_saa and self.scenarios:
            self.q_vars = {}
            num_s = len(self.scenarios)
            recourse_cost_expr = 0.0

            # SIRP Scaffold: Recourse cost is penalized excess capacity per scenario.
            # In a full recourse formulation, this might measure failure returns to depot.
            for s_idx, scenario_demands in enumerate(self.scenarios):
                q_s = self.gurobi_model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"q_recourse_{s_idx}")
                self.q_vars[s_idx] = q_s

                # Link recourse to routing baseline decisions
                scenario_load = gp.quicksum(scenario_demands[i] * self.y_vars[i] for i in self.model.customers)
                self.gurobi_model.addConstr(
                    q_s >= scenario_load - (self.model.num_vehicles * self.model.capacity), name=f"recourse_def_{s_idx}"
                )

                # High penalty for capacity violation in scenario (e.g. 10 units of cost)
                recourse_cost_expr += 10.0 * self.model.C * q_s

            expected_recourse = recourse_cost_expr / num_s
            self.gurobi_model.setObjective(travel_cost - waste_collected + expected_recourse, GRB.MINIMIZE)
        else:
            self.gurobi_model.setObjective(travel_cost - waste_collected, GRB.MINIMIZE)

        # Constraints
        # 1. Degree constraints: sum of edges incident to i equals 2*y[i]
        for i in self.model.customers:
            incident_edges = [(u, v) for u, v in self.model.edges if i in (u, v)]
            self.gurobi_model.addConstr(
                gp.quicksum(self.x_vars[e] for e in incident_edges) == 2 * self.y_vars[i],
                name=f"degree_{i}",
            )

        # Depot degree constraint (allows up to K vehicles)
        depot_edges = [(u, v) for u, v in self.model.edges if self.model.depot in (u, v)]
        self.gurobi_model.addConstr(
            gp.quicksum(self.x_vars[e] for e in depot_edges) <= 2 * self.model.num_vehicles,
            name="depot_degree",
        )

        # 2. Mandatory node constraints
        for i in self.model.mandatory_nodes:
            if i in self.y_vars:
                self.gurobi_model.addConstr(self.y_vars[i] == 1, name=f"mandatory_{i}")

        # 3. Global capacity constraint (Knapsack)
        # sum of demands of visited nodes <= K * Q (aggregate capacity)
        self.gurobi_model.addConstr(
            gp.quicksum(self.model.get_node_demand(i) * self.y_vars[i] for i in self.model.customers)
            <= self.model.num_vehicles * self.model.capacity,
            name="global_capacity",
        )

        self.gurobi_model.update()

    def _set_start_solution(self, tour: List[int]):
        """Provide a warm start solution to Gurobi.

        Args:
            tour: List of node indices representing the initial tour.
        """
        # Set x variables based on tour
        for i, j in self.model.edges:
            self.x_vars[(i, j)].Start = 0.0

        for idx in range(len(tour) - 1):
            edge = tuple(sorted([tour[idx], tour[idx + 1]]))
            if edge in self.x_vars:
                # Increment start value to handle using same edge twice (e.g. [0, 1, 0])
                current_start = getattr(self.x_vars[edge], "Start", 0.0)  # type: ignore[index]
                if np.isnan(current_start):  # Gurobi might initialize with NaN
                    current_start = 0.0
                self.x_vars[edge].Start = current_start + 1.0  # type: ignore[index]

        # Set y variables based on visited nodes
        visited_nodes = set(tour) - {self.model.depot}
        for i in self.model.customers:
            self.y_vars[i].Start = 1.0 if i in visited_nodes else 0.0

    def _extract_solution(self) -> Tuple[List[List[int]], float]:
        """Extract the optimal routes from Gurobi solution.

        Handles multiple routes in the aggregate flow formulation.

        Returns:
            A tuple containing:
                - routes: Optimized collection routes (list-of-lists).
                - profit: Total net profit (Total Revenue - Total Cost).
        """
        assert self.gurobi_model is not None
        if self.gurobi_model.SolCount == 0:
            return [], 0.0

        # Extract active edges into a multiset-like list
        active_edges = []
        for (i, j), var in self.x_vars.items():
            val = round(var.X)
            for _ in range(int(val)):
                active_edges.append(list(sorted([i, j])))

        if not active_edges:
            return [], 0.0

        routes = []
        remaining_edges = active_edges

        # Multi-vehicle route extraction: trace all paths starting from depot
        while True:
            # Find an edge incident to depot
            start_edge = None
            for e in remaining_edges:
                if self.model.depot in e:
                    start_edge = e
                    break

            if start_edge is None:
                break

            # Start a new route
            route = [self.model.depot]
            current = start_edge[1] if start_edge[0] == self.model.depot else start_edge[0]
            remaining_edges.remove(start_edge)
            route.append(current)

            while current != self.model.depot:
                found_next = False
                for e in remaining_edges:
                    if current in e:
                        next_node = e[1] if e[0] == current else e[0]
                        remaining_edges.remove(e)
                        route.append(next_node)
                        current = next_node
                        found_next = True
                        break
                if not found_next:
                    # Should not happen if SECs are working correctly
                    break

            # Remove depots from the route for adapter consistency.
            # Local mapping expects routes as List[List[customers]].
            customers_only = [node for node in route if node != self.model.depot]
            if customers_only:
                routes.append(customers_only)
            elif len(route) == 3 and route[1] != self.model.depot:
                # Case [0, i, 0] where i is a single visit
                routes.append([route[1]])

        # Compute total profit (obj value isMinimized travel_cost - waste_collected)
        profit = -self.gurobi_model.ObjVal

        return routes, profit
