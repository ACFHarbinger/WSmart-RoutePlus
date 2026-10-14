r"""Logic-Based Benders Decomposition (LBBD) Master Problem.

Implements the master allocation problem for LBBD, distributing bins
to different days of the planning horizon while considering predicted
waste levels and overflow penalties.

Attributes:
    LBBDMasterProblem: Master Problem for Logic-Based Benders Decomposition.

Example:
    >>> master = LBBDMasterProblem(config, num_customers)
    >>> master.build(tree)
    >>> assignments, thetas = master.solve()
"""

from typing import Any, Dict, List, Optional, Tuple

try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    gp: Any = None  # type: ignore[assignment,no-redef]
    GRB: Any = None  # type: ignore[assignment,misc,no-redef]

from logic.src.configs.policies.lbbd import LBBDConfig
from logic.src.pipeline.simulations.bins.prediction import ScenarioTree, ScenarioTreeNode


class LBBDMasterProblem:
    r"""Master Problem for Logic-Based Benders Decomposition (LBBD).

    Modelled as an allocation problem across multiple days.

    Attributes:
        config (LBBDConfig): Configuration parameters.
        num_customers (int): Number of customers.
        num_nodes (int): Total number of nodes (including depot).
        customers (List[int]): List of customer indices.
        model (gp.Model): Gurobi model instance.
        current_horizon (int): Number of days in the planning horizon.
    """

    def __init__(self, config: LBBDConfig, num_customers: int):
        """Initialize the LBBD Master Problem.

        Args:
            config (LBBDConfig): Configuration parameters for LBBD.
            num_customers (int): Number of customers (bins) in the problem.
        """
        self.config = config
        self.num_customers = num_customers
        self.num_nodes = num_customers + 1
        self.customers = list(range(1, self.num_nodes))

        self.model: gp.Model
        self._z_vars: Dict[Tuple[int, int], Any] = {}  # (node_id, day)
        self._theta_vars: Dict[int, Any] = {}  # (day)

        # Stochastic variables
        self._q_vars: Dict[Any, Any] = {}  # (node_id, day) or (node_id, tree_node)
        self._w_vars: Dict[Any, Any] = {}
        self._of_vars: Dict[Any, Any] = {}

        self._benders_cuts_count = 0

    def build(self, tree: ScenarioTree, stochastic_master: bool = False) -> None:
        """Builds the allocation model using the ScenarioTree.

        Args:
            tree (ScenarioTree): Scenario tree providing predicted wastes.
            stochastic_master (bool): If True, build an Extensive Form (EF) model.
                                     If False, build an Expected Value (EV) model.
        """
        if not GUROBI_AVAILABLE:
            raise ImportError("Gurobi is required for LBBD Master Problem.")

        self.model = gp.Model("LBBD_Master")
        self.model.Params.OutputFlag = 0
        self.model.Params.LogToConsole = 0
        self.model.Params.MIPGap = self.config.mip_gap

        self.current_horizon = min(self.config.num_days, tree.horizon + 1)
        horizon = self.current_horizon

        # 1. Decision Variables (Allocation is scenario-independent for robust scheduling)
        for d in range(1, horizon + 1):
            self._theta_vars[d] = self.model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"theta_d{d}")
            for i in self.customers:
                self._z_vars[i, d] = self.model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{d}")

        # 2. State Variables and Constraints
        profit = self._build_ev_model(tree, horizon) if not stochastic_master else self._build_ef_model(tree, horizon)

        travel_cost = gp.quicksum(self.config.cost_weight * self._theta_vars[d] for d in range(1, horizon + 1))
        self.model.setObjective(profit - travel_cost, GRB.MAXIMIZE)
        self.model.update()

    def _build_ev_model(self, tree: ScenarioTree, horizon: int) -> Any:
        """Build the Expected Value (EV) version of the master problem.

        Args:
            tree (ScenarioTree): The scenario tree.
            horizon (int): Planning horizon in days.

        Returns:
            Any: The Gurobi expression for the expected profit.
        """
        # Expected Value Mode: Single variable per (bin, day)
        for d in range(1, horizon + 1):
            for i in self.customers:
                self._q_vars[i, d] = self.model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"q_{i}_{d}")
                self._w_vars[i, d] = self.model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"w_{i}_{d}")
                self._of_vars[i, d] = self.model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"of_{i}_{d}")

        # Constraints for EV
        for d in range(1, horizon + 1):
            self.model.addConstr(gp.quicksum(self._q_vars[i, d] for i in self.customers) <= 1.0, name=f"cap_d{d}")
            for i in self.customers:
                if d == 1:
                    w_prev = float(tree.root.wastes[i - 1])
                    q_prev = 0.0
                    increment = 0.0  # root already has wastes
                else:
                    w_prev = self._w_vars[i, d - 1]
                    q_prev = self._q_vars[i, d - 1]
                    # Expected increment at day d-1
                    nodes_next = tree.get_scenarios_at_day(d - 1)
                    nodes_prev = tree.get_scenarios_at_day(d - 2)

                    p_next = sum(n.probability for n in nodes_next)
                    avg_wastes_curr = sum(n.probability * n.wastes[i - 1] for n in nodes_next) / p_next

                    p_prev = sum(n.probability for n in nodes_prev)
                    avg_wastes_prev = sum(n.probability * n.wastes[i - 1] for n in nodes_prev) / p_prev

                    increment = max(0.0, avg_wastes_curr - avg_wastes_prev)

                self.model.addConstr(self._w_vars[i, d] == w_prev - q_prev + increment, name=f"inv_{i}_{d}")
                self.model.addConstr(self._q_vars[i, d] <= self._w_vars[i, d], name=f"q_max_{i}_{d}")
                self.model.addConstr(self._q_vars[i, d] <= 2.0 * self._z_vars[i, d], name=f"q_z_{i}_{d}")
                self.model.addConstr(self._of_vars[i, d] >= self._w_vars[i, d] - 1.0, name=f"of_{i}_{d}")

        return gp.quicksum(
            self.config.waste_weight * self._q_vars[i, d] - self.config.overflow_penalty * self._of_vars[i, d]
            for i in self.customers
            for d in range(1, horizon + 1)
        )

    def _build_ef_model(self, tree: ScenarioTree, horizon: int) -> Any:
        """Build the Extensive Form (EF) version of the master problem.

        Args:
            tree (ScenarioTree): The scenario tree.
            horizon (int): Planning horizon in days.

        Returns:
            Any: The Gurobi expression for the expected profit.
        """
        # Extensive Form Mode: Variable per (bin, tree_node)
        node_map: List[ScenarioTreeNode] = []

        def collect(n):
            """Recursively collect all nodes in the scenario tree into node_map.

            Args:
                n (ScenarioTreeNode): The current node to collect.
            """
            node_map.append(n)
            for c in n.children:
                collect(c)

        collect(tree.root)

        # Map node to variables
        for node in node_map:
            d = node.day + 1
            if d > horizon:
                continue
            for i in self.customers:
                node_id = id(node)
                self._q_vars[i, node] = self.model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"q_{i}_{d}_s{node_id}")
                self._w_vars[i, node] = self.model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"w_{i}_{d}_s{node_id}")
                self._of_vars[i, node] = self.model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"of_{i}_{d}_s{node_id}")

        # Constraints per scenario node
        for node in node_map:
            d = node.day + 1
            if d > horizon:
                continue

            # Capacity per scenario
            self.model.addConstr(gp.quicksum(self._q_vars[i, node] for i in self.customers) <= 1.0)

            parent: Optional[ScenarioTreeNode] = None

            def find_parent(curr: ScenarioTreeNode, target: ScenarioTreeNode):
                """Find the parent of target node starting from curr.

                Args:
                    curr (ScenarioTreeNode): Current node being checked.
                    target (ScenarioTreeNode): Target node whose parent is sought.
                """
                nonlocal parent
                for c in curr.children:
                    if c == target:
                        parent = curr
                        return
                    find_parent(c, target)

            if d > 1:
                find_parent(tree.root, node)

            for i in self.customers:
                if d == 1:
                    w_prev = float(tree.root.wastes[i - 1])
                    q_prev = 0.0
                    increment = 0.0
                else:
                    assert parent is not None
                    w_prev = self._w_vars[i, parent]
                    q_prev = self._q_vars[i, parent]
                    increment = max(0.0, node.wastes[i - 1] - parent.wastes[i - 1])

                self.model.addConstr(self._w_vars[i, node] == w_prev - q_prev + increment)
                self.model.addConstr(self._q_vars[i, node] <= self._w_vars[i, node])
                self.model.addConstr(self._q_vars[i, node] <= 2.0 * self._z_vars[i, d])
                self.model.addConstr(self._of_vars[i, node] >= self._w_vars[i, node] - 1.0)

        return gp.quicksum(
            node.probability
            * (self.config.waste_weight * self._q_vars[i, node] - self.config.overflow_penalty * self._of_vars[i, node])
            for node in node_map
            if node.day + 1 <= horizon
            for i in self.customers
        )

    def add_nogood_cut(self, day: int, assigned_nodes: List[int]) -> None:
        """Adds a cut preventing this set of nodes from being assigned together on this day.

        Args:
            day (int): The day for which the cut is added.
            assigned_nodes (List[int]): The list of nodes assigned to this day.
        """
        subset = [i for i in assigned_nodes if i != 0]
        if not subset:
            return

        self._benders_cuts_count += 1
        self.model.addConstr(
            gp.quicksum(self._z_vars[i, day] for i in subset) <= len(subset) - 1,
            name=f"nogood_{day}_{self._benders_cuts_count}",
        )
        self.model.update()

    def add_optimality_cut(self, day: int, assigned_nodes: List[int], distance: float) -> None:
        """Adds a logic-based optimality cut: theta >= Dist * (1 - sum(1-z_i) - sum(z_j)).

        This is a standard LBBD optimality cut for routing.

        Args:
            day (int): The day for which the cut is added.
            assigned_nodes (List[int]): The list of nodes assigned to this day.
            distance (float): The optimal distance for the subproblem.
        """
        subset = [i for i in assigned_nodes if i != 0]
        others = [i for i in self.customers if i not in subset]

        self._benders_cuts_count += 1
        # theta >= Dist * (1 - sum_{i in S}(1 - z_i) - sum_{j not in S}(z_j))
        # Simplified: theta >= Dist - Dist * (sum_{i in S}(1 - z_i) + sum_{j not in S}(z_j))
        cut_expr = distance * (
            1
            - gp.quicksum(1 - self._z_vars[i, day] for i in subset)
            - gp.quicksum(self._z_vars[j, day] for j in others)
        )
        self.model.addConstr(self._theta_vars[day] >= cut_expr, name=f"opt_cut_{day}_{self._benders_cuts_count}")
        self.model.update()

    def solve(self) -> Tuple[Dict[int, List[int]], Dict[int, float]]:
        """Solves the master problem and returns assignments and theta values.

        Returns:
            Tuple[Dict[int, List[int]], Dict[int, float]]: A tuple containing:
                - assignments: Dict mapping day index to list of assigned node IDs.
                - thetas: Dict mapping day index to value of theta variable.
        """
        self.model.optimize()

        if self.model.Status != GRB.OPTIMAL and self.model.SolCount == 0:
            return {}, {}

        assignments = {}
        thetas = {}
        for d in range(1, self.current_horizon + 1):
            assignments[d] = [i for i in self.customers if self._z_vars[i, d].X > 0.5]
            thetas[d] = self._theta_vars[d].X

        return assignments, thetas
