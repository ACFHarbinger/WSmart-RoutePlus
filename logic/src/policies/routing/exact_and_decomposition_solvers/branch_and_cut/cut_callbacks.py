"""Gurobi callback and cut separation of the branch-and-cut solver.

Attributes:
    BCCutSeparationMixin: Lazy and user cuts (PC-SEC, capacity, stochastic capacity, multistar, lot sizing) of ``BranchAndCutSolver``.

Example:
    >>> class BranchAndCutSolver(BCCutSeparationMixin): ...
"""

import warnings
from typing import Any, List

import numpy as np

from logic.src.policies.helpers.solvers_and_matheuristics import (
    CapacityCut,
    PCSubtourEliminationCut,
)

try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    gp: Any = None  # type: ignore[assignment,no-redef]
    GRB: Any = None  # type: ignore[assignment,misc,no-redef]


try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


class BCCutSeparationMixin:
    """Lazy/user cut separation inside the Gurobi callback of :class:`BranchAndCutSolver`."""

    def _lazy_constraint_callback(self, model, where):
        """Gurobi callback for Optimized Branch-and-Cut.

        This method handles both integer solution checks (MIPSOL) and LP relaxation
        node processing (MIPNODE). It coordinates the separation and addition
        of SECs and Capacity Cuts to enforce feasibility and tighten the relaxation.

        Args:
            model: The Gurobi model instance.
            where: Callback context identifier.
        """
        if where == GRB.Callback.MIPSOL:
            # Integer solution: only SECs are mandatory for feasibility
            self._add_integer_cuts(model)

        elif where == GRB.Callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
            # Fractional LP node: add both SECs and Capacity Cuts as User Cuts
            self._add_fractional_cuts(model)

    def _evaluate_pool_cuts(self, x_vals: np.ndarray, y_vals: np.ndarray) -> List[Any]:
        """Re-evaluate stored cuts against the current LP/MIP solution.

        Paper Section 5.1: pool cuts are checked before any separation call.

        Args:
            x_vals: Current edge variable values.
            y_vals: Current node variable values.

        Returns:
            List of cuts from the pool that are violated by the current solution.
        """
        active = []
        for cut in self._cut_pool:
            if isinstance(cut, PCSubtourEliminationCut):
                # node_i and node_j must be customers (> 0) for non-2.1 forms.
                # The depot cannot appear in S for any valid SEC.
                assert cut.node_i != 0 or cut.facet_form == "2.1", (
                    f"Malformed pool cut: node_i=0 for facet_form={cut.facet_form}"
                )
                cut_val = self.separator._get_cut_value(cut.node_set, x_vals)
                if cut.facet_form == "2.1":
                    violation = 2.0 - cut_val
                elif cut.facet_form == "2.2":
                    yi = y_vals[cut.node_i - 1] if cut.node_i > 0 else 1.0
                    violation = 2.0 * yi - cut_val
                else:
                    yi = y_vals[cut.node_i - 1] if cut.node_i > 0 else 1.0
                    yj = y_vals[cut.node_j - 1] if cut.node_j > 0 else 1.0
                    violation = 2.0 * (yi + yj - 1.0) - cut_val
                if violation > 0.01:
                    active.append(cut)
            elif isinstance(cut, CapacityCut):
                cut_val = self.separator._get_cut_value(cut.node_set, x_vals)
                violation = cut.rhs - cut_val
                if violation > 0.01:
                    active.append(cut)  # type: ignore[arg-type]
        return active

    def _handle_custom_branching(self, model):
        """Custom branching on cuts (Section 5.4, Fischetti et al. 1997).

        Note:
            NOT IMPLEMENTED: Gurobi does not expose a programmatic branching
            interface from within Python callbacks. The paper's "branching on cuts"
            strategy (choosing S such that sum_{e in delta(S)} x_e is fractional-odd
            and imposing sum x_e <= 2k or sum x_e >= 2k+2) cannot be enforced via
            cbBranch() — no such Gurobi API method exists.

        Alternative approaches for influencing branching in Gurobi:
          1. Set BranchPriority on x_vars to prefer edges near 0.5.
          2. Use GRB.Param.BranchDir to prefer branching directions.
          3. Implement a custom B&B loop outside Gurobi (not done here).

        Gurobi's default variable branching (most fractional variable) is used
        instead, which corresponds to the paper's fallback strategy.

        Args:
            model: The Gurobi model instance.
        """

    def _add_integer_cuts(self, model):
        """Add violated cuts at integer solutions (MIPSOL callback).

        This method ensures correctness for the Natural Edge Formulation:
        1. Extracts the current integer solution (x_vals, y_vals) via cbGetSolution()
        2. Invokes the SeparationEngine to detect violated inequalities (fast heuristics)
        3. Adds violated cuts as lazy constraints via model.cbLazy()

        Correctness Contract (Fischetti et al. 1997):
            - If the solution contains a disconnected subtour, the SeparationEngine
              MUST detect it and return a SubtourEliminationCut.
            - If the solution violates vehicle capacity constraints for any node set,
              the SeparationEngine MUST detect it and return a CapacityCut.
            - If no violated cuts are found, the solution is guaranteed to be valid.

        Performance:
            - Uses fast connected-components check for SECs (NetworkX-based)
            - Heuristic capacity separation (demand-based clustering)
            - O(n²) complexity, suitable for integer callback

        Args:
            model: The Gurobi model instance.
        """
        # Get current integer solution values using bulk API extraction
        # Performance Optimization: Bulk cbGetSolution() reduces Python-C API overhead
        x_var_list = [self.x_vars[(i, j)] for i, j in self.model.edges]
        y_var_list = [self.y_vars[i] for i in self.model.customers]

        x_vals = np.array(model.cbGetSolution(x_var_list))
        y_vals_array = np.array(model.cbGetSolution(y_var_list))

        # Re-evaluate pool cuts before fresh separation
        pool_cuts = self._evaluate_pool_cuts(x_vals, y_vals_array)
        for cut in pool_cuts:
            if isinstance(cut, PCSubtourEliminationCut):
                self._add_pcsec_lazy(model, cut)
                self.stats["sec_cuts"] += 1
            elif isinstance(cut, CapacityCut):
                self._add_capacity_cut_lazy(model, cut)
                self.stats["capacity_cuts"] += 1
            self.stats["total_cuts"] += 1

        # Separate both SECs and Capacity Cuts at integer solutions.
        # Capacity cuts are required for correctness: a global knapsack constraint
        # does NOT prevent individual routes from violating per-route capacity.
        # sec_only=False ensures capacity-violating integer solutions are rejected.
        iteration = self.stats["lp_iterations"]
        cuts = self.separator.separate_integer(
            x_vals, y_vals=y_vals_array, max_cuts=self.params.max_cuts_per_round, iteration=iteration, sec_only=False
        )

        # Add cuts as lazy constraints
        for cut in cuts:
            if isinstance(cut, PCSubtourEliminationCut):
                self._add_pcsec_lazy(model, cut)
                self.stats["sec_cuts"] += 1
            elif isinstance(cut, CapacityCut):
                self._add_capacity_cut_lazy(model, cut)
                self.stats["capacity_cuts"] += 1

            self.stats["total_cuts"] += 1

        # Store cuts for pool re-evaluation at child nodes (active via _evaluate_pool_cuts).
        for cut in cuts:
            # Signature: (node_set, class_name, facet_form, node_i, node_j)
            node_set = getattr(cut, "node_set", set())
            sig = (
                frozenset(node_set) if node_set else frozenset(),
                cut.__class__.__name__,
                getattr(cut, "facet_form", None),
                getattr(cut, "node_i", None),
                getattr(cut, "node_j", None),
            )
            if sig not in self._cut_signatures:
                self._cut_pool.append(cut)
                self._cut_signatures.add(sig)

        # SAA SIRP separation scaffolds for integer feasibility
        if self.params.use_saa and self.scenarios:
            self._separate_stochastic_capacity_cuts(model, x_vals, y_vals_array, is_integer=True)
            self._separate_multi_star_inequalities(model, x_vals, y_vals_array, is_integer=True)
            self._separate_lot_sizing_inequalities(model, x_vals, y_vals_array, is_integer=True)

        self.stats["lp_iterations"] += 1

    def _add_fractional_cuts(self, model):
        """Add violated cuts at fractional LP relaxation nodes (MIPNODE callback).

        DIVERGENCE FROM PAPER (Section 5.1, procedure SEPARATION):
        The paper separates in this order:
          1. Pool re-evaluation (cuts found at ancestor nodes)
          2. Fan inequalities (Equation 2.12) — NOT IMPLEMENTED HERE
          3. GSEC_H2 heuristic
          4. GSEC_H1 heuristic
          5. Per-threshold ε ∈ {0.1, 0.01, 0.001} combined pool check
          6. GSEC_SEP exact separation (forced every 10th node)
          7. Generalised comb heuristics — NOT IMPLEMENTED HERE

        Args:
            model: The Gurobi model instance.
        """
        # Get current node count (depth indicator)
        node_count = model.cbGet(GRB.Callback.MIPNODE_NODCNT)

        # Get fractional solution values using bulk API extraction
        # Performance Optimization: Bulk cbGetNodeRel() is O(n) vs O(n²) for loop-based extraction
        # This reduces Python-C API overhead from ~1000 calls to 2 calls for n=50 instances
        x_var_list = [self.x_vars[(i, j)] for i, j in self.model.edges]
        y_var_list = [self.y_vars[i] for i in self.model.customers]

        x_vals = np.array(model.cbGetNodeRel(x_var_list))
        y_vals_array = np.array(model.cbGetNodeRel(y_var_list))

        # Re-evaluate pool cuts before fresh separation
        pool_cuts = self._evaluate_pool_cuts(x_vals, y_vals_array)
        for cut in pool_cuts:
            if isinstance(cut, PCSubtourEliminationCut):
                self._add_pcsec_user(model, cut)
                self.stats["sec_cuts"] += 1
            elif isinstance(cut, CapacityCut):
                self._add_capacity_cut_user(model, cut)
                self.stats["capacity_cuts"] += 1
            self.stats["total_cuts"] += 1

        # Separate violated inequalities (fractional mode: exact max-flow separation)
        self.stats["fractional_iterations"] += 1
        iteration = int(self.stats["fractional_iterations"])
        max_cuts = self.params.max_cuts_per_round // 2  # Limit cuts at fractional nodes

        cuts = self.separator.separate_fractional(
            x_vals, y_vals=y_vals_array, max_cuts=max_cuts, iteration=iteration, node_count=node_count
        )

        # Store cuts for pool re-evaluation at child nodes (active via _evaluate_pool_cuts).
        for cut in cuts:
            # Signature: (node_set, class_name, facet_form, node_i, node_j)
            node_set = getattr(cut, "node_set", set())
            sig = (
                frozenset(node_set) if node_set else frozenset(),
                cut.__class__.__name__,
                getattr(cut, "facet_form", None),
                getattr(cut, "node_i", None),
                getattr(cut, "node_j", None),
            )
            if sig not in self._cut_signatures:
                self._cut_pool.append(cut)
                self._cut_signatures.add(sig)

        # Add cuts as user cuts (not lazy - for LP relaxation strengthening)
        for cut in cuts:
            if isinstance(cut, PCSubtourEliminationCut):
                self._add_pcsec_user(model, cut)
                self.stats["sec_cuts"] += 1
            elif isinstance(cut, CapacityCut):
                self._add_capacity_cut_user(model, cut)
                self.stats["capacity_cuts"] += 1

            self.stats["total_cuts"] += 1

        # SAA SIRP separation scaffolds for fractional LP tightening
        if self.params.use_saa and self.scenarios:
            self._separate_stochastic_capacity_cuts(model, x_vals, y_vals_array, is_integer=False)
            self._separate_multi_star_inequalities(model, x_vals, y_vals_array, is_integer=False)
            self._separate_lot_sizing_inequalities(model, x_vals, y_vals_array, is_integer=False)

    def _add_pcsec_lazy(self, model, cut: PCSubtourEliminationCut):
        """Add a prize-collecting subtour elimination cut (PC-SEC) as a lazy constraint.

        Lazy constraints are only checked for integer solutions and are essential
        for correctness in the Natural Edge Formulation.

        Args:
            model: The Gurobi model instance.
            cut: The subtour elimination cut object to add.
        """
        cut_edges = self.model.delta(cut.node_set)
        edge_vars = [self.x_vars[tuple(sorted(e))] for e in cut_edges if tuple(sorted(e)) in self.x_vars]  # type: ignore[index]

        if edge_vars:
            # Fischetti et al. (1997) Facet Forms
            if cut.facet_form == "2.1":
                model.cbLazy(gp.quicksum(edge_vars) >= 2.0)
            elif cut.facet_form == "2.2":
                yi = self.y_vars[cut.node_i]
                model.cbLazy(gp.quicksum(edge_vars) >= 2 * yi)
            else:  # Form 2.3
                yi = self.y_vars[cut.node_i]
                yj = self.y_vars.get(cut.node_j, 1.0)
                model.cbLazy(gp.quicksum(edge_vars) >= 2 * (yi + yj - 1.0))

    def _add_capacity_cut_lazy(self, model, cut: CapacityCut):
        """Add a capacity cut as a lazy constraint (integer callback).

        Lazy constraints are only checked for integer solutions and are essential
        for correctness when capacity violations cannot be detected by base constraints.

        Args:
            model: The Gurobi model instance.
            cut: The capacity cut object to add.
        """
        cut_edges = self.model.delta(cut.node_set)
        edge_vars = [self.x_vars[tuple(sorted(e))] for e in cut_edges if tuple(sorted(e)) in self.x_vars]  # type: ignore[index]

        if edge_vars:
            model.cbLazy(gp.quicksum(edge_vars) >= cut.rhs)

    def _add_pcsec_user(self, model, cut: PCSubtourEliminationCut):
        """Add a prize-collecting subtour elimination cut (PC-SEC) as a user cut.

        User cuts strengthen the LP relaxation at fractional nodes but are not
        required for correctness (unlike lazy constraints).

        Args:
            model: The Gurobi model instance.
            cut: The subtour elimination cut object to add.
        """
        cut_edges = self.model.delta(cut.node_set)
        edge_vars = [self.x_vars[tuple(sorted(e))] for e in cut_edges if tuple(sorted(e)) in self.x_vars]  # type: ignore[index]

        if edge_vars:
            # Fischetti et al. (1997) Facet Forms
            if cut.facet_form == "2.1":
                model.cbCut(gp.quicksum(edge_vars) >= 2.0)
            elif cut.facet_form == "2.2":
                yi = self.y_vars[cut.node_i]
                model.cbCut(gp.quicksum(edge_vars) >= 2 * yi)
            else:  # Form 2.3
                yi = self.y_vars[cut.node_i]
                yj = self.y_vars.get(cut.node_j, 1.0)
                model.cbCut(gp.quicksum(edge_vars) >= 2 * (yi + yj - 1.0))

    def _add_capacity_cut_user(self, model, cut: CapacityCut):
        """Add a capacity cut as a user cut (fractional callback).

        User cuts strengthen the LP relaxation at fractional nodes but are not
        required for correctness.

        Args:
            model: The Gurobi model instance.
            cut: The capacity cut object to add.
        """
        cut_edges = self.model.delta(cut.node_set)
        edge_vars = [self.x_vars[tuple(sorted(e))] for e in cut_edges if tuple(sorted(e)) in self.x_vars]  # type: ignore[index]

        if edge_vars:
            model.cbCut(gp.quicksum(edge_vars) >= cut.rhs)

    def _separate_stochastic_capacity_cuts(self, model, x_vals: np.ndarray, y_vals: np.ndarray, is_integer: bool):
        """Separate Stochastic Capacity Inequalities (SIRP).

        Evaluates scenarios to find subsets where expected demand frequently exceeds capacity,
        adding cuts that force additional vehicles or node shedding.

        Args:
            model: The Gurobi model instance.
            x_vals: Current edge variable values.
            y_vals: Current node variable values.
            is_integer: Whether the current solution is integer (MIPSOL) or fractional (MIPNODE).
        """
        if not hasattr(self, "scenarios") or not self.scenarios:
            return

        # Find connected components of fractional support graph (edges > 0.1)
        support_edges = [(u, v) for (u, v), val in zip(self.model.edges, x_vals, strict=False) if val > 0.1]
        if NETWORKX_AVAILABLE:
            G = nx.Graph()
            G.add_nodes_from(range(self.model.n_nodes))
            G.add_edges_from(support_edges)
            components = list(nx.connected_components(G))
        else:
            warnings.warn("NetworkX not available, skipping stochastic capacity cuts", stacklevel=2)
            return  # Requires networkx

        num_s = len(self.scenarios)
        avg_demands = {i: sum(scen.get(i, 0.0) for scen in self.scenarios) / num_s for i in self.model.customers}
        y_val_dict = {i: y_vals[idx] for idx, i in enumerate(self.model.customers)}
        edges_list = list(self.model.edges)

        for comp in components:
            if self.model.depot in comp:
                continue

            S = list(comp)
            if len(S) <= 1:
                continue

            expected_demand = sum(avg_demands.get(i, 0.0) * y_val_dict.get(i, 0.0) for i in S)  # type: ignore[no-matching-overload]

            if expected_demand > 0:
                req_capacity = (2.0 * expected_demand) / self.model.capacity

                delta_S_edges = [(u, v) for u, v in self.model.edges if (u in S) != (v in S)]
                delta_S_val = sum(x_vals[edges_list.index((u, v))] for u, v in delta_S_edges)

                if delta_S_val < req_capacity - 1e-4:
                    edge_vars = [
                        self.x_vars[tuple(sorted(e))]  # type: ignore[index]
                        for e in delta_S_edges
                        if tuple(sorted(e)) in self.x_vars
                    ]
                    y_vars_S = [self.y_vars[i] for i in S if i in self.y_vars]
                    demand_S = [avg_demands.get(i, 0.0) for i in S if i in self.y_vars]

                    if edge_vars and y_vars_S:
                        cut_expr = gp.quicksum(edge_vars) >= (2.0 / self.model.capacity) * gp.quicksum(
                            d * y for d, y in zip(demand_S, y_vars_S, strict=False)
                        )
                        if is_integer:
                            model.cbLazy(cut_expr)
                        else:
                            model.cbCut(cut_expr)
                        self.stats["total_cuts"] += 1

    def _separate_multi_star_inequalities(self, model, x_vals: np.ndarray, y_vals: np.ndarray, is_integer: bool):
        """Separate Multi-star Inequalities (SIRP).

        Strengthens the routing structure around subsets of correlated high-demand nodes
        across multiple stochastic scenarios.

        Args:
            model: The Gurobi model instance.
            x_vals: Current edge variable values.
            y_vals: Current node variable values.
            is_integer: Whether the current solution is integer (MIPSOL) or fractional (MIPNODE).
        """
        if not hasattr(self, "scenarios") or not self.scenarios:
            return

        num_s = len(self.scenarios)
        avg_demands = {i: sum(scen.get(i, 0.0) for scen in self.scenarios) / num_s for i in self.model.customers}
        y_val_dict = {i: y_vals[idx] for idx, i in enumerate(self.model.customers)}
        edges_list = list(self.model.edges)

        # Identify "nucleus" nodes: heavily visited with significant demand
        for _idx, i in enumerate(self.model.customers):
            if y_val_dict[i] > 0.5 and avg_demands.get(i, 0.0) > self.model.capacity / 3.0:
                star_edges = []
                star_nodes = {i}
                for j in self.model.customers:
                    if i != j:
                        edge = (min(i, j), max(i, j))
                        if edge in self.model.edges:
                            e_idx = edges_list.index(edge)
                            if x_vals[e_idx] > 0.1:
                                star_edges.append(edge)
                                star_nodes.add(j)

                if len(star_nodes) >= 3:
                    expected_hub_demand = sum(avg_demands.get(k, 0.0) * y_val_dict.get(k, 0.0) for k in star_nodes)
                    if expected_hub_demand > self.model.capacity:
                        edge_vars = [self.x_vars[e] for e in star_edges if e in self.x_vars]
                        y_vars_hub = [self.y_vars[k] for k in star_nodes if k in self.y_vars]
                        demand_hub = [avg_demands.get(k, 0.0) for k in star_nodes if k in self.y_vars]

                        if edge_vars and y_vars_hub:
                            bound = gp.quicksum(y for y in y_vars_hub) - (1.0 / self.model.capacity) * gp.quicksum(
                                d * y for d, y in zip(demand_hub, y_vars_hub, strict=False)
                            )
                            cut_expr = gp.quicksum(edge_vars) <= bound
                            if is_integer:
                                model.cbLazy(cut_expr)
                            else:
                                model.cbCut(cut_expr)
                            self.stats["total_cuts"] += 1

    def _separate_lot_sizing_inequalities(self, model, x_vals: np.ndarray, y_vals: np.ndarray, is_integer: bool):
        """Separate inequalities derived from Deterministic Lot-Sizing problems.

        Limits combinations of disjoint routes that consistently produce capacity failures
        in extreme scenarios by linking fractional routing capacity to continuous recourse.

        Args:
            model: The Gurobi model instance.
            x_vals: Current edge variable values.
            y_vals: Current node variable values.
            is_integer: Whether the current solution is integer (MIPSOL) or fractional (MIPNODE).
        """
        if not hasattr(self, "q_vars") or not self.q_vars or not self.scenarios:
            return

        support_edges = [(u, v) for (u, v), val in zip(self.model.edges, x_vals, strict=False) if val > 0.1]
        if NETWORKX_AVAILABLE:
            G = nx.Graph()
            G.add_nodes_from(range(self.model.n_nodes))
            G.add_edges_from(support_edges)
            components = list(nx.connected_components(G))
        else:
            warnings.warn("NetworkX not available, skipping lot-sizing cuts", stacklevel=2)
            return

        y_val_dict = {i: y_vals[idx] for idx, i in enumerate(self.model.customers)}
        edges_list = list(self.model.edges)

        for s_idx, q_var in self.q_vars.items():
            scenario_demands = self.scenarios[s_idx]
            q_val = model.cbGetSolution(q_var) if is_integer else model.cbGetNodeRel(q_var)

            for comp in components:
                if self.model.depot in comp:
                    continue

                S = list(comp)
                if len(S) <= 1:
                    continue

                scenario_load = sum(scenario_demands.get(i, 0.0) * y_val_dict.get(i, 0.0) for i in S)  # type: ignore[no-matching-overload]

                delta_S_edges = [(u, v) for u, v in self.model.edges if (u in S) != (v in S)]
                delta_S_val = sum(x_vals[edges_list.index((u, v))] for u, v in delta_S_edges)
                provided_capacity = (delta_S_val / 2.0) * self.model.capacity

                if scenario_load - provided_capacity > q_val + 1e-4:
                    edge_vars = [
                        self.x_vars[tuple(sorted(e))]  # type: ignore[index]
                        for e in delta_S_edges
                        if tuple(sorted(e)) in self.x_vars
                    ]
                    y_vars_S = [self.y_vars[i] for i in S if i in self.y_vars]
                    demand_S = [scenario_demands.get(i, 0.0) for i in S if i in self.y_vars]

                    if edge_vars and y_vars_S:
                        cut_expr = q_var >= gp.quicksum(d * y for d, y in zip(demand_S, y_vars_S, strict=False)) - (
                            self.model.capacity / 2.0
                        ) * gp.quicksum(edge_vars)
                        if is_integer:
                            model.cbLazy(cut_expr)
                        else:
                            model.cbCut(cut_expr)
                        self.stats["total_cuts"] += 1
