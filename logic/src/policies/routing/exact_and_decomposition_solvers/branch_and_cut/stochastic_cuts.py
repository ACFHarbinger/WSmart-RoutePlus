"""stochastic cuts methods relocated from cut_callbacks."""

import warnings
from typing import Any

import numpy as np

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


class StochasticCutsMixin:
    """Relocated stochastic cuts methods."""

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
