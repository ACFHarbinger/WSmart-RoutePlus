"""master callbacks methods relocated from master_problem."""

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
    GRB = None  # type: ignore[assignment,misc,no-redef]


class MasterCallbacksMixin:
    """Relocated master callbacks methods."""

    def _lazy_constraint_callback(self, model: Any, where: int) -> None:
        """Gurobi lazy/user cut callback for SEC and RCC separation.

        Dispatches to integer-cut separation at MIPSOL events and to fractional
        LP-cut separation at MIPNODE events, mirroring the BranchAndCutSolver
        callback structure.

        Args:
            model: Gurobi model reference passed by the optimize() framework.
            where: Gurobi callback location constant (GRB.Callback.*).
        """
        assert GRB is not None

        if where == GRB.Callback.MIPSOL:
            self._add_integer_cuts(model)
        elif where == GRB.Callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
            self._add_fractional_cuts(model)

    def _add_integer_cuts(self, model: Any) -> None:
        """Separate and add violated SEC / RCC cuts at an integer solution (MIPSOL).

        Args:
            model: Gurobi model handle from the callback.
        """
        assert GUROBI_AVAILABLE and gp is not None and GRB is not None

        x_var_list = [self._x_vars[(i, j)] for i, j in self.model.edges]
        y_var_list = [self._y_vars[i] for i in self.model.customers]

        x_vals = np.array(model.cbGetSolution(x_var_list))
        y_vals = np.array(model.cbGetSolution(y_var_list))

        cuts = self.separator.separate_integer(
            x_vals,
            y_vals=y_vals,
            max_cuts=self.params.max_cuts_per_round,
            iteration=int(self.stats["sec_cuts"]),
            sec_only=False,
        )

        for cut in cuts:
            if isinstance(cut, PCSubtourEliminationCut):
                cut_edges = self.model.delta(cut.node_set)
                edge_vars = [self._x_vars[tuple(sorted(e))] for e in cut_edges if tuple(sorted(e)) in self._x_vars]  # type: ignore[bad-index,index]
                if edge_vars:
                    if cut.facet_form == "2.1":
                        model.cbLazy(gp.quicksum(edge_vars) >= 2.0)
                    elif cut.facet_form == "2.2":
                        yi = self._y_vars[cut.node_i]
                        model.cbLazy(gp.quicksum(edge_vars) >= 2 * yi)
                    else:  # Form 2.3
                        yi = self._y_vars[cut.node_i]
                        yj = self._y_vars.get(cut.node_j, 1.0)
                        model.cbLazy(gp.quicksum(edge_vars) >= 2 * (yi + yj - 1.0))
                self.stats["sec_cuts"] += 1

            elif isinstance(cut, CapacityCut):
                cut_edges = self.model.delta(cut.node_set)
                edge_vars = [self._x_vars[tuple(sorted(e))] for e in cut_edges if tuple(sorted(e)) in self._x_vars]  # type: ignore[bad-index,index]
                if edge_vars:
                    model.cbLazy(gp.quicksum(edge_vars) >= cut.rhs)
                self.stats["capacity_cuts"] += 1

    def _add_fractional_cuts(self, model: Any) -> None:
        """Separate and add user cuts at a fractional LP node (MIPNODE).

        Args:
            model: Gurobi model handle from the callback.
        """
        assert GUROBI_AVAILABLE and gp is not None and GRB is not None

        x_var_list = [self._x_vars[(i, j)] for i, j in self.model.edges]
        y_var_list = [self._y_vars[i] for i in self.model.customers]

        x_vals = np.array(model.cbGetNodeRel(x_var_list))
        y_vals = np.array(model.cbGetNodeRel(y_var_list))

        node_count = int(model.cbGet(GRB.Callback.MIPNODE_NODCNT))
        cuts = self.separator.separate_fractional(
            x_vals,
            y_vals=y_vals,
            max_cuts=self.params.max_cuts_per_round // 2,
            iteration=int(self.stats["sec_cuts"]),
            node_count=node_count,
        )

        for cut in cuts:
            if isinstance(cut, PCSubtourEliminationCut):
                cut_edges = self.model.delta(cut.node_set)
                edge_vars = [self._x_vars[tuple(sorted(e))] for e in cut_edges if tuple(sorted(e)) in self._x_vars]  # type: ignore[bad-index,index]
                if edge_vars:
                    if cut.facet_form == "2.1":
                        model.cbCut(gp.quicksum(edge_vars) >= 2.0)
                    elif cut.facet_form == "2.2":
                        yi = self._y_vars[cut.node_i]
                        model.cbCut(gp.quicksum(edge_vars) >= 2 * yi)
                    else:  # Form 2.3
                        yi = self._y_vars[cut.node_i]
                        yj = self._y_vars.get(cut.node_j, 1.0)
                        model.cbCut(gp.quicksum(edge_vars) >= 2 * (yi + yj - 1.0))

            elif isinstance(cut, CapacityCut):
                cut_edges = self.model.delta(cut.node_set)
                edge_vars = [self._x_vars[tuple(sorted(e))] for e in cut_edges if tuple(sorted(e)) in self._x_vars]  # type: ignore[bad-index,index]
                if edge_vars:
                    model.cbCut(gp.quicksum(edge_vars) >= cut.rhs)

    def _extract_routes(self) -> List[List[int]]:
        """Extract customer-only route lists from the current Gurobi solution.

        Traces all vehicle paths starting from the depot by following the
        active arc edges.  Matches the multi-vehicle extraction logic in
        BranchAndCutSolver._extract_solution().

        Returns:
            List of routes, each a list of customer indices (depot excluded).
        """
        assert self._gurobi_model is not None

        if self._gurobi_model.SolCount == 0:
            return []

        # Build a multiset of active (sorted) edges
        active_edges: List[List[int]] = []
        for (i, j), var in self._x_vars.items():
            val = round(var.X)
            for _ in range(int(val)):
                active_edges.append([min(i, j), max(i, j)])

        if not active_edges:
            return []

        routes: List[List[int]] = []
        remaining = list(active_edges)

        while True:
            # Find a depot-incident edge to start a new route
            start_edge = next((e for e in remaining if self.model.depot in e), None)
            if start_edge is None:
                break

            route = [self.model.depot]
            current = start_edge[1] if start_edge[0] == self.model.depot else start_edge[0]
            remaining.remove(start_edge)
            route.append(current)

            while current != self.model.depot:
                found = False
                for e in remaining:
                    if current in e:
                        nxt = e[1] if e[0] == current else e[0]
                        remaining.remove(e)
                        route.append(nxt)
                        current = nxt
                        found = True
                        break
                if not found:
                    break

            customers_only = [n for n in route if n != self.model.depot]
            if customers_only:
                routes.append(customers_only)

        return routes
