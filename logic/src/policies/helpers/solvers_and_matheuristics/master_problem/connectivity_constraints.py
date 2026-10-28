"""connectivity constraints methods relocated from constraints."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Set, Tuple, Union

import gurobipy as gp
import numpy as np

if TYPE_CHECKING:
    from .model import Route
    from .problem_support import MasterProblemSupport

logger = logging.getLogger(__name__)


def _cut_decision(master: Any, cut_type: str, cut_id: Any, accepted: bool) -> bool:
    """Return ``accepted`` and, only when a trace is bound, record the cut."""
    if getattr(master, "_exec_trace", None) is not None:
        from logic.src.policies.helpers.solvers_and_matheuristics.exact_trace import emit_cut

        emit_cut(master, cut_type, cut_id, accepted)
    return accepted


class MasterConnectivityConstraintsMixin:
    """Relocated connectivity constraints methods."""

    def add_sec_cut(
        self: MasterProblemSupport,
        node_list: Union[List[int], Set[int], FrozenSet[int]],
        rhs: float,
        cut_name: str = "",
        global_cut: bool = True,
        node_i: int = -1,
        node_j: int = -1,
        facet_form: str = "2.1",
    ) -> bool:
        """Add a Subtour Elimination Cut (SEC) or PC-SEC to the master problem.

        Args:
            node_list: Set of nodes in the subtour.
            rhs: Right-hand side value.
            cut_name: Optional name for the constraint.
            global_cut: If True, the cut is stored in the global registry.
            node_i: Index of node i for Form 2.2 and 2.3 PC-SECs.
            node_j: Index of node j for Form 2.3 PC-SECs.
            facet_form: Form indicator for PC-SECs (e.g. "2.1", "2.3").

        Returns:
            True if added successfully.
        """

        if self.model is None:
            return False

        node_set = frozenset(node_list)
        registry = self.active_sec_cuts if global_cut else self.active_sec_cuts_local

        if node_set in registry:
            return _cut_decision(self, "sec", node_set, False)

        lhs = gp.LinExpr()
        for idx, route in enumerate(self.routes):
            # We calculate the coefficient C_k for route k in the cut Σ C_k λ_k >= RHS.
            # Form 2.3: Σ x_e >= 2(y_i + y_j - 1)  =>  Σ x_e - 2y_i - 2y_j >= -2.
            # C_k = crossings(route_k, S) - 2 * (1 if i in route_k) - 2 * (1 if j in route_k).
            val = float(self._count_crossings(route, node_set))

            if node_i > 0 and node_i in route.node_coverage:
                val -= 2.0
            if node_j > 0 and node_j in route.node_coverage:
                val -= 2.0

            if abs(val) > 1e-6:
                lhs += val * self.lambda_vars[idx]

        name = cut_name if cut_name else f"sec_{abs(hash(node_set))}"
        constr = self.model.addConstr(lhs >= rhs, name=name)
        registry[node_set] = constr

        # Archiving to Global Pool
        if global_cut and node_i < 0 and node_j < 0:
            # Form 2.1 is always valid Σ x_e >= 2
            self.global_cut_pool.add_cut("sec_2.1", node_set)

        self.model.update()
        return _cut_decision(self, "sec", node_set, True)

    def _count_crossings(self: MasterProblemSupport, route: Route, node_set: FrozenSet[int]) -> int:
        """Counts how many times a route crosses the boundary δ(S).

        Args:
            route: The route to check.
            node_set: The node set S.

        Returns:
            int: Number of times the route enters/leaves S.
        """
        crossings = 0
        path_nodes = [0] + route.nodes + [0]
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            if (u in node_set) != (v in node_set):
                crossings += 1
        return crossings

    def remove_local_cuts(self: MasterProblemSupport) -> int:
        """
        Delete all node-local cuts from the Gurobi model and clear registries.

        Returns:
            Number of cuts removed.
        """
        if self.model is None:
            return 0
        removed = 0
        for constr in self.active_sec_cuts_local.values():
            try:
                self.model.remove(constr)
                removed += 1
            except gp.GurobiError:
                continue

        self.active_sec_cuts_local.clear()
        self.dual_sec_cuts_local: Dict[FrozenSet[int], float] = {}
        self.model.update()
        return removed

    def find_and_add_violated_rcc(
        self: MasterProblemSupport,
        route_values: Dict[int, float],
        routes: List[Route],
        max_cuts: int = 5,
    ) -> int:
        """Separate and add Rounded Capacity Cuts (RCC) based on the current LP solution.

        This follows Section 7 of Barnhart et al. (1998) and Desrochers et al. (1992)
        using a connectivity heuristic:
        1. Build the fractional flow support graph (arcs with x_uv > 0).
        2. Identify connected components (S) of customer nodes.
        3. For each component S, check if the routing bound is violated:
           Σ_{k} x^k(δ(S)) λ_k  <  2 * ⌈ (Σ_{i∈S} waste_i) / Q ⌉

        Args:
            route_values: Current LP values for routes.
            routes: List of routes.
            max_cuts: Maximum number of cuts to add.

        Returns:
            Number of cuts successfully added.
        """

        if not route_values or not routes:
            return 0

        # Aggregate arc flows from active routes
        arc_flow: Dict[Tuple[int, int], float] = {}
        for idx, val in route_values.items():
            if val < 1e-6 or idx >= len(routes):
                continue
            path = [0] + routes[idx].nodes + [0]
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                arc = (u, v)
                arc_flow[arc] = arc_flow.get(arc, 0.0) + val

        # Find connected components of customer nodes in the support graph (ignoring depot)
        components = self._find_customer_components(arc_flow)

        # Evaluate each component S as a cut candidate
        cuts_added = 0
        for S in components:
            if not S:
                continue
            total_waste = sum(self.wastes.get(i, 0.0) for i in S)
            rhs = 2.0 * np.ceil(total_waste / self.capacity) if self.capacity > 0 else 2.0

            # Calculate current LHS value (sum of flow crossing the boundary delta(S))
            lhs_val = sum(flow for (u, v), flow in arc_flow.items() if (u in S) != (v in S))

            # Add if violated by more than 1e-4
            if lhs_val < rhs - 1e-4 and self.add_set_packing_capacity_cut(list(S), rhs):
                cuts_added += 1
                if cuts_added >= max_cuts:
                    break
        return cuts_added

    def _find_customer_components(self: MasterProblemSupport, arc_flow: Dict[Tuple[int, int], float]) -> List[Set[int]]:
        """Identify connected components of customer nodes in the support graph.

        Args:
            arc_flow: Mapping of edges to their fractional flow values.

        Returns:
            List[Set[int]]: List of sets, where each set contains node IDs in a component.
        """

        adj: Dict[int, Set[int]] = {}
        customer_nodes = set()
        for u, v in arc_flow.keys():
            if u != 0 and v != 0:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
                customer_nodes.add(u)
                customer_nodes.add(v)

        visited = set()
        components: List[Set[int]] = []
        for node in customer_nodes:
            if node not in visited:
                comp = set()
                stack = [node]
                while stack:
                    curr = stack.pop()
                    if curr not in visited:
                        visited.add(curr)
                        comp.add(curr)
                        stack.extend(adj.get(curr, set()) - visited)
                components.append(comp)
        return components
