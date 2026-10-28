"""Capacity-family cuts: rounded capacity (RCC), knapsack cover and rounded multistar.

Attributes:
    RoundedCapacityCutEngine: RCC separation (Lysgaard et al. 2004).
    KnapsackCoverEngine: Knapsack cover inequalities.
    RoundedMultistarCutEngine: Rounded multistar inequalities.

Example:
    >>> RoundedCapacityCutEngine(v_model, sep_engine).separate_and_add_cuts(master, max_cuts=10)
"""

from typing import Any, Dict, List, Set, Tuple

import gurobipy as gp
import numpy as np

from logic.src.policies.helpers.solvers_and_matheuristics.master_problem import VRPPMasterProblem
from logic.src.policies.helpers.solvers_and_matheuristics.search.cut_engine_base import CuttingPlaneEngine
from logic.src.policies.helpers.solvers_and_matheuristics.separation import (
    CapacityCut,
    PCSubtourEliminationCut,
    SeparationEngine,
)
from logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model import VRPPModel


class RoundedCapacityCutEngine(CuttingPlaneEngine):
    r"""Rounded Capacity Cut (RCC) separation engine for VRPP Set Partitioning.

    Derived from the physical payload capacity (Q) of the fleet, RCCs are a
    standard family of valid inequalities for the Capacitated Vehicle Routing
    Problem (CVRP), adapted here for the VRPP Set Partitioning context.
    They enforce that the total flow crossing the boundary of a subset of nodes
    S must be sufficient to cover the total demand of S.

    Mathematical Formulation:
    -------------------------
    For a cluster of nodes S ⊆ N \ {0}, the RCC is defined as:
        ∑_{e ∈ δ(S)} x_e ≥ 2 * k(S)

    where:
    - δ(S) = {(i, j) ∈ E : i ∈ S, j ∉ S or i ∉ S, j ∈ S} is the edge cutset of S.
    - डिमांड(S) = ∑_{i ∈ S} q_i is the total demand requirement in S.
    - k(S) = ⌈ डिमांड(S) / Q ⌉ is the minimum number of vehicles required to
      service all customers in S given the capacity limit Q.
    - x_e = ∑_{k: e ∈ route_k} λ_k is the aggregated flow of edge e in the RMP solution.

    Theoretical Context:
    --------------------
    RCCs are polyhedrally stronger than simple connectivity constraints because
    they incorporate the physical resource constraint (Q). Since VRPP involves
    selecting nodes to visit, the Master Problem uses Set Partitioning, making
    RCCs essential for pruning fractional solutions where multiple routes
    "fractionally" visit nodes in a cluster without collectively satisfying
    the capacity logic.

    Separation Methodology:
    -----------------------
    This engine utilizes the separation framework of Lysgaard et al. (2004),
    supporting both heuristic (Tabu Search) and exact (Max-Flow Min-Cut)
    separation procedures. The separation operates on the support graph
    defined by edges with fractional flow x_e > 0.

    Integration with Pricing:
    --------------------------
    Dual variables associated with RCCs (dual_S) contribute to the reduced
    cost of labels in the RCSPP pricing subproblem. For every edge e in the
    boundary δ(S), the dual value π_S is subtracted from the edge cost.

    References:
    -----------
    Lysgaard et al. (2004), Section 3.2: "Capacity cuts"

    Attributes:
        v_model (VRPPModel): Model for edge indexing and demand tracking.
        sep_engine (SeparationEngine): Internal engine for cut separation.
    """

    def __init__(self, v_model: VRPPModel, sep_engine: SeparationEngine):
        """Initialize the RCC separation engine.

        Args:
            v_model: VRPP model for edge indexing and demand tracking.
            sep_engine: Separation engine from branch-and-cut module.
        """
        self.v_model = v_model
        self.sep_engine = sep_engine

    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:
        """Separate RCCs and add to master problem.

        Algorithm:
        1. Get current edge usage from master problem's LP solution
           (Elementary only: cycles break max-flow conservation).
        2. Map edge usage to the separation engine's representation
        3. Run separation algorithms (heuristic + exact)
        4. Add violated cuts to master with Set Packing relaxation

        Args:
            master: Master problem instance.
            max_cuts: Maximum number of cuts to add.
            kwargs: Unused (for interface consistency).

        Returns:
            int: Number of cuts added.
        """
        # Task 2: Exact separation requires flow conservation on elementary routes.
        # Use filtered edge usage to prevent invalid cut generation from cyclic support.
        edge_vars = master.get_edge_usage(only_elementary=True)
        if not edge_vars:
            return 0

        # Map edge usage to x_vals array for SeparationEngine
        x_vals = np.zeros(len(self.v_model.edges))
        for (i, j), val in edge_vars.items():
            edge_tuple = (min(i, j), max(i, j))
            if edge_tuple in self.v_model.edge_to_idx:
                idx = self.v_model.edge_to_idx[edge_tuple]
                x_vals[idx] = val

        # Build y_vals: fractional node visitation probabilities from the LP solution.
        # y_vals[i-1] = Σ_{k: i ∈ route_k} λ_k for each customer node i.
        # Index 0 corresponds to customer node 1 (depot node 0 is excluded).
        node_visits = master.get_node_visitation()
        n_customers = self.v_model.n_nodes - 1  # exclude depot
        y_vals = np.zeros(n_customers)
        for node, val in node_visits.items():
            if 1 <= node <= n_customers:
                y_vals[node - 1] = min(val, 1.0)  # clamp to [0,1] for LP fractional values

        # Separate cuts using the existing separation engine
        # BPC separation always operates on fractional LP solutions.
        # Use separate_fractional() for exact max-flow based RCC separation.
        node_depth = kwargs.get("node_depth", 0)
        violated_ineqs = self.sep_engine.separate_fractional(
            x_vals, y_vals=y_vals, max_cuts=max_cuts, node_count=node_depth
        )
        added_cuts = 0

        for ineq in violated_ineqs:
            if isinstance(ineq, CapacityCut):
                node_set = list(ineq.node_set)

                # The RCC  sum crossings(S) >= 2*ceil(q(S)/Q)  assumes every node of S is
                # served. In the VRPP optional bins may be skipped, so the cut is valid
                # only when S consists of mandatory bins (it removed the optimum on a
                # 5-node instance where the best plan serves 2 of the 5 bins).
                if not set(node_set) <= set(master.mandatory_nodes):
                    if getattr(master, "_exec_trace", None) is not None:
                        from logic.src.policies.helpers.solvers_and_matheuristics.exact_trace import emit_cut

                        emit_cut(master, "capacity", frozenset(node_set), False)
                    continue
                if master.add_capacity_cut(node_set, ineq.rhs):
                    added_cuts += 1
            elif isinstance(ineq, PCSubtourEliminationCut):
                node_set = list(ineq.node_set)
                # Form 2.1 is always global. Form 2.3 with fractional RHS is local only.
                # getattr used for defensive runtime robustness.
                is_global = ineq.facet_form == "2.1" and not getattr(ineq, "local_only", False)
                if master.add_sec_cut(
                    node_set,
                    ineq.rhs,
                    cut_name=ineq.facet_form,
                    global_cut=is_global,
                    node_i=ineq.node_i,
                    node_j=ineq.node_j,
                ):
                    added_cuts += 1

        return added_cuts

    def get_name(self) -> str:
        """Returns the identifier for this engine.

        Returns:
            str: The engine name 'rcc'.
        """
        return "rcc"


class KnapsackCoverEngine(CuttingPlaneEngine):
    """Separates Cover Inequalities for knapsack constraints in the Master Problem.

    In the VRPP context, this primarily targets the fleet size (vehicle limit)
    constraint (Σ λ_k ≤ K) or any other resource knapsacks. While simpler
    than a general knapsack with varied weights, cover cuts help strengthen
    the integrality of the master problem by excluding fractional solutions
    that exceed the logical resource boundary.

    Mathematical Formulation:
    -------------------------
    Given a knapsack constraint ∑ w_j x_j ≤ b, a set of routes C is a 'cover'
    if ∑_{j ∈ C} w_j > b. The resulting cover inequality is:
        ∑_{j ∈ C} λ_j ≤ |C| - 1

    Theoretical Rationale:
    ----------------------
    A cover inequality excludes the point where all routes in C are selected
    with fractional values that sum to more than the capacity b. For the
    fleet constraint (w_j = 1), a minimal cover is any subset C of size K+1.
    This implementation adapts the logic of Barnhart et al. (2000).

    Interaction with Master Problem:
    --------------------------------
    If a vehicle_limit (K) is explicitly set in the RMP, the fleet constraint
    already prevents the LP from selecting more than K total routes.
    However, cover cuts provide additional polyhedral strength by
    specifically targeting the most fractional routes in the solution.

    Reference: Barnhart et al. (2000), Section 5.

    Attributes:
        v_model (VRPPModel): Model for node and fleet data.
        sep_engine (SeparationEngine): Shared separation utilities.
    """

    def __init__(self, v_model: VRPPModel, sep_engine: SeparationEngine):
        """Initializes the Knapsack Cover separation engine.

        Args:
            v_model: VRPP model for node and fleet data.
            sep_engine: Shared separation utilities.
        """
        self.v_model = v_model
        self.sep_engine = sep_engine

    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:
        """Identify and add violated cover inequalities for the vehicle limit.

        Args:
            master: Master problem instance.
            max_cuts: Maximum number of cuts to add.
            kwargs: Unused (for interface consistency).

        Returns:
            int: Number of cuts added.
        """
        # Task 4 & 6: LCI already handles fleet-size tightening more effectively.
        # This engine serves as a fallback or for non-fleet knapsacks.
        if master.vehicle_limit is None or master.model is None:
            return 0

        # Σ λ_k ≤ K.
        # A cover C is any subset of routes such that |C| > K.
        # The inequality is Σ_{k ∈ C} λ_k ≤ |C| - 1.
        # Violation occurs if Σ_{k ∈ C} λ_k > |C| - 1.

        # Strategy: find routes with largest fractional values.
        # If the sum of the top K+1 λ_k values exceeds K, we have a violation.
        try:
            route_values = {i: var.X for i, var in enumerate(master.lambda_vars) if var.X > 1e-6}
        except (AttributeError, gp.GurobiError):
            return 0
        if len(route_values) <= master.vehicle_limit:
            return 0

        sorted_indices = sorted(route_values.keys(), key=lambda i: route_values[i], reverse=True)
        top_k_plus_1 = sorted_indices[: master.vehicle_limit + 1]
        sum_top = sum(route_values[i] for i in top_k_plus_1)

        if sum_top > master.vehicle_limit + 1e-4:
            # Violated cover found.
            # The cover is the set of nodes visited by the top K+1 routes.
            # Use their union as the node set S for the capacity cut registry.
            cover_node_set: set = set()
            for i in top_k_plus_1:
                cover_node_set.update(master.routes[i].node_coverage)
            cover_node_set.discard(0)  # exclude depot

            if cover_node_set and master.add_capacity_cut(list(cover_node_set), float(master.vehicle_limit)):
                return 1

        return 0

    def get_name(self) -> str:
        """Returns the identifier for this engine.

        Returns:
            str: The engine name 'cover'.
        """
        return "cover"


class RoundedMultistarCutEngine(CuttingPlaneEngine):
    r"""Fractional/Rounded Multistar Inequality separation engine for VRPP.

    Targets the Generalized Multistar Inequalities (Letchford, Eglese, Lysgaard 2002),
    adapted for the VRPP Set Partitioning formulation.

    Mathematical Formulation:
    -------------------------
    For S ⊆ V \ {0}:
        ∑_{e ∈ δ(S)} x_e ≥ (2/Q) ∑_{i ∈ S} d_i y_i
                           + (2/Q) ∑_{j ∉ S∪{0}} d_j · (∑_{i ∈ S} x_{ij})

    Re-written in route variables as ∑_k a_k λ_k ≥ 0, where per route k:
        a_k = crossings(k,S) − (2/Q)·visit_demand(k,S) − (2/Q)·adj_demand(k,S)

    Stored in Gurobi as ∑_k (−a_k) λ_k ≤ 0 with dual γ_S ≥ 0.

    Pricing Dual Integration:
    -------------------------
    When extending label from node u to v, the per-arc contribution to a_k is:

    Case (u∈S, v∉S) — crossing out:
        arc_a_k = 1 − (2/Q)·d_v     (d_v = adj demand; 0 if v is depot)
    Case (v∈S, u∉S) — crossing in:
        arc_a_k = 1 − (2/Q)·d_u − (2/Q)·d_v
        (d_u = adj demand of u if u ≠ depot; d_v = visit demand of v)
    Case (u∈S, v∈S) — interior:
        arc_a_k = −(2/Q)·d_v
    Case (u∉S, v∉S) — exterior:
        arc_a_k = 0

    Reduced cost delta = +arc_a_k · γ_S  (positive because coefficient in LP
    is −a_k; dual γ_S ≥ 0 for ≤ constraint in maximisation).

    This is implemented in RCSPPSolver._extend_label via the
    ``multistar_items`` argument — see solver.py.

    Attributes:
        v_model (VRPPModel): Model for problem data.
        sep_engine (SeparationEngine): Shared separation utilities.
        epsilon (float): Violation threshold.
    """

    def __init__(self, v_model: VRPPModel, sep_engine: SeparationEngine, epsilon: float = 0.01) -> None:
        """Initializes the Multistar separation engine.

        Args:
            v_model: VRPP model for problem data.
            sep_engine: Separation engine for candidate cluster generation.
            epsilon: Minimum violation threshold to add a cut.
        """
        self.v_model = v_model
        self.sep_engine = sep_engine
        self.epsilon = epsilon

    def _add_inequalities(  # noqa: C901
        self,
        master: "VRPPMasterProblem",
        edge_vars: Dict[Tuple[int, int], float],
        node_visits: Dict[int, float],
        candidate_ineqs: List[Any],
        max_cuts: int,
    ) -> int:
        """Adds the multistar inequalities to the master problem.

        Args:
            master: Master problem instance with current fractional LP solution.
            edge_vars: Edge usage variables.
            node_visits: Node visitation variables.
            candidate_ineqs: List of candidate inequalities to add.
            max_cuts: Maximum number of cuts to add.

        Returns:
            int: Number of cuts successfully added.
        """
        added = 0
        already_added: List[Set[int]] = []
        Q = self.v_model.capacity
        for ineq in candidate_ineqs:
            if added >= max_cuts:
                break
            if not isinstance(ineq, CapacityCut):
                continue

            S: Set[int] = set(ineq.node_set)
            if any(prev_S == S for prev_S in already_added):
                continue

            # Compute LP violation of the multistar inequality
            cross_flow = 0.0
            adj_demand = 0.0
            for (u, v), val in edge_vars.items():
                u_in, v_in = u in S, v in S
                if u_in != v_in:
                    cross_flow += val
                    if u_in and v != 0 and not v_in:
                        adj_demand += self.v_model.get_node_demand(v) * val
                    elif v_in and u != 0 and not u_in:
                        adj_demand += self.v_model.get_node_demand(u) * val

            visit_demand = sum(self.v_model.get_node_demand(i) * node_visits.get(i, 0.0) for i in S)
            rhs_val = (2.0 / Q) * visit_demand + (2.0 / Q) * adj_demand

            if rhs_val - cross_flow <= self.epsilon:
                continue  # Not violated

            # Compute per-route structural coefficients.
            # No LP variable value (var.X) is needed here — coefficients depend
            # only on the route's structural relationship to S, not on λ_k.
            coefficients: Dict[int, float] = {}
            for ridx, route in enumerate(master.routes):
                path = [0] + route.nodes + [0]
                k_cross = 0
                k_adj_demand = 0.0
                k_visit_demand = sum(self.v_model.get_node_demand(n) for n in route.nodes if n in S)
                for p in range(len(path) - 1):
                    u, v = path[p], path[p + 1]
                    u_in, v_in = u in S, v in S
                    if u_in != v_in:
                        k_cross += 1
                        if u_in and v != 0 and not v_in:
                            k_adj_demand += self.v_model.get_node_demand(v)
                        elif v_in and u != 0 and not u_in:
                            k_adj_demand += self.v_model.get_node_demand(u)

                # Constraint: Σ a_k λ_k ≥ 0  →  Σ (−a_k) λ_k ≤ 0
                a_k = k_cross - (2.0 / Q) * k_visit_demand - (2.0 / Q) * k_adj_demand
                if abs(a_k) > 1e-6:
                    coefficients[ridx] = -a_k  # negate for ≤ 0 Gurobi constraint

            if not coefficients:
                continue

            # Add to master — prefer dedicated method for correct dual registration
            # (multistar duals are then passed to the RCSPP via multistar_duals key).
            success = False
            if hasattr(master, "add_multistar_cut"):
                success = master.add_multistar_cut(list(S), coefficients)
            else:
                success = master.add_lci_cut(list(S), 0.0, coefficients)

            if success:
                already_added.append(S)
                added += 1

        return added

    def separate_and_add_cuts(self, master: "VRPPMasterProblem", max_cuts: int, **kwargs) -> int:
        """Separate Fractional Multistar Inequalities and add violated ones.

        Args:
            master: Master problem instance with current fractional LP solution.
            max_cuts: Maximum number of cuts to add.
            kwargs: Forwarded kwargs (e.g., node_depth, cut_orthogonality_threshold).

        Returns:
            int: Number of cuts successfully added.
        """
        if master.model is None or not master.lambda_vars:
            return 0

        edge_vars = master.get_edge_usage(only_elementary=True)
        if not edge_vars:
            return 0

        # Build x_vals and y_vals for the separation engine
        x_vals = np.zeros(len(self.v_model.edges))
        for (i, j), val in edge_vars.items():
            edge_tuple = (min(i, j), max(i, j))
            if edge_tuple in self.v_model.edge_to_idx:
                x_vals[self.v_model.edge_to_idx[edge_tuple]] = val

        node_visits = master.get_node_visitation()
        n_customers = self.v_model.n_nodes - 1
        y_vals = np.zeros(n_customers)
        for node, val in node_visits.items():
            if 1 <= node <= n_customers:
                y_vals[node - 1] = min(val, 1.0)

        node_depth = kwargs.get("node_depth", 0)
        candidate_ineqs = self.sep_engine.separate_fractional(
            x_vals, y_vals=y_vals, max_cuts=max_cuts * 3, node_count=node_depth
        )
        return self._add_inequalities(
            master,
            edge_vars,
            node_visits,
            candidate_ineqs=candidate_ineqs,
            max_cuts=max_cuts,
        )

    def get_name(self) -> str:
        """Returns the identifier for this engine.

        Returns:
            str: The engine name 'multistar'.
        """
        return "multistar"
