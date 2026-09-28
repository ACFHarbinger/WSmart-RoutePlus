r"""Master Problem Model for VRPP Branch-and-Price-and-Cut.

Implements the Set Partitioning Problem (SPP) formulation where each column
represents a feasible route. The master problem selects routes to maximize
total collected profit across a finite fleet.

Theoretical Basis: Barnhart et al. (1998).

Attributes:
    VRPPMasterProblem: Main solver class for the master problem.

Example:
    >>> master = VRPPMasterProblem(n_nodes=10, mandatory_nodes={1, 2}, cost_matrix=dist,
    ...                           wastes=w, capacity=100.0, revenue_per_kg=1.0, cost_per_km=0.1)
    >>> master.solve_lp_relaxation()
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from logic.src.policies.helpers.solvers_and_matheuristics.master_problem.constraints import (
    VRPPMasterProblemConstraintsMixin,
)
from logic.src.policies.helpers.solvers_and_matheuristics.master_problem.pool import GlobalCutPool
from logic.src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support import (
    MasterProblemSupport,
    VRPPMasterProblemSupportMixin,
)

if TYPE_CHECKING:
    from logic.src.policies.helpers.solvers_and_matheuristics.common.route import Route

logger = logging.getLogger(__name__)


class VRPPMasterProblem(VRPPMasterProblemConstraintsMixin, VRPPMasterProblemSupportMixin, MasterProblemSupport):
    """
    Master Problem (MP) for the Branch-and-Price-and-Cut algorithm.

    Formulates and solves the Set Partitioning Problem (SPP) relaxation of the VRPP.
    The Master Problem is the core coordinator for the column generation loop,
    managing the interaction between the linear relaxation (Gurobi), the
    cutting plane engines, and the branching constraints.

    Mathematical Formulation:
    -------------------------
    Maximize ∑_{k ∈ Ω} p_k * λ_k
    Subject to:
        ∑_{k ∈ Ω} a_{ik} * λ_k = 1           ∀ i ∈ Mandatory Customers (Set Partitioning)
        ∑_{k ∈ Ω} a_{jk} * λ_k ≤ 1           ∀ j ∈ Optional Customers (Set Packing)
        ∑_{k ∈ Ω} λ_k <= K                   (Vehicle Fleet Constraint / Knapsack)
        ∑_{k ∈ Ω} γ_{Sk} * λ_k ≤ RHS_S       (Cutting Planes: RCC, SRI, Fleet Cover)
        λ_k ∈ {0, 1}

    Theoretical Exactness & Farkas Pricing:
    ---------------------------------------
    To maintain theoretical exactness and support Ryan-Foster branching, this
    implementation utilizes strict Set Partitioning for all mandatory nodes.
    In the event of LP infeasibility (common during the early stages of
    branching), the RMP utilizes a 2-Phase approach. Instead of using
    arbitrary Big-M artificial variables which can cause numerical instability,
    we extract the `FarkasDual` ray from Gurobi. This dual ray points in the
    direction of infeasibility, allowing the pricing subproblem to generate
    columns that specifically resolve the unmet coverage requirements.

    Dual Smoothing & Stabilization:
    -------------------------------
    To mitigate the "tailing-off" effect and dual degeneracy in column
    generation, this MP supports Exponential Dual Smoothing (EDS) as
    proposed by Wentges (1997). This stabilizes the dual values π passed
    to the subproblem, accelerating convergence by preventing the
    dual signal from oscillating between extreme basin points.

    Reference:
        - Barnhart, C., et al. (1998, 2000).
        - Wentges, T. (1997). "Weighted Dantzig-Wolfe decomposition."

    Attributes:
        n_nodes (int): Number of customer nodes.
        mandatory_nodes (Set[int]): Indices of mandatory customers.
        optional_nodes (Set[int]): Indices of optional customers.
        cost_matrix (np.ndarray): Distance/cost matrix.
        wastes (Dict[int, float]): Demand/waste per node.
        capacity (float): Vehicle capacity.
        R (float): Revenue per unit collected.
        C (float): Cost per unit distance.
        vehicle_limit (Optional[int]): Fleet size limit K.
        global_cut_pool (GlobalCutPool): Shared valid inequalities.
        model (Optional[gp.Model]): Underlying Gurobi model.
        routes (List[Route]): Generated columns.
        lambda_vars (List[gp.Var]): Binary/continuous route variables.
        dual_node_coverage (Dict[int, float]): Dual values for node constraints.
        dual_vehicle_limit (float): Dual value for fleet limit.
        phase (int): Optimization phase (1: Feasibility, 2: Optimality).

    Example:
        >>> master = VRPPMasterProblem(10, {1}, dist, w, 100.0, 1.0, 0.1)
    """

    def __init__(
        self,
        n_nodes: int,
        mandatory_nodes: Set[int],
        cost_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        revenue_per_kg: float,
        cost_per_km: float,
        vehicle_limit: Optional[int] = None,
        global_cut_pool: Optional[GlobalCutPool] = None,
    ) -> None:
        """
        Initialise the Master Problem for VRPP.

        Framework (Barnhart, Hane, and Vance 2000):
        This class maintains the Restricted Master Problem (RMP) using the
        Barnhart-style sequencing of CG followed by separation.

        Set Partitioning & Branching:
        To maintain Ryan-Foster (1981) compatibility, we use strict equality (== 1)
        for mandatory nodes. Artificial variables (Big-M) are omitted in favor of
        Farkas dual extraction (Phase I), ensuring that primary infeasibility guides
        the search for feasible basis columns.

        Dual Stabilization:
        Includes support for Exponential Dual Smoothing (Wentges, 1997) to
        stabilize the price signal and mitigate the "heading-in" effect.

        Args:
            n_nodes: Number of customer nodes (excluding depot).
            mandatory_nodes: Set of node indices that must be visited.
            cost_matrix: Distance matrix where [0] is the depot.
            wastes: Waste volume (kg) per node.
            capacity: Vehicle payload capacity (kg).
            revenue_per_kg: Revenue coefficient (R).
            cost_per_km: Operating cost coefficient (C).
            vehicle_limit: Maximum fleet size (K).
            global_cut_pool: Central repository for cross-node valid inequalities.
        """
        self.n_nodes = n_nodes
        self.mandatory_nodes = mandatory_nodes
        self.optional_nodes = set(range(1, n_nodes + 1)) - mandatory_nodes
        self.cost_matrix = cost_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = revenue_per_kg
        self.C = cost_per_km
        self.vehicle_limit = vehicle_limit
        self.global_cut_pool = global_cut_pool or GlobalCutPool()


        self.model: Optional[gp.Model] = None
        self.routes: List[Route] = []
        self.lambda_vars: List[gp.Var] = []

        self.dual_node_coverage: Dict[int, float] = {}
        self.dual_vehicle_limit: float = 0.0
        self.global_column_pool: List[Route] = []
        self.phase: int = 2

        # Active cuts registries
        self.active_src_cuts: Dict[FrozenSet[int], gp.Constr] = {}
        self.active_sec_cuts: Dict[FrozenSet[int], gp.Constr] = {}
        self.active_sec_cuts_local: Dict[FrozenSet[int], gp.Constr] = {}
        self.active_rcc_cuts: Dict[FrozenSet[int], gp.Constr] = {}
        self.active_capacity_cuts: Dict[FrozenSet[int], gp.Constr] = {}
        self.active_lci_cuts: Dict[FrozenSet[int], gp.Constr] = {}
        self.active_lci_node_alphas: Dict[FrozenSet[int], Dict[int, float]] = {}
        self.active_lci_arcs: Dict[FrozenSet[int], Optional[Tuple[int, int]]] = {}
        self.active_sri_cuts: Dict[FrozenSet[int], gp.Constr] = {}
        self.active_edge_clique_cuts: Dict[Tuple[int, int], Tuple[gp.Constr, Dict[int, float]]] = {}
        self.active_multistar_cuts: Dict[FrozenSet[int], gp.Constr] = {}

        # Dual value registries
        self.dual_src_cuts: Dict[FrozenSet[int], float] = {}
        self.dual_sec_cuts: Dict[FrozenSet[int], float] = {}
        self.dual_sec_cuts_local: Dict[FrozenSet[int], float] = {}
        self.dual_rcc_cuts: Dict[FrozenSet[int], float] = {}
        self.dual_capacity_cuts: Dict[FrozenSet[int], float] = {}
        self.dual_lci_cuts: Dict[FrozenSet[int], float] = {}
        self.dual_sri_cuts: Dict[FrozenSet[int], float] = {}
        self.dual_edge_clique_cuts: Dict[Tuple[int, int], float] = {}
        self.dual_multistar_cuts: Dict[FrozenSet[int], float] = {}

        self.farkas_duals: Dict[str, Dict[Any, float]] = {}

        self.column_deletion_enabled: bool = True
        self.strict_set_partitioning: bool = True

    def build_model(self, initial_routes: Optional[List[Route]] = None) -> None:
        """
        Build the Gurobi model for the set-covering master problem.

        Implements the Restricted Master Problem (RMP) formulation:

        1. Mandatory-node coverage constraints use == 1 (Set Partitioning) to
           ensure that Ryan-Foster branching logic remains mathematically valid.
        2. Artificial variables (Big-M) are intentionally omitted.  When the
           column pool cannot cover a mandatory node, Gurobi returns INFEASIBLE
           and we extract the Farkas dual ray to guide Phase I pricing toward
           feasibility (Barnhart et al. 2000, §3).  Big-M variables would
           suppress infeasibility and deadlock the Phase I / Farkas loop.
        3. Optional-node constraints are packing inequalities (<= 1).

        Args:
            initial_routes: If provided, replaces self.routes with this list
                before building the model.
        """
        self.model = gp.Model("VRPP_Master")
        self.model.Params.OutputFlag = 0
        self.model.Params.LogToConsole = 0
        self.model.Params.Method = 1  # Dual Simplex is preferred for SPP/CG re-optimizations
        self.model.Params.InfUnbdInfo = 1
        self.model.Params.DualReductions = 0

        self.lambda_vars = []
        if initial_routes is not None:
            self.routes = initial_routes
        # add_route() de-duplicates against these signatures. Routes buffered before
        # the build may have been replaced above; a stale signature would make the
        # master reject a column pricing keeps returning, and column generation would
        # stop with an understated bound.
        self._route_signatures = {tuple(r.nodes) for r in self.routes}

        for idx, route in enumerate(self.routes):
            var = self.model.addVar(obj=route.profit, vtype=GRB.BINARY, name=f"route_{idx}")
            self.lambda_vars.append(var)

        # Coverage Constraints
        for node in self.mandatory_nodes:
            lhs = gp.LinExpr()
            for idx, route in enumerate(self.routes):
                if node in route.node_coverage:
                    lhs += self.lambda_vars[idx]
            if self.strict_set_partitioning:
                self.model.addConstr(lhs == 1.0, name=f"coverage_{node}")
            else:
                self.model.addConstr(lhs >= 1.0, name=f"coverage_{node}")

        for node in self.optional_nodes:
            lhs = gp.LinExpr()
            for idx, route in enumerate(self.routes):
                if node in route.node_coverage:
                    lhs += self.lambda_vars[idx]
            self.model.addConstr(lhs <= 1, name=f"coverage_{node}")

        # Fleet limit Constraint
        if self.vehicle_limit is not None:
            self.model.addConstr(gp.quicksum(self.lambda_vars) <= self.vehicle_limit, name="vehicle_limit")

        self.model.ModelSense = GRB.MAXIMIZE
        self.model.update()

    def solve_lp_relaxation(self) -> Tuple[float, Dict[int, float]]:
        """
        Solve the LP relaxation of the RMP and extract dual values.

        The integrality constraints on all λ variables are temporarily relaxed
        to continuous [0, 1]. The model is solved using the Dual Simplex method
        (Reference: Barnhart et al. 1998) to ensure efficient warm-starting.

        Dual Extraction & Interpretation:
        - Mandatory Nodes (== 1): π_i is the shadow price.
        - Optional Nodes (<= 1): π_j >= 0.
        - Vehicle Limit (<= K): π_K >= 0 (fleet opportunity cost).
        - Cuts (Σ aλ <= RHS): Dual weights used to penalize "violated" routes in pricing.

        Infeasibility & Farkas:
        If the current column pool cannot satisfy mandatory coverage, Gurobi returns
        GRB.INFEASIBLE. We then extract the Farkas Dual Ray (FarkasDual) which
        mathematically identifies the direction of infeasibility, guiding the
        pricing subproblem to find feasible columns (Phase I).

        Returns:
            Tuple of (LP objective value, {route_index: λ_k value}).
            If infeasible, returns (-inf, {}) and populates self.farkas_duals.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Relax integrality
        for var in self.lambda_vars:
            var.VType = GRB.CONTINUOUS

        self.model.update()
        if self.model.NumVars == 0:
            return 0.0, {}

        self.model.optimize()
        status = self.model.Status

        if status == GRB.INFEASIBLE:
            return self._handle_infeasibility()

        if status != GRB.OPTIMAL:
            # Not a bound: returning 0.0 with no route values was read as a valid LP
            # value and as an (empty) integer solution. Callers treat None as failure.
            logger.warning(f"LP solve returned non-optimal status {status}.")
            return None, {}  # type: ignore[return-value]

        obj_value = self.model.ObjVal
        try:
            route_values = {idx: var.X for idx, var in enumerate(self.lambda_vars)}
        except (gp.GurobiError, AttributeError):
            return 0.0, {}

        self._extract_duals()
        return obj_value, route_values

    def _handle_infeasibility(self) -> Tuple[float, Dict[int, float]]:
        """
        Handle infeasibility by extracting the Farkas dual ray.

        When the RMP is infeasible, the Farkas dual ray provides a certificate
        of infeasibility and guides the pricing subproblem toward finding
        feasible columns (Phase I).

        Returns:
            Tuple of (-inf, {}) and populates self.farkas_duals.
        """
        try:
            # Pricing looks for columns with positive Farkas weight (sum of these values
            # over the column's nodes). Gurobi's FarkasDual has the opposite sign here: a
            # mandatory node with no usable column gets -1, so negate the ray once.
            farkas_node_duals: Dict[Union[int, str], float] = {}
            for node in range(1, self.n_nodes + 1):
                constr = self.model.getConstrByName(f"coverage_{node}")
                if constr is not None:
                    farkas_node_duals[node] = -constr.FarkasDual

            if self.vehicle_limit is not None:
                constr = self.model.getConstrByName("vehicle_limit")
                if constr is not None:
                    farkas_node_duals["vehicle_limit"] = -constr.FarkasDual

            self.farkas_duals = {
                "node_duals": farkas_node_duals,
                "rcc_duals": {s: -c.FarkasDual for s, c in self.active_capacity_cuts.items()},
                "sri_duals": {s: -c.FarkasDual for s, c in self.active_sri_cuts.items()},
                "edge_clique_duals": {e: -c[0].FarkasDual for e, c in self.active_edge_clique_cuts.items()},
            }
            return -float("inf"), {}
        except AttributeError:
            logger.warning("FarkasDual extraction failed. Ensure InfUnbdInfo=1 is set.")
            return -float("inf"), {}

    def _extract_duals(self) -> None:
        """Extract dual values for all constraints in an optimal LP solution.

        Dual Extraction & Interpretation:
        - Mandatory Nodes (== 1): π_i is the shadow price.
        - Optional Nodes (<= 1): π_j >= 0.
        - Vehicle Limit (<= K): π_K >= 0 (fleet opportunity cost).
        - Cuts (Σ aλ <= RHS): Dual weights used to penalize "violated" routes in pricing.
        """
        self.dual_node_coverage = {}
        for node in range(1, self.n_nodes + 1):
            constr = self.model.getConstrByName(f"coverage_{node}")
            if constr is not None:
                with contextlib.suppress(gp.GurobiError):
                    # Set Partitioning (==) allows unrestricted duals. Set Packing (<=) restricts to >= 0.
                    if node in self.mandatory_nodes and self.strict_set_partitioning:
                        self.dual_node_coverage[node] = constr.Pi
                    else:
                        self.dual_node_coverage[node] = max(0.0, constr.Pi)

        self.dual_vehicle_limit = 0.0
        if self.vehicle_limit is not None:
            constr = self.model.getConstrByName("vehicle_limit")
            if constr is not None:
                with contextlib.suppress(gp.GurobiError):
                    self.dual_vehicle_limit = max(0.0, constr.Pi)

        # Extract active cut duals
        # RCC: >= constraint in MAX LP → Pi ≤ 0, penalty magnitude = -Pi
        self.dual_capacity_cuts = {s: max(0.0, -c.Pi) for s, c in self.active_capacity_cuts.items()}
        # SRI: <= constraint in MAX LP → Pi ≥ 0 already
        self.dual_sri_cuts = {s: max(0.0, c.Pi) for s, c in self.active_sri_cuts.items()}
        self.dual_lci_cuts = {s: max(0.0, c.Pi) for s, c in self.active_lci_cuts.items()}
        self.dual_edge_clique_cuts = {e: max(0.0, c[0].Pi) for e, c in self.active_edge_clique_cuts.items()}
        # Multistar: ≤ 0 constraint in a MAX LP → dual Pi ≥ 0.
        self.dual_multistar_cuts = {s: max(0.0, c.Pi) for s, c in self.active_multistar_cuts.items()}

    def solve_ip(self) -> Tuple[float, List[Route]]:
        """Solve the integer programme at the current B&B node.

        Re-imposes binary integrality on all λ variables and calls Gurobi's
        MIP solver.

        Returns:
            Tuple containing:
                - IP objective value.
                - List of routes selected in the optimal solution (λ_k = 1).

        Raises:
            ValueError: If build_model() has not been called.
            RuntimeError: If the MIP solve fails.
        """

        if self.model is None:
            raise ValueError("Model not built.")
        if self.model.NumVars == 0:
            return 0.0, []

        # Enforce Integrality
        for var in self.lambda_vars:
            var.VType = GRB.BINARY
        self.model.update()

        try:
            self.model.optimize()
            if self.model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
                raise RuntimeError(f"IP solve failed: {self.model.Status}")
            return self.model.ObjVal, [self.routes[i] for i, v in enumerate(self.lambda_vars) if v.X > 0.5]
        finally:
            # Revert to continuous for further CG iterations
            for var in self.lambda_vars:
                var.VType = GRB.CONTINUOUS
            self.model.update()

    def add_route(self, route: Route) -> bool:
        """Add a route (column) to the master problem.

        If the Gurobi model has already been built, the column is inserted
        into the live model immediately. Otherwise it is buffered in
        self.routes and will be included when build_model() is called.

        Args:
            route: Route to insert as a new λ variable.

        Returns:
            bool: True if the route was successfully added, False if it was a duplicate.
        """
        if not hasattr(self, "_route_signatures"):
            self._route_signatures = {tuple(r.nodes) for r in self.routes}

        sig = tuple(route.nodes)
        if sig in self._route_signatures:
            return False

        self.routes.append(route)
        self._route_signatures.add(sig)
        if self.model is not None:
            self._add_column_to_model(route)

        return True

    def _add_column_to_model(self, route: Route) -> None:
        """Dynamically inserts a new column (route) into the Restricted Master Problem.

        This must inject the variable into both the base constraints AND any active cuts.

        Args:
            route (Route): Route whose λ variable is being added.
        """
        if self.model is None:
            return

        var = self.model.addVar(obj=route.profit, vtype=GRB.BINARY, name=f"route_{len(self.lambda_vars)}")

        # Base Coverage Constraints
        for node in route.node_coverage:
            constr = self.model.getConstrByName(f"coverage_{node}")
            if constr is not None:
                self.model.chgCoeff(constr, var, 1.0)

        # Base Vehicle Limit
        if self.vehicle_limit is not None:
            constr = self.model.getConstrByName("vehicle_limit")
            if constr is not None:
                self.model.chgCoeff(constr, var, 1.0)

        self.lambda_vars.append(var)

        # Cut Plane Injection
        self._wire_route_into_active_cuts(route, var)
        self.model.update()

    def _wire_route_into_active_cuts(self, route: Route, var: gp.Var) -> None:  # noqa: C901
        """Calculate the exact coefficient of the new route for every active cut.

        Guarantees that newly priced columns strictly respect previously separated inequalities.
        This ensures that columns added after some cuts have been discovered correctly
        participate in those cuts, maintaining LP relaxation validity.

        Args:
            route: The new route to inject.
            var: The Gurobi variable associated with the route.
        """

        if self.model is None:
            return

        full_path = [0] + route.nodes + [0]

        # 1. Capacity Cuts (RCCs) & SECs
        # The coefficient is the number of times the route crosses the cut boundary δ(S)
        for cut_dict in [self.active_capacity_cuts, self.active_sec_cuts, self.active_sec_cuts_local]:
            for node_set, constr in cut_dict.items():
                crossings = sum(
                    1 for i in range(len(full_path) - 1) if (full_path[i] in node_set) != (full_path[i + 1] in node_set)
                )
                if crossings > 0:
                    self.model.chgCoeff(constr, var, float(crossings))

        # 2. Subset Row Inequalities (SRIs)
        # The coefficient is ⌊|route ∩ S| / 2⌋
        for subset, constr in self.active_sri_cuts.items():
            intersection_size = sum(1 for n in subset if n in route.node_coverage)
            coeff = intersection_size // 2
            if coeff > 0:
                self.model.chgCoeff(constr, var, float(coeff))

        # 3. Localized Capacity Inequalities (LCIs)
        for cover_set, constr in self.active_lci_cuts.items():
            arc = self.active_lci_arcs.get(cover_set)
            if arc is not None:
                # Arc-saturation LCI: coefficient is 1 if route traverses arc
                contains_arc = any(
                    full_path[i] == arc[0] and full_path[i + 1] == arc[1] for i in range(len(full_path) - 1)
                )
                if contains_arc:
                    self.model.chgCoeff(constr, var, 1.0)
            else:
                # Node-based LCI: sum over node alphas
                node_alphas_for_set = self.active_lci_node_alphas.get(cover_set, {})
                alpha_k = sum(
                    node_alphas_for_set.get(n, 1.0 if n in cover_set else 0.0) for n in route.node_coverage if n != 0
                )
                if alpha_k > 1e-6:
                    self.model.chgCoeff(constr, var, alpha_k)

        # 4. Edge Clique Cuts
        for edge_tuple, (constr, _) in self.active_edge_clique_cuts.items():
            contains_edge = any(
                tuple(sorted((full_path[i], full_path[i + 1]))) == edge_tuple for i in range(len(full_path) - 1)
            )
            if contains_edge:
                self.model.chgCoeff(constr, var, 1.0)

        # 5. Multistar Cuts (Letchford, Eglese, Lysgaard 2002)
        # Coefficient for route k in cut Σ (-a_k) λ_k ≤ 0:
        #   a_k = crossings(k,S) - (2/Q)*visit_demand(k,S) - (2/Q)*adj_demand(k,S)
        for node_set, constr in self.active_multistar_cuts.items():
            S = node_set
            Q = self.capacity
            k_cross = 0
            k_adj = 0.0
            k_visit = sum(self.wastes.get(n, 0.0) for n in route.nodes if n in S)
            for p in range(len(full_path) - 1):
                u, v = full_path[p], full_path[p + 1]
                u_in, v_in = u in S, v in S
                if u_in != v_in:
                    k_cross += 1
                    if u_in and v != 0 and not v_in:
                        k_adj += self.wastes.get(v, 0.0)
                    elif v_in and u != 0 and not u_in:
                        k_adj += self.wastes.get(u, 0.0)
            a_k = k_cross - (2.0 / Q) * k_visit - (2.0 / Q) * k_adj
            neg_a_k = -a_k  # stored coefficient in the ≤ 0 constraint
            if abs(neg_a_k) > 1e-6:
                self.model.chgCoeff(constr, var, neg_a_k)

    def purge_useless_columns(self, tolerance: float = -0.1) -> int:
        """Remove non-basic columns with significantly negative reduced cost.

        In a maximization LP, non-basic columns have RC <= 0. If a column is
        not in the basis and its reduced cost is highly negative, it is
        unlikely to enter the basis soon. Removing such columns prevents
        basis matrix bloat and keeps the RMP solver fast.

        Args:
            tolerance: Reduced cost threshold. Columns with RC < tolerance
                that are non-basic will be removed.

        Returns:
            Number of columns purged from the Gurobi model.
        """

        if not self.column_deletion_enabled or self.model is None or self.model.Status != GRB.OPTIMAL:
            return 0

        to_remove = []
        try:
            for i, v in enumerate(self.lambda_vars):
                if v.VBasis != 0 and tolerance > v.RC:
                    to_remove.append(i)
        except Exception:
            return 0

        if not to_remove:
            return 0

        for i in sorted(to_remove, reverse=True):
            route_to_remove = self.routes[i]
            self.global_column_pool.append(route_to_remove)
            self.model.remove(self.lambda_vars[i])
            self.lambda_vars.pop(i)
            self.routes.pop(i)
            if hasattr(self, "_route_signatures"):
                sig = tuple(route_to_remove.nodes)
                if sig in self._route_signatures:
                    self._route_signatures.remove(sig)

        self.model.update()
        return len(to_remove)

    def get_reduced_cost_coefficients(self) -> Dict[str, Any]:
        """Return the dual-value coefficients used by the pricing subproblem.

        Sign Convention & Maximization:
            The VRPP is formulated as a maximization problem. Under Gurobi's
            standard dual representation, the reduced cost calculation for
            pricing (searching for columns to boost the objective) is:
            rc(r) = profit(r) - Σ (a_{ik} * π_i), where π_i are the duals.
            - For >= constraints (e.g. RCC), Pi is non-positive; we use -Pi.
            - For <= constraints (e.g. SRI, Vehicle Limit), Pi is non-negative; we use +Pi.

        Returns:
            Dictionary mapping node ID (int) -> dual value, plus keys for cut duals.
        """
        if self.model is None or self.model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            return {}

        # node_duals: include vehicle_limit dual under its own key
        node_duals = dict(self.dual_node_coverage)
        if self.dual_vehicle_limit > 1e-9:
            node_duals["vehicle_limit"] = self.dual_vehicle_limit

        # rcc_duals: dict of (set: dual) consumed by RCSPPSolver.solve()
        rcc_duals = {s: d for s, d in self.dual_capacity_cuts.items() if d > 1e-8}

        # sri_duals: dict of (set: dual)
        sri_duals = {s: d for s, d in self.dual_sri_cuts.items() if d > 1e-8}

        # lci: pass both the dual values AND the node-alpha maps AND the source arcs
        lci_duals = {s: d for s, d in self.dual_lci_cuts.items() if d > 1e-8}

        return {
            "node_duals": node_duals,
            "rcc_duals": rcc_duals,
            "sri_duals": sri_duals,
            "edge_clique_duals": dict(self.dual_edge_clique_cuts),
            "lci_duals": lci_duals,
            "lci_node_alphas": dict(self.active_lci_node_alphas),
            "lci_arcs": dict(self.active_lci_arcs),
            "multistar_duals": dict(self.dual_multistar_cuts),
        }

    def get_node_visitation(self) -> Dict[int, float]:
        """Aggregate fractional node-visitation values from the current LP solution.

        For each node i, returns Σ_{k: i ∈ route_k} λ_k, i.e. the total
        fractional "coverage" of that node across all selected routes.

        Returns:
            Mapping from node ID to aggregated λ coverage.
        """

        if self.model is None:
            return {}
        try:
            visits: Dict[int, float] = {}
            for i, v in enumerate(self.lambda_vars):
                if v.X > 1e-6:
                    for n in self.routes[i].node_coverage:
                        visits[n] = visits.get(n, 0.0) + v.X
            return visits
        except Exception:
            return {}

    def get_edge_usage(self, only_elementary: bool = False) -> Dict[Tuple[int, int], float]:
        """Aggregate fractional edge visitation values from current LP solution.

        For an edge (i, j), returns Σ_{k: (i,j) ∈ route_k} λ_k.
        Edges are returned as canonical sorted tuples (min, max).

        Args:
            only_elementary: If True, only elementary routes are considered.

        Returns:
            Mapping from (u, v) -> fractional visitation sum.
        """

        if self.model is None:
            return {}

        edge_usage: Dict[Tuple[int, int], float] = {}
        try:
            for idx, var in enumerate(self.lambda_vars):
                val = var.X
                if val > 1e-6:
                    route = self.routes[idx]

                    # Task 2: Filter cyclic support for exact separation
                    if only_elementary and len(set(route.nodes)) != len(route.nodes):
                        continue

                    nodes = [0] + route.nodes + [0]
                    for i in range(len(nodes) - 1):
                        u, v = nodes[i], nodes[i + 1]
                        edge = (min(u, v), max(u, v))
                        edge_usage[edge] = edge_usage.get(edge, 0.0) + val
            return edge_usage
        except Exception:
            return {}

    def save_basis(self) -> Optional[Tuple[List[int], List[int]]]:
        """Save the basis (VBasis, CBasis) for variables and constraints.

        Returns:
            Tuple of (variable_basis, constraint_basis) lists, or None if failed.
        """

        if self.model is None or self.model.Status != GRB.OPTIMAL:
            return None
        try:
            vbasis = [v.VBasis for v in self.model.getVars()]
            cbasis = [c.CBasis for c in self.model.getConstrs()]
            return (vbasis, cbasis)
        except Exception:
            return None

    def restore_basis(self, vbasis: List[int], cbasis: List[int]) -> None:
        """Restore the basis to warm-start the simplex algorithm.

        This robust version handles partial restorations in case of
        column-pool size mismatches across B&B nodes.

        Args:
            vbasis: Variable basis list.
            cbasis: Constraint basis list.
        """

        if self.model is None or vbasis is None or cbasis is None:
            return

        # Partial restore robust to column-pool mismatch
        vars_ = self.model.getVars()
        constrs_ = self.model.getConstrs()

        n_vars_saved = len(vbasis)
        n_constrs_saved = len(cbasis)
        n_vars_now = len(vars_)
        n_constrs_now = len(constrs_)

        if n_vars_now != n_vars_saved or n_constrs_now != n_constrs_saved:
            logger.debug(
                f"restore_basis: size mismatch — saved ({n_vars_saved} vars, "
                f"{n_constrs_saved} constrs) vs current ({n_vars_now} vars, "
                f"{n_constrs_now} constrs). Partial restore; warm-start may be degraded."
            )

        n_vars = min(n_vars_saved, n_vars_now)
        n_constrs = min(n_constrs_saved, n_constrs_now)

        for i in range(n_vars):
            vars_[i].VBasis = vbasis[i]

        for i in range(n_constrs):
            constrs_[i].CBasis = cbasis[i]

        self.model.update()
