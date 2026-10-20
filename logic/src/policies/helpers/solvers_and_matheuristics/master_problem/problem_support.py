r"""
Master problem support mixin for Vehicle Routing Problem with Profit Maximization (VRPP).

This module provides helper methods for the VRPP master problem, including:
- Constraint addition and management (coverage, capacity, SRI, SEC, RCC, LCI).
- Route pool management (adding, removing, merging routes).
- Phase control (Phase 1: feasibility, Phase 2: optimality).
- Reduced cost calculation with support for all cut types.
- Artificial variable detection (Phase 1 infeasibility).

Attributes:
    MasterProblemSupport: Structural protocol for master problem implementations.
    VRPPMasterProblemSupportMixin: Helper methods for column pooling and sifting.

Example:
    >>> support = MasterProblemSupport()
    >>> rc = support.calculate_reduced_cost(route, duals)

Parts are relocated to column_pool_support.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Protocol, Set, Tuple

import gurobipy as gp
import numpy as np

if TYPE_CHECKING:
    from logic.src.policies.helpers.solvers_and_matheuristics.branching.constraints import AnyBranchingConstraint
    from logic.src.policies.helpers.solvers_and_matheuristics.common.route import Route
    from logic.src.policies.helpers.solvers_and_matheuristics.master_problem.pool import GlobalCutPool

logger = logging.getLogger(__name__)


class MasterProblemSupport(Protocol):
    """
    Structural interface defining the requirements for a VRPP Master Problem.

    Attributes:
        n_nodes: Number of nodes in the problem.
        mandatory_nodes: Set of mandatory nodes.
        optional_nodes: Set of optional nodes.
        cost_matrix: Matrix of travel costs between nodes.
        wastes: Dictionary of wastes for each node.
        capacity: Capacity of each vehicle.
        R: Maximum distance each vehicle can travel.
        C: Cost of using each vehicle.
        vehicle_limit: Maximum number of vehicles that can be used.
        global_cut_pool: Pool of active cuts.
        BIG_M: Large constant for big-M constraints.
        model: Gurobi model for the master problem.
        routes: List of routes.
        lambda_vars: List of lambda variables corresponding to the routes.
        dual_node_coverage: Dictionary of dual values for node coverage constraints.
        dual_vehicle_limit: Dual value for vehicle limit constraint.
        global_column_pool: Pool of candidate routes.
        phase: Phase of the algorithm (1 for feasibility, 2 for optimality).
        active_src_cuts: Dictionary of active subset-row inequality (SRC) cuts.
        active_sec_cuts: Dictionary of active subset-edge clique (SEC) cuts.
        active_sec_cuts_local: Dictionary of local SEC cuts.
        active_rcc_cuts: Dictionary of active restricted cut cover (RCC) cuts.
        active_capacity_cuts: Dictionary of active capacity cuts.
        active_lci_cuts: Dictionary of active lifted cover inequality (LCI) cuts.
        active_lci_node_alphas: Dictionary of node alphas for LCI cuts.
        active_lci_arcs: Dictionary of arcs for LCI cuts.
    """

    n_nodes: int
    mandatory_nodes: Set[int]
    optional_nodes: Set[int]
    cost_matrix: np.ndarray[Any, Any]
    wastes: Dict[int, float]
    capacity: float
    R: float
    C: float
    vehicle_limit: Optional[int]
    global_cut_pool: "GlobalCutPool"
    BIG_M: float

    model: Optional[gp.Model]
    routes: List[Route]
    lambda_vars: List[gp.Var]
    dual_node_coverage: Dict[int, float]
    dual_vehicle_limit: float
    global_column_pool: List[Route]
    phase: int

    # Active cuts registries
    active_src_cuts: Dict[FrozenSet[int], gp.Constr]
    active_sec_cuts: Dict[FrozenSet[int], gp.Constr]
    active_sec_cuts_local: Dict[FrozenSet[int], gp.Constr]
    active_rcc_cuts: Dict[FrozenSet[int], gp.Constr]
    active_capacity_cuts: Dict[FrozenSet[int], gp.Constr]
    active_lci_cuts: Dict[FrozenSet[int], gp.Constr]
    active_lci_node_alphas: Dict[FrozenSet[int], Dict[int, float]]
    active_lci_arcs: Dict[FrozenSet[int], Optional[Tuple[int, int]]]
    active_sri_cuts: Dict[FrozenSet[int], gp.Constr]
    active_edge_clique_cuts: Dict[Tuple[int, int], Tuple[gp.Constr, Dict[int, float]]]
    # Multistar cuts: maps frozenset(S) → Gurobi constraint (Σ −a_k λ_k ≤ 0)
    active_multistar_cuts: Dict[FrozenSet[int], gp.Constr]

    # Dual value registries
    dual_src_cuts: Dict[FrozenSet[int], float]
    dual_sec_cuts: Dict[FrozenSet[int], float]
    dual_sec_cuts_local: Dict[FrozenSet[int], float]
    dual_rcc_cuts: Dict[FrozenSet[int], float]
    dual_capacity_cuts: Dict[FrozenSet[int], float]
    dual_lci_cuts: Dict[FrozenSet[int], float]
    dual_sri_cuts: Dict[FrozenSet[int], float]
    dual_edge_clique_cuts: Dict[Tuple[int, int], float]
    # Multistar duals: γ_S ≥ 0, emitted in get_reduced_cost_coefficients
    # under key "multistar_duals" so the RCSPP pricer can apply arc-level penalties.
    dual_multistar_cuts: Dict[FrozenSet[int], float]

    # Dual stabilization configurations
    prev_dual_node_coverage: Dict[int, float]
    prev_dual_vehicle_limit: float
    prev_dual_capacity_cuts: Dict[FrozenSet[int], float]
    prev_dual_sri_cuts: Dict[FrozenSet[int], float]
    farkas_duals: Dict[str, Dict[Any, float]]

    column_deletion_enabled: bool
    strict_set_partitioning: bool

    # Core Methods
    def build_model(self, initial_routes: Optional[List[Route]] = None) -> None:
        """Constructs the initial Gurobi model for the RMP.

        Args:
            initial_routes: Optional list of starting routes.
        """
        ...

    def solve_lp_relaxation(self) -> Tuple[float, Dict[int, float]]:
        """Solves the LP relaxation of the RMP and returns the duals.

        Returns:
            Tuple[float, Dict[int, float]]: (Objective value, dual variables dictionary).
        """
        ...

    def _handle_infeasibility(self) -> Tuple[float, Dict[int, float]]:
        """Handles RMP infeasibility using Farkas duals or artificial variables.

        Returns:
            Tuple[float, Dict[int, float]]: (Infeasibility cost, Farkas duals dictionary).
        """
        ...

    def _extract_duals(self) -> None:
        """Extracts dual values from the solved LP into internal registries."""
        ...

    def solve_ip(self, time_limit: Optional[float] = None) -> Tuple[float, List[Route]]:
        """Solves the RMP as an Integer Program.

        Args:
            time_limit: Optional time limit in seconds for the IP solve.

        Returns:
            Tuple[float, List[Route]]: (IP objective value, list of selected routes).
        """
        ...

    def add_route(self, route: Route) -> None:
        """Adds a new route (column) to the RMP.

        Args:
            route: The route to add.
        """
        ...

    def _add_column_to_model(self, route: Route) -> None:
        """Internal helper to add a route variable to the Gurobi model.

        Args:
            route: The route to add.
        """
        ...

    def _wire_route_into_active_cuts(self, route: Route, var: Any) -> None:
        """Connects a route variable to all relevant active cut constraints.

        Args:
            route: The route representation.
            var: The Gurobi variable for the route.
        """
        ...

    def purge_useless_columns(self, tolerance: float = -0.1) -> int:
        """Removes columns with high reduced costs to keep the RMP compact.

        Args:
            tolerance: Reduced cost threshold for deletion.

        Returns:
            int: Number of columns purged.
        """
        ...

    def get_reduced_cost_coefficients(self) -> Dict[str, Any]:
        """Calculates coefficients for pricing subproblems.

        Returns:
            Dict[str, Any]: Mapping of dual components (nodes, cuts) to their values.
        """
        ...

    def get_node_visitation(self) -> Dict[int, float]:
        """Calculates node visitation probabilities (y_i) from fractional route values.

        Returns:
            Dict[int, float]: Mapping of node index to visitation probability.
        """
        ...

    def get_edge_usage(self, only_elementary: bool = False) -> Dict[Tuple[int, int], float]:
        """Calculates edge usage frequencies (x_ij) from fractional route values.

        Args:
            only_elementary: Whether to only consider elementary routes.

        Returns:
            Dict[Tuple[int, int], float]: Mapping of (u, v) edge to usage value.
        """
        ...

    def save_basis(self) -> Optional[Tuple[List[int], List[int]]]:
        """Saves the current basis status of RMP variables and constraints.

        Returns:
            Optional[Tuple[List[int], List[int]]]: (vbasis, cbasis) status arrays,
                                                or None if no model exists.
        """
        ...

    def restore_basis(self, vbasis: List[int], cbasis: List[int]) -> None:
        """Restores the RMP basis from saved status arrays.

        Args:
            vbasis: Variable basis statuses.
            cbasis: Constraint basis statuses.
        """
        ...

    # Problem-support methods
    def sift_global_column_pool(
        self: MasterProblemSupport,
        node_duals: Dict[int, float],
        rcc_duals: Dict[FrozenSet[int], float],
        sri_duals: Dict[FrozenSet[int], float],
        edge_clique_duals: Dict[Tuple[int, int], float],
        lci_duals: Optional[Dict[FrozenSet[int], float]] = None,
        lci_node_alphas: Optional[Dict[FrozenSet[int], Dict[int, float]]] = None,
        branching_constraints: Optional[List["AnyBranchingConstraint"]] = None,
        rc_tolerance: float = 1e-5,
    ) -> int:
        """Scan the global pool for profitable columns under current duals and branching.

        Args:
            node_duals: Base duals from coverage and vehicle limits.
            rcc_duals: Current Root-Capacity Cut duals.
            sri_duals: Current Subset-Row Inequality duals.
            edge_clique_duals: Current Edge Clique cut duals.
            lci_duals: Current Lifted Cover Inequality duals.
            lci_node_alphas: Node lifting coefficients.
            branching_constraints: Active branching constraints.
            rc_tolerance: Threshold to prevent injection of insignificant columns.

        Returns:
            int: Number of routes re-activated and added to the RMP.
        """
        ...

    def calculate_reduced_cost(self: MasterProblemSupport, route: Route, dual_values: Dict[str, Any]) -> float:
        """Calculates the reduced cost of a route using current dual values.

        Args:
            route: The route to evaluate.
            dual_values: Dictionary of dual components.

        Returns:
            float: The calculated reduced cost.
        """
        ...

    def set_phase(self: MasterProblemSupport, phase: int) -> None:
        """Switch between Phase 1 (Feasibility) and Phase 2 (Optimality).

        Args:
            phase: 1 for Feasibility, 2 for Optimality.
        """
        ...

    def deduplicate_column_pool(self: MasterProblemSupport, tol: float = 1e-6) -> int:
        """Prune mathematically equivalent routes from the global column pool.

        Args:
            tol: Numerical tolerance for profit comparison.

        Returns:
            int: Number of redundant routes removed.
        """
        ...

    def add_set_packing_capacity_cut(self: MasterProblemSupport, node_list: List[int], rhs: float) -> bool:
        """Adds a capacity cut specifically for set packing formulations.

        Args:
            node_list: Nodes in the cut set.
            rhs: Right-hand side.

        Returns:
            bool: True if the cut was successfully added.
        """
        ...

    def _count_crossings(self: MasterProblemSupport, route: Route, node_set: FrozenSet[int]) -> int:
        """Counts how many times a route enters/exits a given node set.

        Args:
            route: The route to analyze.
            node_set: The set of nodes to check crossings for.

        Returns:
            Number of crossings (entries/exits).
        """
        ...
