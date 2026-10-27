"""Branch-and-Price Solver for VRPP.

Implements the complete Branch-and-Price algorithm with:
- Column generation at each node
- Pluggable branching strategy (edge branching or Ryan-Foster)
- LP relaxation solving with pricing
- Integer solution recovery

Based on Sections 5 and 6 of Barnhart et al. (1998).

References:
    Barnhart, C., Johnson, E. L., Nemhauser, G. L., Savelsbergh, M. W. P., & Vance, P. H. (1998).
    "Branch-and-price: Column Generation for Solving Huge Integer Programs".

Attributes:
    BranchAndPriceSolver (class): The core optimization engine.

Example:
    >>> solver = BranchAndPriceSolver(n_nodes=50, cost_matrix=dist, ...)
    >>> tour, profit, stats = solver.solve()

Parts are relocated to bp_search, bp_routes.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from logic.src.configs.policies.routing.exact_and_decomposition_solvers.bp import BPConfig
from logic.src.policies.helpers.solvers_and_matheuristics import (
    BranchAndBoundTree,
    BranchNode,
    RCSPPSolver,
    Route,
    VRPPMasterProblem,
)

from .bp_routes import BranchAndPriceRoutesMixin
from .bp_search import BranchAndPriceSearchMixin
from .pricing_subproblem import PricingSubproblem


class BranchAndPriceSolver(BranchAndPriceRoutesMixin, BranchAndPriceSearchMixin):
    """Branch-and-Price Algorithm for VRPP with Column Generation.

    Algorithm:
        1. Initialise with greedy routes.
        2. Solve LP relaxation with column generation.
        3. If LP solution is integer, terminate.
        4. Otherwise branch using the configured strategy and recurse.

    Branching strategies:
        - "edge" (default): Branches on the most-fractional directed arc (x_{uv}).
        - "ryan_foster": Selection of a node-pair (r, s) to branch on.

    Attributes:
        params (BPConfig): Standardized configuration parameters.
        n_nodes (int): Number of customer nodes.
        cost_matrix (np.ndarray): Symmetric distance matrix.
        wastes (Dict[int, float]): Map from node to waste volume.
        capacity (float): Vehicle capacity.
        R (float): Revenue per kg.
        C (float): Cost per km.
        mandatory_nodes (Set[int]): Indices of nodes that must be visited.
        depot (int): Index of the depot node (always 0).
        proven_optimal (bool): Whether the final solution is proven optimal.
        num_iterations (int): Counter for pricing rounds.
        num_columns_generated (int): Total columns added to the pool.
        lp_bound (float): Best LP bound found at the root.
        ip_solution (float): Objective value of the best integer solution.
        tree (Optional[BranchAndBoundTree]): The search tree object.
    """

    def __init__(
        self,
        n_nodes: int,
        cost_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        revenue_per_kg: float,
        cost_per_km: float,
        mandatory_nodes: Optional[Set[int]] = None,
        params: Optional[BPConfig] = None,
    ) -> None:
        """Initialise the Branch-and-Price solver.

        Args:
            n_nodes: Number of customer nodes (excluding depot, depot = 0).
            cost_matrix: Distance matrix of shape (n_nodes+1, n_nodes+1).
            wastes: Mapping from node ID to waste volume.
            capacity: Vehicle payload capacity.
            revenue_per_kg: Revenue per unit of waste collected.
            cost_per_km: Operating cost per unit of distance travelled.
            mandatory_nodes: Node indices that must be visited.
            params: BP parameters (default: ``BPConfig()``).
        """
        params = params if params is not None else BPConfig()
        self.params = params
        self.n_nodes = n_nodes
        self.cost_matrix = cost_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = revenue_per_kg
        self.C = cost_per_km
        self.mandatory_nodes: Set[int] = mandatory_nodes or set()
        self.depot = 0

        # Tracking exactness
        self.proven_optimal: bool = True

        # Precompute ng-neighborhoods once for reuse across all B&B nodes.
        self._ng_neighborhoods: Optional[Dict[int, Set[int]]] = None
        if self.params.use_exact_pricing:
            from logic.src.policies.helpers.solvers_and_matheuristics import RCSPPSolver as _R

            _tmp = _R(
                n_nodes=self.n_nodes,
                cost_matrix=self.cost_matrix,
                wastes=self.wastes,
                capacity=self.capacity,
                revenue_per_kg=self.R,
                cost_per_km=self.C,
                use_ng_routes=self.params.use_ng_routes,
                ng_neighborhood_size=self.params.ng_neighborhood_size,
            )
            self._ng_neighborhoods = _tmp.ng_neighborhoods

        branching_strategy = self.params.branching_strategy
        if branching_strategy == "edge" and (self.params.use_ryan_foster_branching or self.params.use_ryan_foster):
            branching_strategy = "ryan_foster"

        if branching_strategy == "divergence" and not self.params.multiple_waste_types:
            logging.info(
                "Single-commodity VRP detected. Promoting 'divergence' branching to 'multi_edge_partition' strategy."
            )
            branching_strategy = "multi_edge_partition"

        self.params.branching_strategy = branching_strategy
        self.params.use_ryan_foster = branching_strategy == "ryan_foster"

        if self.params.use_ryan_foster and not self.params.allow_heuristic_ryan_foster:
            raise ValueError(
                "Ryan-Foster branching is theoretically inexact when used with a Set Covering "
                "master problem (the default). It can prune optimal over-covering solutions. "
                "To acknowledge this and proceed, pass 'allow_heuristic_ryan_foster=True' to the solver, "
                "or use the mathematically exact 'edge' branching strategy."
            )

        if self.params.use_ryan_foster:
            warnings.warn(
                "Ryan-Foster branching is theoretically a heuristic when used with a Set Covering "
                "master problem (>= 1). It may lead to unbalanced trees or over-coverage issues. "
                "Edge branching is the mathematically exact default.",
                UserWarning,
                stacklevel=2,
            )

        # Statistics
        self.num_iterations: int = 0
        self.num_columns_generated: int = 0
        self.lp_bound: float = 0.0
        self.ip_solution: float = 0.0
        self.tree: Optional[BranchAndBoundTree] = None

    # ------------------------------------------------------------------
    # Top-level solve
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[int], float, Dict[str, Any]]:
        """Solves the VRPP using Branch-and-Price.

        Returns:
            A 3-tuple containing:
                - best_tour: Optimized sequence of nodes.
                - best_profit: Final objective value.
                - statistics: Dictionary of execution metrics.
        """
        # Use the full B&P tree for all branching strategies.
        # _solve_without_branching is kept as a fast path only when
        # branching_strategy is explicitly set to "none" or similar sentinel.
        if self.params.branching_strategy in ("edge", "ryan_foster", "divergence", "multi_edge_partition"):
            return self._solve_with_branching()  # type: ignore[return-value]
        return self._solve_without_branching()

    # ------------------------------------------------------------------
    # Solve without branching (column generation + direct IP)
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Solve with branching
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Node solving
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Column generation
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # Pricing dispatch
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Initial column generation
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # Tour construction with deduplication
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_master(self) -> VRPPMasterProblem:
        """Construct a fresh VRPPMasterProblem with the solver's parameters.

        Returns:
            VRPPMasterProblem: The initialized RMP.
        """
        return VRPPMasterProblem(  # type: ignore[abstract]  # protocol attrs BIG_M and prev_dual_* are not set in this constructor
            n_nodes=self.n_nodes,
            mandatory_nodes=self.mandatory_nodes,
            cost_matrix=self.cost_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            revenue_per_kg=self.R,
            cost_per_km=self.C,
            vehicle_limit=self.params.vehicle_limit,
        )

    def _make_pricing(self) -> Tuple[PricingSubproblem, Optional[RCSPPSolver]]:
        """Construct the pricing solvers for this configuration.

        Returns:
            A tuple containing:
                - heuristic_pricer: Always provided (PricingSubproblem).
                - exact_pricer: RCSPPSolver if use_exact_pricing=True, else None.
        """
        kwargs: Dict[str, Any] = dict(
            n_nodes=self.n_nodes,
            cost_matrix=self.cost_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            revenue_per_kg=self.R,
            cost_per_km=self.C,
            mandatory_nodes=self.mandatory_nodes,
        )

        heuristic_pricer = PricingSubproblem(**kwargs)
        exact_pricer = None

        if self.params.use_exact_pricing:
            exact_pricer = RCSPPSolver(
                **kwargs,
                use_ng_routes=self.params.use_ng_routes,
                ng_neighborhood_size=self.params.ng_neighborhood_size,
                ng_neighborhoods=self._ng_neighborhoods,
            )

        return heuristic_pricer, exact_pricer

    def _build_master_for_node(self, node: BranchNode, routes: List[Route]) -> VRPPMasterProblem:
        """Build a master problem pre-loaded with constraint-feasible routes.

        Args:
            node: The search node defining feasibility constraints.
            routes: Pool of routes to filter and load.

        Returns:
            VRPPMasterProblem: The populated RMP.
        """
        master = self._make_master()
        feasible = [r for r in routes if node.is_route_feasible(r)]
        master.build_model(feasible)
        return master

    def _is_integer_solution(self, route_values: Dict[int, float], tol: float = 1e-5) -> bool:
        """Return True if every LP route value is within *tol* of 0 or 1.

        Args:
            route_values: Mapping from route index to its continuous value.
            tol: Tolerance for integrality check.

        Returns:
            bool: True if solution is integer.
        """
        return all(abs(v - round(v)) <= tol for v in route_values.values())

    def _get_statistics(self) -> Dict[str, Any]:
        """Collect and return solver statistics.

        Returns:
            Dict[str, Any]: Mapping of metric names to values.
        """
        lp_b = self.lp_bound if self.lp_bound is not None else 0.0
        ip_s = self.ip_solution if self.ip_solution is not None else 0.0
        gap = abs(lp_b - ip_s) / max(abs(lp_b), 1e-10)

        stats: Dict[str, Any] = {
            "num_iterations": self.num_iterations,
            "num_columns_generated": self.num_columns_generated,
            "lp_bound": self.lp_bound,
            "ip_solution": self.ip_solution,
            "gap": gap,
            "branching_strategy": self.params.branching_strategy,
            "use_ng_routes": self.params.use_ng_routes,
            "ng_neighborhood_size": self.params.ng_neighborhood_size,
            "proven_optimal": self.proven_optimal,
        }
        if self.tree is not None:
            stats.update(self.tree.get_statistics())
        return stats
