"""In-house branch-and-price route improvement method."""

import logging
from typing import Any, Dict, List, Optional, cast

import numpy as np

logger = logging.getLogger(__name__)


class BranchPriceInhouseMixin:
    """Relocated branch price inhouse methods."""

    config: Dict[str, Any]  # set by the owning route improver

    def _solve_inhouse(
        self,
        input_routes: List[List[int]],
        dm: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        cost_per_km: float,
        revenue_kg: float,
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Optional[List[List[int]]]:
        """Delegate to the in-house BranchAndPriceSolver.

        Args:
            input_routes: Initial segmented routes.
            dm: Distance matrix.
            wastes: Bin waste amounts.
            capacity: Vehicle capacity.
            cost_per_km: Distance cost coefficient.
            revenue_kg: Revenue coefficient.
            mandatory_nodes: List of nodes that must be visited.
            kwargs: Algorithmic parameters prefixed with 'bp_'.

        Returns:
            Optional[List[List[int]]]: Re-optimized routes if improved, else None.
        """
        from .branch_and_price import BPConfig, BranchAndPriceSolver

        n_nodes = dm.shape[0] - 1

        # Build BPConfig from kwargs with bp_* prefixes, falling back to
        # route improver-appropriate defaults.
        #
        # Note: use_exact_pricing defaults to True here because a route improver
        # called once per episode should use the high-quality pricer, even though
        # the solver's own default is False (for speed inside inner loops).
        max_iterations = cast(
            int,
            kwargs.get("bp_max_iterations", 0)
            if kwargs.get("bp_max_iterations") is not None
            else self.config.get("bp_max_iterations", 100)
        )
        max_routes_per_iteration = cast(
            int,
            kwargs.get("bp_max_routes_per_iteration", 0)
            if kwargs.get("bp_max_routes_per_iteration") is not None
            else self.config.get("bp_max_routes_per_iteration", 10)
        )
        optimality_gap = cast(
            float,
            kwargs.get("bp_optimality_gap", 0.0)
            if kwargs.get("bp_optimality_gap") is not None
            else self.config.get("bp_optimality_gap", 1e-4)
        )
        branching_strategy = cast(
            str,
            kwargs.get("bp_branching_strategy", "")
            if kwargs.get("bp_branching_strategy") is not None
            else self.config.get("bp_branching_strategy", "edge")
        )
        max_branch_nodes = cast(
            int,
            kwargs.get("bp_max_branch_nodes", 0)
            if kwargs.get("bp_max_branch_nodes") is not None
            else self.config.get("bp_max_branch_nodes", 1000)
        )
        use_exact_pricing = cast(
            bool,
            kwargs.get("bp_use_exact_pricing", False)
            if kwargs.get("bp_use_exact_pricing") is not None
            else self.config.get("bp_use_exact_pricing", True)
        )
        use_ng_routes = cast(
            bool,
            kwargs.get("bp_use_ng_routes", False)
            if kwargs.get("bp_use_ng_routes") is not None
            else self.config.get("bp_use_ng_routes", True)
        )
        ng_neighborhood_size = cast(
            int,
            kwargs.get("bp_ng_neighborhood_size", 0)
            if kwargs.get("bp_ng_neighborhood_size") is not None
            else self.config.get("bp_ng_neighborhood_size", 8)
        )
        tree_search_strategy = cast(
            str,
            kwargs.get("bp_tree_search_strategy", "")
            if kwargs.get("bp_tree_search_strategy") is not None
            else self.config.get("bp_tree_search_strategy", "best_first")
        )
        vehicle_limit = cast(
            int,
            kwargs.get("bp_vehicle_limit", 0)
            if kwargs.get("bp_vehicle_limit") is not None
            else self.config.get("bp_vehicle_limit", 0)
        )
        cleanup_frequency = cast(
            int,
            kwargs.get("bp_cleanup_frequency", 0)
            if kwargs.get("bp_cleanup_frequency") is not None
            else self.config.get("bp_cleanup_frequency", 20)
        )
        cleanup_threshold = cast(
            float,
            kwargs.get("bp_cleanup_threshold", 0.0)
            if kwargs.get("bp_cleanup_threshold") is not None
            else self.config.get("bp_cleanup_threshold", -100.0)
        )
        early_termination_gap = cast(
            float,
            kwargs.get("bp_early_termination_gap", 0.0)
            if kwargs.get("bp_early_termination_gap") is not None
            else self.config.get("bp_early_termination_gap", 1e-3)
        )
        allow_heuristic_ryan_foster = cast(
            bool,
            kwargs.get("bp_allow_heuristic_ryan_foster", False)
            if kwargs.get("bp_allow_heuristic_ryan_foster") is not None
            else self.config.get("bp_allow_heuristic_ryan_foster", False)
        )
        params = BPConfig(
            max_iterations=max_iterations,
            max_routes_per_iteration=max_routes_per_iteration,
            optimality_gap=optimality_gap,
            branching_strategy=branching_strategy,
            max_branch_nodes=max_branch_nodes,
            use_exact_pricing=use_exact_pricing,
            use_ng_routes=use_ng_routes,
            ng_neighborhood_size=ng_neighborhood_size,
            tree_search_strategy=tree_search_strategy,
            vehicle_limit=vehicle_limit,
            cleanup_frequency=cleanup_frequency,
            cleanup_threshold=cleanup_threshold,
            early_termination_gap=early_termination_gap,
            allow_heuristic_ryan_foster=allow_heuristic_ryan_foster,
        )

        solver = BranchAndPriceSolver(
            n_nodes=n_nodes,
            cost_matrix=dm,
            wastes=wastes,
            capacity=capacity,
            revenue_per_kg=revenue_kg,
            cost_per_km=cost_per_km,
            mandatory_nodes=set(mandatory_nodes),
            params=params,
        )

        flat_tour, profit, statistics = solver.solve()

        if not flat_tour or len(flat_tour) < 2:
            return None

        logger.info(
            "branch_and_price (in-house): profit=%.2f, iterations=%d, columns=%d, proven_optimal=%s",
            profit,
            statistics.get("num_iterations", 0),
            statistics.get("num_columns_generated", 0),
            statistics.get("proven_optimal", False),
        )

        # Convert flat depot-separated tour to list-of-routes
        routes: List[List[int]] = []
        current: List[int] = []
        for node in flat_tour:
            if node == 0:
                if current:
                    routes.append(current)
                    current = []
            else:
                current.append(node)
        if current:
            routes.append(current)

        return routes
