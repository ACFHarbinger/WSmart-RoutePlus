"""Initial solution and LKH-3 routing of the LKH-3 ALNS.

Attributes:
    LKH3ALNSRoutingMixin: Builds and optimises the routes of ``LKH3_ALNS`` with LKH-3.

Example:
    >>> class LKH3_ALNS(LKH3ALNSRoutingMixin, LKH3ALNSOperatorsMixin): ...
"""

from __future__ import annotations

import logging
from random import Random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyStateRecorder

logger = logging.getLogger(__name__)


class LKH3ALNSRoutingMixin:
    """Initial solution, LKH-3 routing and route costing of :class:`LKH3_ALNS`."""

    # Instance state is assigned by LKH3_ALNS.__init__; declared here for the mixin.
    distance_matrix: np.ndarray
    wastes: Dict[int, float]
    capacity: float
    revenue: float
    cost_unit: float
    profit_aware_operators: bool
    mandatory_nodes: Set[int]
    seed: int
    rng: Random
    np_rng: np.random.Generator
    max_pool_size: int
    global_alpha: Optional[np.ndarray]
    _viz: Optional[PolicyStateRecorder]

    def _initialize_solution(
        self,
        lkh_trials: int,
        n_vehicles: int,
        popmusic_subpath_size: int,
        popmusic_trials: int,
        popmusic_max_candidates: int,
        max_k_opt: int,
        use_ip_merging: bool,
        subgradient_iterations: int,
        dynamic_topology_discovery: bool = False,
    ) -> Tuple[List[List[int]], float, float]:
        """
        Build initial feasible solution:
        - Start with mandatory nodes
        - Greedily add profitable nodes by marginal profit score
        - Stop when capacity limit reached or no profitable nodes remain

        Args:
            lkh_trials: Max LKH-3 iterations.
            n_vehicles: Number of vehicles.
            popmusic_subpath_size: POPMUSIC parameter.
            popmusic_trials: POPMUSIC parameter.
            popmusic_max_candidates: Maximum number of candidates for POPMUSIC.
            max_k_opt: Maximum k for k-opt.
            use_ip_merging: IP-based tour merging flag.
            subgradient_iterations: Subgradient optimization iterations.
            dynamic_topology_discovery: Toggle recursive B&B search.

        Returns:
            Tuple[List[List[int]], float, float]: (routes, objective, penalty).
        """
        # VRPP: Start with mandatory nodes + greedy Clarke-Wright savings
        active_nodes = list(self.mandatory_nodes)
        n = len(self.distance_matrix)

        # Compute Clarke-Wright savings-based score for each optional node.
        # savings(node) = (d(0, node) + d(node, 0) - d(node, nearest_neighbor))
        #                 * fill_factor
        # Higher savings → stronger candidate for merging into existing routes.
        profit_scores: List[Tuple[float, int]] = []
        for node in range(1, n):
            if node not in self.mandatory_nodes:
                fill = self.wastes.get(node, 0.0)
                depot_cost = self.distance_matrix[0, node] + self.distance_matrix[node, 0]

                # Find nearest neighbor (not depot, not self)
                nn_dist = float("inf")
                for other in range(1, n):
                    if other != node and self.distance_matrix[node, other] < nn_dist:
                        nn_dist = self.distance_matrix[node, other]

                # Clarke-Wright savings: how much we save by merging node
                # into an existing route vs. serving it with a dedicated route
                savings = depot_cost - nn_dist if nn_dist < float("inf") else 0.0

                # Combine savings with profit signal
                revenue = self.revenue * fill
                profit = revenue - self.cost_unit * depot_cost

                # Score = profit * (1 + savings_ratio) to favor
                # clustered profitable nodes
                savings_ratio = savings / max(depot_cost, 1e-6)
                score = profit * (1.0 + savings_ratio)
                profit_scores.append((score, node))

        # Sort by score descending
        profit_scores.sort(reverse=True)

        # Greedily add profitable nodes within capacity (First-Fit Bin-Packing heuristic)
        current_loads = [0.0] * n_vehicles
        # Account for mandatory nodes first
        for node in active_nodes:
            node_waste = self.wastes.get(node, 0.0)
            # Try to fit in existing vehicle
            fitted = False
            for v_idx in range(n_vehicles):
                if current_loads[v_idx] + node_waste <= self.capacity:
                    current_loads[v_idx] += node_waste
                    fitted = True
                    break
            if not fitted:
                # If mandatory nodes don't fit, we still proceed but with penalty
                current_loads[0] += node_waste

        for score, node in profit_scores:
            if node not in active_nodes and (not self.profit_aware_operators or score > 0):
                node_waste = self.wastes.get(node, 0.0)
                # First-Fit: try to find a vehicle with enough remaining capacity
                can_fit = False
                for v_idx in range(n_vehicles):
                    if current_loads[v_idx] + node_waste <= self.capacity:
                        current_loads[v_idx] += node_waste
                        can_fit = True
                        break
                if can_fit:
                    active_nodes.append(node)

        return self._route_nodes(
            active_nodes,
            lkh_trials,
            n_vehicles,
            popmusic_subpath_size,
            popmusic_trials,
            popmusic_max_candidates,
            max_k_opt,
            use_ip_merging,
            subgradient_iterations,
            dynamic_topology_discovery=dynamic_topology_discovery,
        )

    def _route_nodes(  # noqa: C901
        self,
        nodes: List[int],
        lkh_trials: int,
        n_vehicles: int,
        popmusic_subpath_size: int,
        popmusic_trials: int,
        popmusic_max_candidates: int,
        max_k_opt: int,
        use_ip_merging: bool,
        subgradient_iterations: int = 0,
        initial_routes: Optional[List[List[int]]] = None,
        dynamic_topology_discovery: bool = False,
        sa_max_trials: int = 0,
    ) -> Tuple[List[List[int]], float, float]:
        """
        Route a given set of nodes with LKH-3.

        Creates a sub-problem containing only the specified nodes, runs LKH-3,
        and maps the result back to original node indices.

        **Phase 2 Fix: Warm-Start Integration**
        If `initial_routes` is provided (non-None), the method flattens the routes
        into a 1D augmented tour with dummy depot indices and passes it to LKH-3
        as `initial_tour`. This preserves the repaired structure from destroy-repair
        operators and avoids expensive nearest-neighbor reconstruction.

        Args:
            nodes: List of customer node IDs to route.
            lkh_trials: Max LKH-3 iterations.
            n_vehicles: Number of vehicles.
            popmusic_subpath_size: POPMUSIC parameter.
            popmusic_trials: POPMUSIC parameter.
            popmusic_max_candidates: Maximum number of candidates for POPMUSIC.
            max_k_opt: Maximum k for k-opt.
            use_ip_merging: IP-based tour merging flag.
            initial_routes: Optional warm-start routes (global node indices).
                If provided, flattened to 1D tour and passed to solve_lkh3.
            dynamic_topology_discovery: Toggle recursive B&B search.
            subgradient_iterations: Number of subgradient iterations.
            sa_max_trials: Number of SA trials.

        Returns:
            Tuple[List[List[int]], float, float]: (routes_global, objective, penalty).
        """
        if not nodes:
            return [[]], 0.0, 0.0

        # Validate mandatory nodes in warm-start
        if initial_routes is not None and self.mandatory_nodes:
            warm_start_nodes = {n for r in initial_routes for n in r}
            missing = self.mandatory_nodes - warm_start_nodes
            if missing:
                logger.warning(
                    "Warm-start dropped mandatory nodes: %s — adding them back.",
                    missing,
                )
                nodes = list(set(nodes) | missing)

        # Build sub-problem distance matrix and demand array
        local_to_global = {0: 0}  # Depot always at index 0
        global_to_local = {0: 0}  # Global → Local mapping
        for idx, node in enumerate(nodes, start=1):
            local_to_global[idx] = node
            global_to_local[node] = idx

        n_sub = len(nodes) + 1
        sub_dist = np.zeros((n_sub, n_sub))
        sub_waste = np.zeros(n_sub)

        for i in range(n_sub):
            for j in range(n_sub):
                orig_i = local_to_global[i]
                orig_j = local_to_global[j]
                sub_dist[i, j] = self.distance_matrix[orig_i, orig_j]

        for i in range(1, n_sub):
            orig_node = local_to_global[i]
            sub_waste[i] = self.wastes.get(orig_node, 0.0)

        # Phase 2: Flatten initial_routes to 1D augmented tour (warm-start)
        initial_tour_local: Optional[List[int]] = None
        if initial_routes is not None:
            # Map global routes to local sub-problem indices
            local_routes: List[List[int]] = []
            for route in initial_routes:
                local_route = [global_to_local[n] for n in route if n in global_to_local]
                if local_route:
                    local_routes.append(local_route)

            # Flatten to 1D tour with augmented dummy depots [N, N+1, ...]
            if local_routes:
                # Flatten with depot separators: [0, route1, 0, route2, 0, ...]
                # Then convert to augmented dummy depot encoding
                n_original = n_sub  # Original graph size (before augmentation)
                n_routes = len(local_routes)

                # Build augmented tour: [0, r1, N, r2, N+1, r3, 0]
                initial_tour_local = [0]
                for idx, route in enumerate(local_routes):
                    initial_tour_local.extend(route)
                    # Cap number of dummies at n_vehicles - 1
                    if idx < n_routes - 1 and idx < n_vehicles - 1:
                        # Insert augmented dummy depot: N, N+1, N+2, ...
                        dummy_idx = n_original + idx
                        initial_tour_local.append(dummy_idx)
                initial_tour_local.append(0)

        # Phase 4: Filter global alpha-measures in O(1) for this subset
        local_candidates = {}
        if self.global_alpha is not None:
            for local_idx in range(n_sub):
                global_node = local_to_global[local_idx]
                row = self.global_alpha[global_node]

                # Get alpha measures for ALL active nodes relative to this one
                active_alphas = []
                for other_local_idx in range(n_sub):
                    if other_local_idx != local_idx:
                        other_global = local_to_global[other_local_idx]
                        active_alphas.append((row[other_global], other_local_idx))

                # Sort by alpha measure and take top K
                active_alphas.sort()
                local_candidates[local_idx] = [idx for _, idx in active_alphas[:popmusic_max_candidates]]
        else:
            # Fallback (should not happen with regular solve entry)
            local_candidates = {i: [] for i in range(n_sub)}

        from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.lkh3 import (
            solve_lkh3,
        )

        routes, cost, penalty = solve_lkh3(
            distance_matrix=sub_dist,
            initial_tour=initial_tour_local,  # Phase 2: Pass warm-start tour
            waste=sub_waste,
            capacity=self.capacity,
            max_trials=lkh_trials,
            popmusic_subpath_size=popmusic_subpath_size,
            popmusic_trials=popmusic_trials,
            popmusic_max_candidates=popmusic_max_candidates,
            max_k_opt=max_k_opt,
            use_ip_merging=use_ip_merging,
            max_pool_size=self.max_pool_size,
            subgradient_iterations=subgradient_iterations,
            n_vehicles=n_vehicles,
            candidate_set=local_candidates,  # Phase 3: Use correctly-mapped candidates
            recorder=self._viz,
            np_rng=self.np_rng,
            rng=self.rng,
            seed=self.seed,
            n_original=0,  # Use sub-problem size for augmentation
            dynamic_topology_discovery=dynamic_topology_discovery,
            sa_max_trials=sa_max_trials,
        )

        # Extract routes and map back to original indices
        routes_global = [[local_to_global[n] for n in route] for route in routes]

        # Compute objective
        routing_cost = self._compute_routing_cost(routes_global)
        if self.profit_aware_operators:
            # VRPP: Profit = Revenue - Cost
            collected_fill = sum(self.wastes.get(n, 0.0) for route in routes_global for n in route)
            profit = self.revenue * collected_fill - self.cost_unit * routing_cost
            return routes_global, float(profit), penalty
        else:
            # CVRP: Minimize cost + Penalty (return negative for maximization framework)
            return routes_global, -float(routing_cost + penalty * 1e6), penalty

    def _optimize_routes(
        self,
        routes: List[List[int]],
        lkh_trials: int,
        n_vehicles: int,
        popmusic_subpath_size: int,
        popmusic_trials: int,
        popmusic_max_candidates: int,
        max_k_opt: int,
        use_ip_merging: bool,
        subgradient_iterations: int,
        dynamic_topology_discovery: bool = False,
        sa_max_trials: int = 0,
    ) -> Tuple[List[List[int]], float, float]:
        """
        Re-optimize current routes with LKH-3 using warm-start.

        Passes the current route structure as ``initial_routes`` so that
        LKH-3 starts from the repaired tour rather than regenerating
        from scratch (fixes Cold Start Amnesia).

        Args:
            routes: Current routes to warm-start from.
            lkh_trials: Max LKH-3 iterations.
            n_vehicles: Number of vehicles.
            popmusic_subpath_size: POPMUSIC parameter.
            popmusic_trials: POPMUSIC parameter.
            popmusic_max_candidates: Maximum number of candidates for POPMUSIC.
            max_k_opt: Maximum k for k-opt.
            use_ip_merging: IP-based tour merging flag.
            subgradient_iterations: Subgradient optimization iterations.
            dynamic_topology_discovery: Toggle recursive B&B search.
            sa_max_trials: Number of SA trials.

        Returns:
            Tuple[List[List[int]], float, float]: (optimized_routes, objective, penalty).
        """
        active_nodes = [n for route in routes for n in route]
        return self._route_nodes(
            active_nodes,
            lkh_trials,
            n_vehicles,
            popmusic_subpath_size,
            popmusic_trials,
            popmusic_max_candidates,
            max_k_opt,
            use_ip_merging,
            subgradient_iterations=subgradient_iterations,
            initial_routes=routes,  # Phase 2: Warm-start from repaired routes
            dynamic_topology_discovery=dynamic_topology_discovery,
            sa_max_trials=sa_max_trials,
        )

    def _compute_routing_cost(self, routes: List[List[int]]) -> float:
        """
        Compute total routing distance.

        Sums up all depot-to-customer, customer-to-customer, and
        customer-to-depot edges across all routes.

        Args:
            routes: List of routes (no depot nodes).

        Returns:
            Total routing cost.
        """
        cost = 0.0
        for route in routes:
            if not route:
                continue
            # Depot to first customer
            cost += self.distance_matrix[0, route[0]]
            # Customer to customer
            for i in range(len(route) - 1):
                cost += self.distance_matrix[route[i], route[i + 1]]
            # Last customer to depot
            cost += self.distance_matrix[route[-1], 0]
        return cost
