"""
LKH-3 Adaptive Large Neighborhood Search (ALNS) Matheuristic.

Wraps the LKH-3 k-opt core in an outer ruin-and-recreate loop to handle
VRPP subset selection. The LKH-3 engine optimizes routing for the current
active subset, while ALNS operators manage node inclusion/exclusion based
on profitability.

Architecture
------------
1. **Initialize**: Build feasible subset + route with LKH-3
2. **Optimize**: Run LKH-3 until local routing optimum
3. **Plateau Check**: If stuck, apply Destroy → Repair
4. **Deep Plateau**: If stuck for N iters, apply Perturbation
5. **Loop**: Return modified solution to step 2

Operator Dispatch:
    - ``profit_aware_operators=False``: Standard CVRP operators (removed nodes only)
    - ``profit_aware_operators=True``: VRPP operators (removed + unvisited pool)

Attributes:
    LKH3_ALNS: Main solver class combining LKH-3 routing with ALNS outer loop.

Example:
    >>> from logic.src.policies.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search import LKH3_ALNS
    >>> solver = LKH3_ALNS(
    ...     distance_matrix=dist,
    ...     wastes={1: 10, 2: 20, 3: 30},
    ...     capacity=100.0,
    ...     revenue=2.0,
    ...     cost_unit=1.0,
    ...     profit_aware_operators=True,
    ...     mandatory_nodes=[1],
    ...     seed=42,
    ... )
    >>> routes, profit = solver.solve(max_iterations=100, lkh_trials=500, n_vehicles=3)
"""

from __future__ import annotations

import logging
from random import Random
from typing import Dict, List, Optional, Tuple

import numpy as np

# solve_lkh3 imported locally in _optimize_routes or solve() to break circularity
# Destroy operators
# Perturbation operators
# Repair operators
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.alns_operators import LKH3ALNSOperatorsMixin
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.alns_routing import LKH3ALNSRoutingMixin
from logic.src.tracking.viz_mixin import PolicyStateRecorder

logger = logging.getLogger(__name__)


class LKH3_ALNS(LKH3ALNSRoutingMixin, LKH3ALNSOperatorsMixin):
    """
    LKH-3 + Adaptive Large Neighborhood Search matheuristic for VRPP.

    Implements a two-level optimization architecture where the LKH-3 core
    handles routing optimization for a given subset of nodes, while the ALNS
    outer loop manages subset selection via adaptive destroy-repair operators.

    Attributes:
        distance_matrix: (N×N) distance matrix (node 0 = depot).
        wastes: Dict mapping node ID to demand/profit value.
        capacity: Vehicle capacity constraint.
        revenue: Revenue per unit collected (R parameter).
        cost_unit: Cost per distance unit (C parameter).
        profit_aware_operators: If True, operators consider unvisited nodes
            and make profit-based decisions (VRPP mode).
        mandatory_nodes: Set of nodes that must be visited regardless of profit.
        coords: Node coordinates (N×2) for geometric operators like sector removal.
        np_rng: NumPy random generator for array operations.
        rng: Random number generator for reproducibility.
        seed: Seed for the random number generator.
        recorder: Optional state recorder for visualization.
        history: Historical cost tracking for adaptive operator selection.
        elite_pool: Pool of elite solutions for perturbation operators.
        max_pool_size: Maximum number of elite solutions to retain.
        n_original: Number of original nodes (before augmentation).
        R: Revenue per unit collected (VRPP parameter).
        C: Cost per distance unit.
    """

    def __init__(
        self,
        distance_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        revenue: float = 1.0,
        cost_unit: float = 1.0,
        profit_aware_operators: bool = False,
        mandatory_nodes: Optional[List[int]] = None,
        coords: Optional[np.ndarray] = None,
        np_rng: Optional[np.random.Generator] = None,
        rng: Optional[Random] = None,
        seed: int = 42,
        recorder: Optional[PolicyStateRecorder] = None,
        max_pool_size: int = 5,
        n_original: int = 0,
        R: float = 1.0,
        C: float = 1.0,
        perturb_operator_weights: Optional[List[float]] = None,
    ):
        """Initialize LKH-3 ALNS matheuristic solver.

        Args:
            distance_matrix: (N×N) symmetric distance matrix.
            wastes: Node demands/profits as {node_id: value}.
            capacity: Vehicle capacity.
            revenue: Revenue per unit collected (VRPP parameter).
            cost_unit: Cost per distance unit.
            profit_aware_operators: Toggle VRPP mode (subset selection).
            mandatory_nodes: Nodes that must be visited.
            coords: Node coordinates (N×2) for geometric operators.
            np_rng: NumPy random generator for array operations.
            rng: Random number generator for reproducibility.
            seed: Seed for the random number generator.
            recorder: Optional state recorder for visualization.
            max_pool_size: Maximum number of elite solutions to retain.
            n_original: Number of original nodes (before augmentation).
            R: Revenue per unit collected (VRPP parameter).
            C: Cost per distance unit.
            perturb_operator_weights: Weights for each perturbation operator.
        """
        self.distance_matrix = distance_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.revenue = revenue
        self.cost_unit = cost_unit
        self.profit_aware_operators = profit_aware_operators
        self.mandatory_nodes = set(mandatory_nodes) if mandatory_nodes else set()
        self.coords = coords
        self.seed = seed
        self.rng = rng if rng is not None else Random(seed)
        # Seed np_rng slightly differently to avoid correlation with pure Python Random
        self.np_rng = np_rng if np_rng is not None else np.random.default_rng(seed + 1 if seed is not None else None)
        self.n_original = n_original
        self.R = R
        self.C = C
        self.perturb_operator_weights = perturb_operator_weights if perturb_operator_weights is not None else [0.6, 0.4]

        # Historical memory for adaptive operator selection
        self.history: Dict[int, float] = {}

        # ALNS: Adaptive operator selection weights
        self.destroy_weights = np.ones(3)  # [hist, neighbor, sector]
        self.repair_weights = np.ones(3)  # [savings, deep, nearest]
        self.destroy_scores = np.zeros(3)
        self.repair_scores = np.zeros(3)
        self.destroy_calls = np.zeros(3)
        self.repair_calls = np.zeros(3)
        self.last_destroy_idx = -1
        self.last_repair_idx = -1

        # Global alpha-measures (Phase 4: Optimization)
        # Computed once in solve() and filtered in O(1) in _route_nodes
        self.global_alpha: Optional[np.ndarray] = None

        # Elite solution pool for perturbation
        self.elite_pool: List[List[List[int]]] = []
        self.max_pool_size = max_pool_size

        # Cached candidates for warm-starting
        self.cached_candidates: Optional[Dict[int, List[int]]] = None

        # Recorder for output visualization
        self._viz = recorder

    def solve(
        self,
        max_iterations: int = 100,
        lkh_trials: int = 500,
        n_vehicles: int = 3,
        plateau_limit: int = 10,
        deep_plateau_limit: int = 30,
        popmusic_subpath_size: int = 50,
        popmusic_trials: int = 50,
        popmusic_max_candidates: int = 5,
        max_k_opt: int = 5,
        use_ip_merging: bool = True,
        subgradient_iterations: int = 50,
        dynamic_topology_discovery: bool = False,
        native_prize_collecting: bool = False,
        sa_max_trials: int = 0,
    ) -> Tuple[List[List[int]], float, float]:
        """
        Main ALNS+LKH-3 matheuristic loop.

        Algorithm outline:
        1. If native_prize_collecting is True, bypass ALNS and solve directly
           using the Jonker-Volgenant ATSP transformation.
        2. Else, run the standard LNS loop with ALNS weight adaptation.

        Args:
            max_iterations: Total ALNS iterations.
            lkh_trials: Max k-opt iterations per LKH-3 run.
            n_vehicles: Number of available vehicles.
            plateau_limit: Iterations without improvement before Destroy-Repair.
            deep_plateau_limit: Iterations before Perturbation.
            popmusic_subpath_size: POPMUSIC sub-path size for large instances.
            popmusic_trials: Number of POPMUSIC runs.
            popmusic_max_candidates: Maximum number of candidates for POPMUSIC.
            max_k_opt: Maximum k for k-opt moves (2-5).
            use_ip_merging: If True, use IP-based tour recombination.
            subgradient_iterations: Iterations for subgradient optimization.
            dynamic_topology_discovery: Toggle recursive B&B search.
            native_prize_collecting: Toggle Jonker-Volgenant ATSP.
            sa_max_trials: Number of SA trials.

        Returns:
            Tuple of (best_routes, best_objective, best_penalty) where:
            - best_routes: List of routes (each route is list of node IDs)
            - best_objective: Profit (VRPP) or negative cost (CVRP)
            - best_penalty: Final penalty (should be 0 for feasible solutions)
        """
        if native_prize_collecting:
            from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.graph_augmentation import (
                augment_prize_collecting_graph,
            )
            from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.lkh3 import solve_lkh3

            # ATSP Transformation
            aug_dist, _, n_original = augment_prize_collecting_graph(
                self.distance_matrix,
                self.wastes,
            )

            # Solve directly with LKH-3 engine
            routes, cost, penalty = solve_lkh3(
                distance_matrix=aug_dist,
                max_trials=lkh_trials,
                max_k_opt=max_k_opt,
                use_ip_merging=use_ip_merging,
                subgradient_iterations=subgradient_iterations,
                np_rng=self.np_rng,
                rng=self.rng,
                seed=self.seed,
                dynamic_topology_discovery=dynamic_topology_discovery,
                # Mandatory nodes are handled structurally in ATSP
            )

            # Recalculate objective on global scale
            objective = self._compute_objective(routes)
            return routes, objective, penalty

        # --- Standard ALNS Loop ---
        # 1. Initialize with greedy solution
        # Pre-compute global alpha-measures once (Phase 4 optimization)
        from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.lkh3 import (
            compute_alpha_measures,
        )

        self.global_alpha = compute_alpha_measures(self.distance_matrix)

        best_routes, best_obj, best_penalty = self._initialize_solution(
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
        current_routes = [r[:] for r in best_routes]

        iterations_since_improvement = 0
        iterations_since_last_deep_perturb = 0

        # ALNS parameters
        r_ema = 0.1  # Adaptation rate

        # 2. Main ALNS loop
        for _iteration in range(max_iterations):
            # 2a. Optimize current subset with LKH-3
            optimized_routes, obj, penalty = self._optimize_routes(
                current_routes,
                lkh_trials,
                n_vehicles,
                popmusic_subpath_size,
                popmusic_trials,
                popmusic_max_candidates,
                max_k_opt,
                use_ip_merging,
                subgradient_iterations,
                dynamic_topology_discovery=dynamic_topology_discovery,
                sa_max_trials=sa_max_trials,
            )

            # 2b. Update best solution and ALNS scoring
            score = 0
            if penalty < best_penalty - 1e-6 or (abs(penalty - best_penalty) < 1e-6 and obj > best_obj + 1e-6):
                # Update global best score
                score = 3
                best_routes = [r[:] for r in optimized_routes]
                best_obj = obj
                best_penalty = penalty
                iterations_since_improvement = 0
                iterations_since_last_deep_perturb = 0
                self._update_elite_pool(best_routes)
            elif abs(penalty - best_penalty) < 1e-6 and obj > self._compute_objective(current_routes):
                # Local improvement score
                score = 1
                iterations_since_improvement += 1
                iterations_since_last_deep_perturb += 1
            else:
                # Rejected score
                score = 0
                iterations_since_improvement += 1
                iterations_since_last_deep_perturb += 1

            # Update scores for the last used operators
            if self.last_destroy_idx != -1:
                self.destroy_scores[self.last_destroy_idx] += score
            if self.last_repair_idx != -1:
                self.repair_scores[self.last_repair_idx] += score

            # Periodically update weights (e.g., every iteration or block)
            # Using EMA update for weights
            if self.last_destroy_idx != -1:
                idx = self.last_destroy_idx
                perf = self.destroy_scores[idx] / max(1, self.destroy_calls[idx])
                self.destroy_weights[idx] = (1 - r_ema) * self.destroy_weights[idx] + r_ema * perf

            if self.last_repair_idx != -1:
                idx = self.last_repair_idx
                perf = self.repair_scores[idx] / max(1, self.repair_calls[idx])
                self.repair_weights[idx] = (1 - r_ema) * self.repair_weights[idx] + r_ema * perf

            # Record state for visualization
            if self._viz:
                self._viz.record(
                    iteration=_iteration,
                    best_obj=best_obj,
                    curr_obj=obj,
                )

            # Update historical scores for adaptive operator selection
            self._update_history(optimized_routes, obj)

            # 2c. Standard plateau: Destroy-Repair
            if iterations_since_improvement >= plateau_limit:
                current_routes = self._destroy_repair(optimized_routes)
                iterations_since_improvement = 0  # Only reset standard counter

            # 2d. Deep plateau: Perturbation (INDEPENDENT check, not elif)
            if iterations_since_last_deep_perturb >= deep_plateau_limit and self.elite_pool:
                current_routes = self._perturbation(optimized_routes)
                iterations_since_last_deep_perturb = 0  # Only reset deep counter

        return best_routes, best_obj, best_penalty

    def _destroy_repair(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Apply Destroy (Ruin) → Repair (Recreate) operators.

        Randomly selects one destroy operator and one repair operator,
        removes 10-20% of nodes, then reinserts them.

        Args:
            routes: Current solution.

        Returns:
            Modified routes after destroy-repair.
        """
        # Select operators via roulette wheel
        destroy_op = self._select_destroy_operator()
        repair_op = self._select_repair_operator()

        # Determine removal size (10-20% of nodes)
        total_nodes = sum(len(r) for r in routes)
        n_remove = max(1, int(self.rng.uniform(0.1, 0.2) * total_nodes))

        # Destroy phase
        modified_routes, removed_nodes = destroy_op(routes, n_remove)

        # Repair phase
        repaired_routes = repair_op(modified_routes, removed_nodes)

        return repaired_routes

    def _compute_objective(self, routes: List[List[int]]) -> float:
        """Compute current objective (profit or negative cost).

        Args:
            routes: List of routes (no depot nodes).

        Returns:
            float: Profit (VRPP) or negative routing cost (CVRP).
        """
        routing_cost = self._compute_routing_cost(routes)
        if self.profit_aware_operators:
            collected_fill = sum(self.wastes.get(n, 0.0) for route in routes for n in route)
            return self.revenue * collected_fill - self.cost_unit * routing_cost
        else:
            return -float(routing_cost)

    # -----------------------------------------------------------------------
    # Operator Wrappers (Standard CVRP)
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Operator Wrappers (VRPP Profit-Aware)
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Perturbation Operator
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Utility Methods
    # -----------------------------------------------------------------------
