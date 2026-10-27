"""Operator selection, perturbation, history and elite pool of the LKH-3 ALNS.

These methods were split out of ``LKH3ALNSOperatorsMixin`` in ``alns_operators``
and are combined back into it by inheritance.
"""

from __future__ import annotations

from random import Random
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from logic.src.policies.helpers.operators.destroy_ruin.route import (
    route_profit_removal,
    route_removal,
)
from logic.src.policies.helpers.operators.perturbation_shaking.evolutionary import (
    evolutionary_perturbation,
    evolutionary_perturbation_profit,
)
from logic.src.policies.helpers.operators.perturbation_shaking.genetic_transformation import (
    genetic_transformation,
    genetic_transformation_profit,
)


class LKH3ALNSSelectionMixin:
    """Operator selection, worst-route selection, perturbation, history and elite pool."""

    # Instance state is assigned by LKH3_ALNS.__init__; declared here for the mixin.
    distance_matrix: np.ndarray
    wastes: Dict[int, float]
    capacity: float
    revenue: float
    cost_unit: float
    R: float
    C: float
    profit_aware_operators: bool
    mandatory_nodes: Set[int]
    coords: Optional[np.ndarray]
    rng: Random
    history: Dict[int, float]
    elite_pool: List[List[List[int]]]
    max_pool_size: int
    destroy_weights: np.ndarray
    repair_weights: np.ndarray
    destroy_calls: np.ndarray
    repair_calls: np.ndarray
    perturb_operator_weights: List[float]
    if TYPE_CHECKING:
        # Implemented on LKH3ALNSOperatorsMixin; declared so this mixin can call them.
        def _wrap_historical_removal(
            self, routes: List[List[int]], n_remove: int
        ) -> Tuple[List[List[int]], List[int]]: ...
        def _wrap_neighbor_removal(
            self, routes: List[List[int]], n_remove: int
        ) -> Tuple[List[List[int]], List[int]]: ...
        def _wrap_sector_removal(
            self, routes: List[List[int]], n_remove: int
        ) -> Tuple[List[List[int]], List[int]]: ...
        def _wrap_historical_profit_removal(
            self, routes: List[List[int]], n_remove: int
        ) -> Tuple[List[List[int]], List[int]]: ...
        def _wrap_neighbor_profit_removal(
            self, routes: List[List[int]], n_remove: int
        ) -> Tuple[List[List[int]], List[int]]: ...
        def _wrap_sector_profit_removal(
            self, routes: List[List[int]], n_remove: int
        ) -> Tuple[List[List[int]], List[int]]: ...
        def _wrap_savings_insertion(self, routes: List[List[int]], removed_nodes: List[int]) -> List[List[int]]: ...
        def _wrap_deep_insertion(self, routes: List[List[int]], removed_nodes: List[int]) -> List[List[int]]: ...
        def _wrap_nearest_insertion(self, routes: List[List[int]], removed_nodes: List[int]) -> List[List[int]]: ...
        def _wrap_savings_profit_insertion(
            self, routes: List[List[int]], removed_nodes: List[int]
        ) -> List[List[int]]: ...
        def _wrap_deep_profit_insertion(self, routes: List[List[int]], removed_nodes: List[int]) -> List[List[int]]: ...
        def _wrap_nearest_profit_insertion(
            self, routes: List[List[int]], removed_nodes: List[int]
        ) -> List[List[int]]: ...

    def _select_destroy_operator(self) -> Callable[[List[List[int]], int], Tuple[List[List[int]], List[int]]]:
        """Select a destroy operator via ALNS roulette wheel.

        Returns:
            Callable: Destroy operator function (routes, n_remove) -> (routes, removed).
        """
        if self.profit_aware_operators:
            operators = [
                self._wrap_historical_profit_removal,
                self._wrap_neighbor_profit_removal,
                self._wrap_sector_profit_removal,
            ]
        else:
            operators = [
                self._wrap_historical_removal,
                self._wrap_neighbor_removal,
                self._wrap_sector_removal,
            ]

        # ALNS selection
        idx = self.rng.choices(range(len(operators)), weights=self.destroy_weights.tolist(), k=1)[0]
        self.last_destroy_idx = idx
        self.destroy_calls[idx] += 1
        return operators[idx]

    def _select_repair_operator(self) -> Callable[[List[List[int]], List[int]], List[List[int]]]:
        """Select a repair operator via ALNS roulette wheel.

        Returns:
            Callable: Repair operator function (routes, removed_nodes) -> routes.
        """
        if self.profit_aware_operators:
            operators = [
                self._wrap_savings_profit_insertion,
                self._wrap_deep_profit_insertion,
                self._wrap_nearest_profit_insertion,
            ]
        else:
            operators = [
                self._wrap_savings_insertion,
                self._wrap_deep_insertion,
                self._wrap_nearest_insertion,
            ]

        # ALNS selection
        idx = self.rng.choices(range(len(operators)), weights=self.repair_weights.tolist(), k=1)[0]
        self.last_repair_idx = idx
        self.repair_calls[idx] += 1
        return operators[idx]

    def _select_worst_routes(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Select the n worst routes based on profit-oriented strategy.

        Args:
            routes: Current solution routes.
            n: Number of routes to remove.

        Returns:
            Tuple[List[List[int]], List[int]]: (remaining_routes, removed_nodes).
        """
        target_routes = []
        all_removed_nodes = []
        for _ in range(n):
            # Break early if we run out of routes
            if not routes:
                break

            if self.profit_aware_operators:
                routes, removed = route_profit_removal(
                    routes=routes,
                    strategy="worst_profit",
                    dist_matrix=self.distance_matrix,
                    wastes=self.wastes,
                    R=self.R,
                    C=self.C,
                    rng=self.rng,
                )
            else:
                routes, removed = route_removal(
                    routes=routes,
                    strategy="costliest",
                    dist_matrix=self.distance_matrix,
                    wastes=self.wastes,
                    rng=self.rng,
                )
            target_routes.append(routes)
            all_removed_nodes.extend(removed)
        return target_routes, all_removed_nodes  # type: ignore[return-value]

    def _perturbation(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Apply deep perturbation operator against elite solution.

        Randomly selects an elite solution from the pool and applies
        genetic transformation to preserve common edges while rewiring
        the rest.

        Args:
            routes: Current solution.

        Returns:
            Perturbed routes.
        """
        # 1. Fallback: If no elite solutions exist, you MUST use Evolutionary
        if not self.elite_pool:
            # Target the 2 smallest/worst routes for localized balancing
            target_routes, _ = self._select_worst_routes(routes, n=2)
            if self.profit_aware_operators:
                return evolutionary_perturbation_profit(
                    routes,
                    self.distance_matrix,
                    self.capacity,
                    self.wastes,
                    self.R,
                    self.C,
                    target_routes,
                    rng=self.rng,
                )
            else:
                return evolutionary_perturbation(
                    routes, self.distance_matrix, self.capacity, self.wastes, target_routes, rng=self.rng
                )

        # 2. Adaptive Selection (Roulette Wheel)
        operators = ["genetic", "evolutionary"]
        chosen_op = self.rng.choices(operators, weights=self.perturb_operator_weights, k=1)[0]
        if chosen_op == "genetic":
            # Pull a random elite solution to ensure diversity
            elite_ref = self.rng.choice(self.elite_pool)

            if self.profit_aware_operators:
                return genetic_transformation_profit(
                    routes, elite_ref, self.distance_matrix, self.wastes, self.capacity, self.R, self.C, self.rng
                )
            else:
                return genetic_transformation(
                    routes, elite_ref, self.distance_matrix, self.wastes, self.capacity, self.rng
                )

        elif chosen_op == "evolutionary":
            target_routes, _ = self._select_worst_routes(routes, n=2)
            if self.profit_aware_operators:
                return evolutionary_perturbation_profit(
                    routes,
                    self.distance_matrix,
                    self.capacity,
                    self.wastes,
                    self.R,
                    self.C,
                    target_routes,
                    rng=self.rng,
                )
            else:
                return evolutionary_perturbation(
                    routes, self.distance_matrix, self.capacity, self.wastes, target_routes, rng=self.rng
                )

        return routes

    def _update_history(self, routes: List[List[int]], obj: float) -> None:
        """
        Update historical cost scores for nodes.

        Uses exponential moving average to track node quality over time.
        Nodes frequently appearing in high-quality solutions get HIGHER scores,
        since higher obj is better (VRPP: profit, CVRP: negative cost).

        Args:
            routes: Current solution.
            obj: Objective value (profit or negative cost).
        """
        alpha = 0.1  # EMA weight for new observation

        for route in routes:
            for node in route:
                # We store -obj in history because destruction operators (like historical_removal)
                # assume higher scores are WORSE (higher penalty/cost).
                # VRPP: obj is profit -> store -profit
                # CVRP: obj is -cost -> store -(-cost) = cost
                score = -float(obj)
                if node not in self.history:
                    self.history[node] = score
                else:
                    # Update: history[n] ← α·score + (1-α)·history[n]
                    self.history[node] = alpha * score + (1 - alpha) * self.history[node]

    def _update_elite_pool(self, routes: List[List[int]]) -> None:
        """
        Add solution to elite pool.

        Maintains a fixed-size pool of diverse high-quality solutions
        for use in perturbation operators.

        Args:
            routes: Solution to add to pool.
        """
        # Deep copy routes
        self.elite_pool.append([r[:] for r in routes])

        # Maintain pool size
        if len(self.elite_pool) > self.max_pool_size:
            self.elite_pool.pop(0)
