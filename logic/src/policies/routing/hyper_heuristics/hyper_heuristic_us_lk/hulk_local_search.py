"""Intra-route local search moves (2-opt, 3-opt, swap, relocate) for HULK, mixed into HULKOperators as HULKLocalSearchMixin."""

import random
from typing import (
    Dict,
    List,
    Tuple,
)

import numpy as np

from .solution import Solution


class HULKLocalSearchMixin:
    # Provided by HULKOperators.__init__; annotations only.
    dist_matrix: np.ndarray
    wastes: Dict[int, float]
    capacity: float
    R: float
    C: float
    rng: random.Random

    def _simple_2_opt(self, route: List[int]) -> Tuple[List[int], bool]:
        """
        Simple 2-opt implementation.

        Args:
            route: Route to apply the move to.

        Returns:
            Tuple[List[int], bool]: Tuple containing the new route and a boolean indicating whether the route was improved.
        """
        n = len(route)
        if n <= 3:
            return route, False

        best_route = list(route)
        best_dist = self._calc_route_distance(best_route)
        improved = False

        for i in range(n - 2):
            for j in range(i + 2, n):
                # Reverse segment between i+1 and j
                new_route = route[: i + 1] + route[i + 1 : j + 1][::-1] + route[j + 1 :]
                new_dist = self._calc_route_distance(new_route)

                if new_dist < best_dist - 1e-6:
                    best_route = new_route
                    best_dist = new_dist
                    improved = True

        return best_route, improved

    def _calc_route_distance(self, route: List[int]) -> float:
        """
        Calculate distance for a single route.

        Args:
            route: Route to calculate the distance for.

        Returns:
            float: Distance of the route.
        """
        if not route:
            return 0.0
        dist = self.dist_matrix[0][route[0]]
        for i in range(len(route) - 1):
            dist += self.dist_matrix[route[i]][route[i + 1]]
        dist += self.dist_matrix[route[-1]][0]
        return dist

    def apply_2_opt(self, solution: Solution) -> Solution:
        """Apply 2-opt local search.

        Args:
            solution: Solution to apply the move to.

        Returns:
            Solution: New solution with the move applied.
        """
        routes = [list(r) for r in solution.routes]
        improved = False

        for r_idx in range(len(routes)):
            new_route, improvement = self._simple_2_opt(routes[r_idx])
            if improvement:
                routes[r_idx] = new_route
                improved = True

        if improved:
            return Solution(routes, self.dist_matrix, self.wastes, self.capacity, self.R, self.C)
        return solution

    def apply_3_opt(self, solution: Solution) -> Solution:
        """
        Apply 3-opt (K-opt) local search.
        Follows the 3-opt reconnection logic for VRP tours.

        Args:
            solution: Solution to apply the move to.

        Returns:
            Solution: New solution with the move applied.
        """
        routes = [list(r) for r in solution.routes]
        improved = False

        for r_idx in range(len(routes)):
            route = routes[r_idx]
            if len(route) < 4:
                continue

            # Simplified 3-opt: try random 3-edge reconnections
            for _ in range(5):
                if len(route) < 4:
                    break
                i, j, k = sorted(self.rng.sample(range(len(route)), 3))

                # Standard 3-opt moves
                best_r = route
                best_d = self._calc_route_distance(route)

                s1, s2, s3, s4 = route[: i + 1], route[i + 1 : j + 1], route[j + 1 : k + 1], route[k + 1 :]

                options = [
                    s1 + s2[::-1] + s3[::-1] + s4,
                    s1 + s3 + s2 + s4,
                    s1 + s3[::-1] + s2 + s4,
                    s1 + s2 + s3[::-1] + s4,
                    s1 + s3 + s2[::-1] + s4,
                ]

                for opt in options:
                    d = self._calc_route_distance(opt)
                    if d < best_d - 1e-6:
                        best_d = d
                        best_r = opt
                        improved = True

                route = best_r
            routes[r_idx] = route

        if improved:
            return Solution(routes, self.dist_matrix, self.wastes, self.capacity, self.R, self.C)
        return solution

    def apply_swap(self, solution: Solution) -> Solution:
        """
        Apply swap move between routes.

        Args:
            solution: Solution to apply the move to.

        Returns:
            Solution: New solution with the move applied.
        """
        routes = [list(r) for r in solution.routes]
        if len(routes) < 2:
            return solution

        best_solution = solution
        best_cost = solution._cost
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                if not routes[i] or not routes[j]:
                    continue

                for pos_i in range(len(routes[i])):
                    for pos_j in range(len(routes[j])):
                        test_routes = [list(r) for r in routes]
                        test_routes[i][pos_i], test_routes[j][pos_j] = (
                            test_routes[j][pos_j],
                            test_routes[i][pos_i],
                        )

                        load_i = sum(self.wastes.get(n, 0) for n in test_routes[i])
                        load_j = sum(self.wastes.get(n, 0) for n in test_routes[j])

                        if load_i <= self.capacity and load_j <= self.capacity:
                            test_sol = Solution(
                                test_routes, self.dist_matrix, self.wastes, self.capacity, self.R, self.C
                            )
                            test_cost = test_sol._cost
                            if test_cost < best_cost - 1e-6:
                                best_solution = test_sol
                                best_cost = test_cost
        return best_solution

    def apply_relocate(self, solution: Solution) -> Solution:
        """Apply relocate move.

        Args:
            solution: Solution to apply the move to.

        Returns:
            Solution: New solution with the move applied.
        """
        routes = [list(r) for r in solution.routes]
        best_solution = solution
        best_cost = solution._cost
        for i in range(len(routes)):
            if not routes[i]:
                continue
            for pos in range(len(routes[i])):
                for j in range(len(routes)):
                    if i == j:
                        continue
                    for ins_pos in range(len(routes[j]) + 1):
                        test_routes = [list(r) for r in routes]
                        removed = test_routes[i].pop(pos)
                        test_routes[j].insert(ins_pos, removed)
                        load_j = sum(self.wastes.get(n, 0) for n in test_routes[j])
                        if load_j <= self.capacity:
                            test_sol = Solution(
                                test_routes, self.dist_matrix, self.wastes, self.capacity, self.R, self.C
                            )
                            test_cost = test_sol._cost
                            if test_cost < best_cost - 1e-6:
                                best_solution = test_sol
                                best_cost = test_cost
        return best_solution
