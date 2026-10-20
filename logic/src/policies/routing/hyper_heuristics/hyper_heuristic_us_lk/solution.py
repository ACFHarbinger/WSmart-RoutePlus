"""
Solution representation for HULK hyper-heuristic.

Provides efficient solution evaluation and manipulation for VRP problems.

Attributes:
    Solution: Class for solution representation.

Example:
    >>> from logic.src.policies.routing.hyper_heuristics.hyper_heuristic_us_lk import Solution
    >>> solution = Solution([[1, 2], [3, 4]], np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), {1: 10, 2: 20, 3: 30, 4: 40}, 100, 2, 3)
    >>> print(solution)
    Solution(routes=2, nodes=4, profit=60.00, cost=60.00, revenue=200.00)
"""

from typing import Dict, List, Optional

import numpy as np


class Solution:
    """
    Solution wrapper for HULK hyper-heuristic.

    Maintains routes and efficiently computes cost/profit metrics.

    Attributes:
        routes: List of routes (each route is a list of node indices).
        dist_matrix: Distance matrix.
        wastes: Node waste dictionary.
        capacity: Vehicle capacity.
        R: Revenue per unit waste.
        C: Cost per unit distance.

        _cost: Cached cost.
        _revenue: Cached revenue.
        _profit: Cached profit.
    """

    def __init__(
        self,
        routes: List[List[int]],
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
    ):
        """
        Initialize solution.

        Args:
            routes: List of routes (each route is a list of node indices).
            dist_matrix: Distance matrix.
            wastes: Node waste dictionary.
            capacity: Vehicle capacity.
            R: Revenue per unit waste.
            C: Cost per unit distance.
        """
        self.routes = routes
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C

        # Compute metrics
        self._cost = 0.0
        self._revenue = 0.0
        self._profit = 0.0
        self._compute_metrics()

    def _compute_metrics(self):
        """
        Compute cost, revenue, and profit.
        """
        self._cost = self.calculate_cost()
        self._revenue = self.calculate_revenue()
        self._profit = self._revenue - self._cost

    def calculate_cost(self) -> float:
        """
        Calculate total routing cost.

        Returns:
            float: Total routing cost.
        """
        total_dist = 0.0
        for route in self.routes:
            if not route:
                continue
            # Depot to first node
            dist = self.dist_matrix[0][route[0]]
            # Between nodes
            for i in range(len(route) - 1):
                dist += self.dist_matrix[route[i]][route[i + 1]]
            # Last node to depot
            dist += self.dist_matrix[route[-1]][0]
            total_dist += dist
        return total_dist * self.C

    def calculate_revenue(self) -> float:
        """
        Calculate total revenue from collected waste.

        Returns:
            float: Total revenue.
        """
        total_waste = 0.0
        for route in self.routes:
            for node in route:
                total_waste += self.wastes.get(node, 0.0)
        return total_waste * self.R

    @property
    def cost(self) -> Optional[float]:
        """
        Get solution cost.

        Returns:
            float: Solution cost.
        """
        return self._cost

    @property
    def revenue(self) -> Optional[float]:
        """
        Get solution revenue.

        Returns:
            float: Solution revenue.
        """
        return self._revenue

    @property
    def profit(self) -> Optional[float]:
        """
        Get solution profit.

        Returns:
            float: Solution profit.
        """
        return self._profit

    def is_feasible(self) -> bool:
        """
        Check if solution respects capacity constraints.

        Returns:
            bool: True if solution is feasible, False otherwise.
        """
        for route in self.routes:
            route_waste = sum(self.wastes.get(node, 0.0) for node in route)
            if route_waste > self.capacity:
                return False
        return True

    def get_total_nodes(self) -> int:
        """
        Get total number of nodes in solution.

        Returns:
            int: Total number of nodes.
        """
        return sum(len(route) for route in self.routes)

    def copy(self) -> "Solution":
        """
        Create a deep copy of the solution.

        Returns:
            Solution: Deep copy of the solution.
        """
        routes_copy = [list(route) for route in self.routes]
        return Solution(
            routes_copy,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            self.R,
            self.C,
        )

    def __repr__(self) -> str:
        """
        String representation.

        Returns:
            str: String representation.
        """
        return (
            f"Solution(routes={len(self.routes)}, "
            f"nodes={self.get_total_nodes()}, "
            f"profit={self.profit:.2f}, "
            f"cost={self.cost:.2f}, "
            f"revenue={self.revenue:.2f})"
        )
