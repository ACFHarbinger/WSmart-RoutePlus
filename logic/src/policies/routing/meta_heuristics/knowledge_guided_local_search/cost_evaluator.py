"""
Cost evaluation utilities for Knowledge-Guided Local Search (KGLS).

Attributes:
    CostEvaluator: Evaluates and penalizes edges based on geometric criteria.

Example:
    >>> evaluator = CostEvaluator(dist_matrix)
"""

import math
from typing import Dict, List, Tuple

import numpy as np


class CostEvaluator:
    """
    KGLS Cost Evaluator.

    Maintains the original distance matrix and an actively perturbed distance matrix.
    Computes geometric properties of routes (length, width) to identify 'bad' edges
    and applies increasing penalty scalars to explicitly guide the local search.

    Attributes:
        dist_matrix: NxN array representing true edge distances.
        penalized_dist_matrix: Distance matrix with active penalties.
        n_nodes: Number of non-depot nodes.
        nodes: List of node indices.
        penalization_enabled: Whether to return penalized costs.
        edge_penalties: Map of (u, v) -> number of times penalized.
        baseline_cost: Average distance used for penalty scaling.
    """

    def __init__(self, dist_matrix: np.ndarray):
        """Initialize the Cost Evaluator.

        Args:
            dist_matrix: NxN array representing edge distances. First index is depot (0).

        Returns:
            None.
        """
        self.dist_matrix = dist_matrix
        self.penalized_dist_matrix = np.copy(dist_matrix)

        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))

        # Track active stringency
        self.penalization_enabled = False

        # Map of (u, v) -> penalty count
        self.edge_penalties: Dict[Tuple[int, int], int] = {}

        # Baseline cost (average distance)
        self.baseline_cost = self._compute_baseline_cost()

    def _compute_baseline_cost(self) -> float:
        """Compute the average distance between all nodes excluding the depot.

        Args:
            None.

        Returns:
            float: Average node-to-node distance.
        """
        total = 0.0
        count = 0
        for i in self.nodes:
            for j in self.nodes:
                if i != j:
                    total += self.dist_matrix[i][j]
                    count += 1
        return total / count if count > 0 else 0.0

    def enable_penalization(self):
        """Enable returning penalized distances in the active matrix.

        Args:
            None.

        Returns:
            None.
        """
        self.penalization_enabled = True

    def disable_penalization(self):
        """Disable returning penalized distances.

        Args:
            None.

        Returns:
            None.
        """
        self.penalization_enabled = False

    def reset_penalties(self):
        """Clear all active edge penalties and reset the perturbed matrix.

        Args:
            None.

        Returns:
            None.
        """
        self.edge_penalties.clear()
        self.penalized_dist_matrix = np.copy(self.dist_matrix)

    def get_distance_matrix(self) -> np.ndarray:
        """Get the active distance matrix based on penalization state.

        Args:
            None.

        Returns:
            np.ndarray: The distance matrix to use (original or penalized).
        """
        if self.penalization_enabled:
            return self.penalized_dist_matrix
        return self.dist_matrix

    def _compute_route_center(self, route: List[int], locations: np.ndarray) -> Tuple[float, float]:
        """Compute theoretical center coordinates of a given route.

        Args:
            route: List of node indices in the route.
            locations: Coordinate matrix for all nodes.

        Returns:
            Tuple[float, float]: Mean X and Y coordinates.
        """
        if not route:
            return 0.0, 0.0

        sum_x = sum(locations[n][0] for n in route)
        sum_y = sum(locations[n][1] for n in route)
        return sum_x / len(route), sum_y / len(route)

    def _compute_edge_width(self, u: int, v: int, cx: float, cy: float, locations: np.ndarray) -> float:
        """Compute edge width relative to the depot and route center.

        Based on Arnold & Sorensen (2019), Equation (2).
        The width of an edge (u, v) is defined as the distance between the
        projections of nodes u and v onto the axis perpendicular to the
        depot-to-center line.

        Args:
            u: Source node index.
            v: Target node index.
            cx: Route center X coordinate.
            cy: Route center Y coordinate.
            locations: Coordinate matrix for all nodes.

        Returns:
            float: Geometric width of the edge.
        """
        depot_x, depot_y = locations[0]
        ux, uy = locations[u]
        vx, vy = locations[v]

        # Vector from depot to center
        dx, dy = cx - depot_x, cy - depot_y
        dist_depot_center = math.hypot(dx, dy)

        if dist_depot_center < 1e-6:
            return 0.0

        # Normal vector perpendicular to (dx, dy)
        nx, ny = -dy / dist_depot_center, dx / dist_depot_center

        # Project node vectors (relative to depot) onto the normal vector
        proj_u = (ux - depot_x) * nx + (uy - depot_y) * ny
        proj_v = (vx - depot_x) * nx + (vy - depot_y) * ny

        return abs(proj_u - proj_v)

    def evaluate_and_penalize_edges(
        self, routes: List[List[int]], locations: np.ndarray, criterium: str, num_perturbations: int
    ) -> List[int]:
        """Evaluate and penalize edges based on specified criterion.

        Identifies 'bad' edges in the solution using geometric metrics and
        updates the penalized distance matrix to guide local search away from them.

        Args:
            routes: Current route solution structure.
            locations: Coordinates for width calculations.
            criterium: Geometric metric to use ("length", "width", "width_length").
            num_perturbations: Number of top worst edges to penalize.

        Returns:
            List[int]: Unique node IDs attached to the penalized edges.
        """
        edge_scores = []

        for route in routes:
            if not route:
                continue

            cx, cy = 0.0, 0.0
            if "width" in criterium:
                cx, cy = self._compute_route_center(route, locations)

            # Build full edge tour including depot
            full_tour = [0] + route + [0]
            for i in range(len(full_tour) - 1):
                u = full_tour[i]
                v = full_tour[i + 1]

                # Compute raw metric
                val = 0.0
                if criterium == "length":
                    val = self.dist_matrix[u][v]
                elif criterium == "width":
                    val = self._compute_edge_width(u, v, cx, cy, locations)
                elif criterium == "width_length":
                    val = self.dist_matrix[u][v] + self._compute_edge_width(u, v, cx, cy, locations)

                # Normalize by current penalty
                penalty_count = self.edge_penalties.get((u, v), 0)
                score = val / (1.0 + penalty_count)

                edge_scores.append((score, u, v))

        # Sort descending to find the worst edges
        edge_scores.sort(key=lambda x: x[0], reverse=True)

        affected_nodes = set()
        for i in range(min(num_perturbations, len(edge_scores))):
            _, u, v = edge_scores[i]

            # Increment penalty
            count_uv = self.edge_penalties.get((u, v), 0) + 1
            self.edge_penalties[(u, v)] = count_uv
            self.edge_penalties[(v, u)] = count_uv

            # Update penalized matrix: true_cost + 0.1 * baseline * penalty_count
            alpha = 0.1
            penalty_term = alpha * self.baseline_cost * count_uv
            self.penalized_dist_matrix[u][v] = self.dist_matrix[u][v] + penalty_term
            self.penalized_dist_matrix[v][u] = self.dist_matrix[v][u] + penalty_term

            if u != 0:
                affected_nodes.add(u)
            if v != 0:
                affected_nodes.add(v)

        return list(affected_nodes)
