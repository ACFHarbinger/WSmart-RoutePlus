"""ALNS Perturbation Context Module.

This module defines the context object used by perturbation and destruction
heuristics in the ALNS (Adaptive Large Neighborhood Search) framework.

Attributes:
    ALNSPerturbationContext: Context object containing routes, distances, and wastes.

Example:
    >>> context = ALNSPerturbationContext(routes, dist_matrix, wastes, capacity)
    >>> route_load = context._get_load_cached(0)
"""

import copy
from typing import Dict, List

import numpy as np


class ALNSPerturbationContext:
    """
    Context object required by perturbation operators.

    This class serves as a state-carrier for the various perturbation
    and destruction heuristics, providing a unified interface for
    route data, waste demands, and distance matrices.

    Attributes:
        routes: List of current solution routes.
        dist_matrix: Problem distance matrix.
        d: Alias for dist_matrix.
        waste: Dictionary of node waste demands.
        Q: Vehicle capacity.
        node_map: Mapping from node ID to (route_index, position).
    """

    def __init__(self, routes: List[List[int]], dist_matrix: np.ndarray, wastes: Dict[int, float], capacity: float):
        """
        Initialize the perturbation context.

        Args:
            routes: Current solution routes.
            dist_matrix: NxN distance matrix.
            wastes: Dictionary of node waste demands.
            capacity: Maximum vehicle capacity.
        """
        self.routes = copy.deepcopy(routes)
        self.dist_matrix = dist_matrix
        self.d = dist_matrix
        self.waste = wastes
        self.Q = capacity
        self._build_structures()

    def _build_structures(self) -> None:
        """Construct auxiliary structures like node_map for efficient lookup."""
        self.node_map = {}
        for r_idx, route in enumerate(self.routes):
            for pos, node in enumerate(route):
                self.node_map[node] = (r_idx, pos)

    def _update_map(self, changed_routes: set) -> None:
        """
        Rebuild internal maps after modifications.

        Args:
            changed_routes: Indices of routes that were modified (for efficiency).
        """
        self._build_structures()

    def _get_load_cached(self, ri: int) -> float:
        """
        Calculate total waste load of a specific route.

        Args:
            ri: Index of the route.

        Returns:
            float: Sum of waste for all nodes in the route.
        """
        return sum(self.waste.get(n, 0) for n in self.routes[ri])
