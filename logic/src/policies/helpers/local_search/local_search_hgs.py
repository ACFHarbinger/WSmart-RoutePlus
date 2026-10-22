"""
HGS Local Search Module.

This module provides the local search implementation specifically for
Hybrid Genetic Search (HGS). It applies operators to HGS 'Individual' objects.

Attributes:
    HGSLocalSearch: HGS-specific local search implementation.

Example:
    >>> from logic.src.policies.helpers.local_search.local_search_hgs import HGSLocalSearch
    >>> ls = HGSLocalSearch(dist_matrix, waste, capacity, R, C, params)
    >>> improved_ind = ls.optimize(individual)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from .local_search_base import LocalSearch

if TYPE_CHECKING:
    from logic.src.configs.policies.routing.meta_heuristics.hgs import HGSConfig
    from logic.src.policies.routing.meta_heuristics.hybrid_genetic_search.individual import Individual


class HGSLocalSearch(LocalSearch):
    """
    Local Search module for HGS.

    Attributes:
        None
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        waste: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        mandatory_nodes: Optional[List[int]],
        params: "HGSConfig",
    ):
        """
        Initialize HGS Local Search.

        Args:
            dist_matrix: Distance matrix.
            waste: Dictionary of waste amounts for each node.
            capacity: Vehicle capacity.
            R: Parameter R.
            C: Parameter C.
            params: HGS parameters.
        """
        super().__init__(dist_matrix, waste, capacity, R, C, params, mandatory_nodes)

    def optimize(self, solution: "Individual") -> "Individual":
        """
        Iteratively improve an individual using local search operators.
        Reconstructs the genotype by proportionally interleaving unvisited nodes
        to prevent evolutionary degradation.

        Args:
            solution: The individual to optimize.

        Returns:
            The optimized individual.
        """
        if not solution.routes:
            return solution

        self.routes = [r[:] for r in solution.routes]

        # Run optimization with current profit context
        self._optimize_internal(initial_profit=solution.profit_score)

        solution.routes = self.routes
        solution.profit_score = self.current_profit

        # Flatten active routes into a single continuous sequence
        route_positions = {node: i for i, node in enumerate(solution.giant_tour)}
        self.routes.sort(key=lambda r: route_positions.get(r[0], float("inf")) if r else float("inf"))
        active_seq = [node for r in self.routes for node in r]
        visited_set = set(active_seq)

        # Rigorous Genotype Reconstruction:
        # We iterate through the *original* giant tour structure.
        # This perfectly preserves the spatial entropy of unvisited nodes while
        # maintaining the continuous sequential chunks of the optimized routes.
        new_gt = []
        active_idx = 0
        for orig_node in solution.giant_tour:
            if orig_node in visited_set:
                # Slot originally held an active node; place the next optimized active node
                if active_idx < len(active_seq):
                    new_gt.append(active_seq[active_idx])
                    active_idx += 1
            else:
                # Slot originally held an unvisited node; preserve its relative position
                new_gt.append(orig_node)

        # Safety fallback (invariant safeguard)
        while active_idx < len(active_seq):
            new_gt.append(active_seq[active_idx])
            active_idx += 1

        solution.giant_tour = new_gt
        return solution
