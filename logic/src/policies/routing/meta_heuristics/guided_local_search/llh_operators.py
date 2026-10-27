"""Low-level heuristic (LLH) pool for the Guided Local Search solver.

Moved unchanged from :mod:`solver`: the mixin holding the six
ruin-and-recreate operators (random/worst/cluster/penalized removal paired
with greedy or regret-2 insertion) that GLS selects from in its inner loop.
"""

from __future__ import annotations

import random
from typing import Dict, List

import numpy as np

from logic.src.configs.policies.routing.meta_heuristics.gls import GLSConfig
from logic.src.envs.common.expand_pool import current_expand_pool
from logic.src.policies.helpers.operators import (
    cluster_removal,
    greedy_insertion,
    greedy_profit_insertion,
    penalized_removal,
    random_removal,
    regret_2_insertion,
    regret_2_profit_insertion,
    worst_removal,
)


class GLSLLHOperatorsMixin:
    """LLH ruin-and-recreate operators for the G-LNS solver."""

    # Provided by GLSSolver.__init__; annotations only.
    dist_matrix: np.ndarray
    wastes: Dict[int, float]
    capacity: float
    R: float
    C: float
    params: GLSConfig
    mandatory_nodes: List[int]
    nodes: List[int]
    random: random.Random
    penalties: np.ndarray

    def _llh0(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """Random removal + Greedy insertion.

        Args:
            routes: Current routes.
            n: Number of nodes to remove.

        Returns:
            Repaired routes.
        """
        if self.params.profit_aware_operators:
            partial, removed = random_removal(routes, n, self.random)
            return greedy_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                current_expand_pool(),
            )
        else:
            partial, removed = random_removal(routes, n, self.random)
            return greedy_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=current_expand_pool(),
            )

    def _llh1(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """Worst removal + Regret-2 insertion.

        Args:
            routes: Current routes.
            n: Number of nodes to remove.

        Returns:
            Repaired routes.
        """
        if self.params.profit_aware_operators:
            partial, removed = worst_removal(routes, n, self.dist_matrix)
            return regret_2_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                current_expand_pool(),
            )
        else:
            partial, removed = worst_removal(routes, n, self.dist_matrix)
            return regret_2_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=current_expand_pool(),
            )

    def _llh2(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """Cluster removal + Greedy insertion.

        Args:
            routes: Current routes.
            n: Number of nodes to remove.

        Returns:
            Repaired routes.
        """
        if self.params.profit_aware_operators:
            partial, removed = cluster_removal(routes, n, self.dist_matrix, self.nodes)
            return greedy_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                current_expand_pool(),
            )
        else:
            partial, removed = cluster_removal(routes, n, self.dist_matrix, self.nodes)
            return greedy_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=current_expand_pool(),
            )

    def _llh3(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """Worst removal + Greedy insertion.

        Args:
            routes: Current routes.
            n: Number of nodes to remove.

        Returns:
            Repaired routes.
        """
        if self.params.profit_aware_operators:
            partial, removed = worst_removal(routes, n, self.dist_matrix)
            return greedy_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                current_expand_pool(),
            )
        else:
            partial, removed = worst_removal(routes, n, self.dist_matrix)
            return greedy_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=current_expand_pool(),
            )

    def _llh4(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """Random removal + Regret-2 insertion.

        Args:
            routes: Current routes.
            n: Number of nodes to remove.

        Returns:
            Repaired routes.
        """
        if self.params.profit_aware_operators:
            partial, removed = random_removal(routes, n, self.random)
            return regret_2_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                current_expand_pool(),
            )
        else:
            partial, removed = random_removal(routes, n, self.random)
            return regret_2_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=current_expand_pool(),
            )

    def _llh5(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """Penalized removal + Greedy insertion.

        Args:
            routes: Current routes.
            n: Number of nodes to remove.

        Returns:
            Repaired routes.
        """
        """Penalized Removal (Targeted Destruction): The G-LNS equivalent of FLS."""
        partial, removed = penalized_removal(routes, n, self.penalties, rng=self.random)  # type: ignore[arg-type]
        if self.params.profit_aware_operators:
            return greedy_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                current_expand_pool(),
            )
        else:
            return greedy_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=current_expand_pool(),
            )
