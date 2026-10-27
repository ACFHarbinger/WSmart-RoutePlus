"""Low-level heuristic moves for the HMM-GD hyper-heuristic, mixed into HMMGDHHSolver as HMMLowLevelMixin."""

import random
from typing import Dict, List

import numpy as np

from logic.src.configs.policies.routing.hyper_heuristics.hmm_gd_hh import HMMGDHHConfig
from logic.src.envs.common.expand_pool import current_expand_pool
from logic.src.policies.helpers.operators import (
    cluster_removal,
    greedy_insertion,
    greedy_profit_insertion,
    random_removal,
    regret_2_insertion,
    regret_2_profit_insertion,
    shaw_profit_removal,
    shaw_removal,
    string_removal,
    worst_profit_removal,
    worst_removal,
)


class HMMLowLevelMixin:
    # Provided by HMMGDHHSolver.__init__; annotations only.
    params: HMMGDHHConfig
    dist_matrix: np.ndarray
    wastes: Dict[int, float]
    capacity: float
    R: float
    C: float
    mandatory_nodes: List[int]
    nodes: List[int]
    random: random.Random

    # ------------------------------------------------------------------
    # Forward Algorithm helpers (REMOVED: Old Gaussian PDF logic)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # LLH pool
    # ------------------------------------------------------------------

    def _llh0(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L0: random_removal + greedy_insertion.

        Args:
            routes: List of routes.
            n: Number of nodes to remove.

        Returns:
            List[List[int]]: List of routes.
        """
        use_profit = self.params.profit_aware_operators
        expand_pool = current_expand_pool()

        p, r = random_removal(routes, n, self.random)
        if use_profit:
            return greedy_profit_insertion(
                p, r, self.dist_matrix, self.wastes, self.capacity, self.R, self.C, expand_pool=expand_pool
            )
        return greedy_insertion(
            p,
            r,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
        )

    def _llh1(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L1: worst_removal + regret_2_insertion.

        Args:
            routes: List of routes.
            n: Number of nodes to remove.

        Returns:
            List[List[int]]: List of routes.
        """
        use_profit = self.params.profit_aware_operators
        expand_pool = current_expand_pool()

        if use_profit:
            p, r = worst_profit_removal(routes, n, self.dist_matrix, self.wastes, self.R, self.C)
            return regret_2_profit_insertion(
                p, r, self.dist_matrix, self.wastes, self.capacity, self.R, self.C, expand_pool=expand_pool
            )
        p, r = worst_removal(routes, n, self.dist_matrix)
        return regret_2_insertion(
            p,
            r,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
        )

    def _llh2(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L2: cluster_removal + greedy_insertion.

        Args:
            routes: List of routes.
            n: Number of nodes to remove.

        Returns:
            List[List[int]]: List of routes.
        """
        use_profit = self.params.profit_aware_operators
        expand_pool = current_expand_pool()

        p, r = cluster_removal(routes, n, self.dist_matrix, self.nodes)
        if use_profit:
            return greedy_profit_insertion(
                p, r, self.dist_matrix, self.wastes, self.capacity, self.R, self.C, expand_pool=expand_pool
            )
        return greedy_insertion(
            p,
            r,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
        )

    def _llh3(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L3: worst_removal + greedy_insertion.

        Args:
            routes: List of routes.
            n: Number of nodes to remove.

        Returns:
            List[List[int]]: List of routes.
        """
        use_profit = self.params.profit_aware_operators
        expand_pool = current_expand_pool()

        if use_profit:
            p, r = worst_profit_removal(routes, n, self.dist_matrix, self.wastes, self.R, self.C)
            return greedy_profit_insertion(
                p, r, self.dist_matrix, self.wastes, self.capacity, self.R, self.C, expand_pool=expand_pool
            )
        p, r = worst_removal(routes, n, self.dist_matrix)
        return greedy_insertion(
            p,
            r,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
        )

    def _llh4(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L4: random_removal + regret_2_insertion.

        Args:
            routes: List of routes.
            n: Number of nodes to remove.

        Returns:
            List[List[int]]: List of routes.
        """
        use_profit = self.params.profit_aware_operators
        expand_pool = current_expand_pool()

        p, r = random_removal(routes, n, self.random)
        if use_profit:
            return regret_2_profit_insertion(
                p, r, self.dist_matrix, self.wastes, self.capacity, self.R, self.C, expand_pool=expand_pool
            )
        return regret_2_insertion(
            p,
            r,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
        )

    def _llh5(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L5: shaw_removal + greedy_insertion.

        Args:
            routes: List of routes.
            n: Number of nodes to remove.

        Returns:
            List[List[int]]: List of routes.
        """
        use_profit = self.params.profit_aware_operators
        expand_pool = current_expand_pool()

        if use_profit:
            p, r = shaw_profit_removal(routes, n, self.dist_matrix, self.wastes, self.R, self.C, rng=self.random)
            return greedy_profit_insertion(
                p, r, self.dist_matrix, self.wastes, self.capacity, self.R, self.C, expand_pool=expand_pool
            )
        p, r = shaw_removal(routes, n, self.dist_matrix)
        return greedy_insertion(
            p,
            r,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
        )

    def _llh6(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L6: string_removal + regret_2_insertion.

        Args:
            routes: List of routes.
            n: Number of nodes to remove.

        Returns:
            List[List[int]]: List of routes.
        """
        use_profit = self.params.profit_aware_operators
        expand_pool = current_expand_pool()

        p, r = string_removal(routes, n, self.dist_matrix, rng=self.random)
        if use_profit:
            return regret_2_profit_insertion(
                p, r, self.dist_matrix, self.wastes, self.capacity, self.R, self.C, expand_pool=expand_pool
            )
        return regret_2_insertion(
            p,
            r,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
        )
