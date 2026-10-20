"""Low-level heuristic moves for the sequence-based selection hyper-heuristic, mixed into SSHHSolver as SSHHLowLevelMixin."""

from typing import List

from logic.src.pipeline.learning.envs.common.expand_pool import current_expand_pool
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


class SSHHLowLevelMixin:
    # ------------------------------------------------------------------
    # LLH pool
    # ------------------------------------------------------------------

    def _llh0(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """
        L0: random_removal + greedy_insertion.

        Args:
            routes: List of routes.
            n: Number of nodes.

        Returns:
            List[List[int]]: List of routes after applying the heuristic.
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
        """
        L1: worst_removal + regret_2_insertion.

        Args:
            routes: List of routes.
            n: Number of nodes.

        Returns:
            List[List[int]]: List of routes after applying the heuristic.
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
        """
        L2: cluster_removal + greedy_insertion.

        Args:
            routes: List of routes.
            n: Number of nodes.

        Returns:
            List[List[int]]: List of routes after applying the heuristic.
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
        """
        L3: worst_removal + greedy_insertion.

        Args:
            routes: List of routes.
            n: Number of nodes.

        Returns:
            List[List[int]]: List of routes after applying the heuristic.
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
        """
        L4: random_removal + regret_2_insertion.

        Args:
            routes: List of routes.
            n: Number of nodes.

        Returns:
            List[List[int]]: List of routes after applying the heuristic.
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
        """
        L5: shaw_removal + greedy_insertion.

        Args:
            routes: List of routes.
            n: Number of nodes.

        Returns:
            List[List[int]]: List of routes after applying the heuristic.
        """
        use_profit = self.params.profit_aware_operators
        expand_pool = current_expand_pool()

        if use_profit:
            p, r = shaw_profit_removal(routes, n, self.dist_matrix, self.wastes, self.R, self.C)
            return greedy_profit_insertion(
                p, r, self.dist_matrix, self.wastes, self.capacity, self.R, self.C, expand_pool=expand_pool
            )
        else:
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
        """
        L6: string_removal + regret_2_insertion.

        Args:
            routes: List of routes.
            n: Number of nodes.

        Returns:
            List[List[int]]: List of routes after applying the heuristic.
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
