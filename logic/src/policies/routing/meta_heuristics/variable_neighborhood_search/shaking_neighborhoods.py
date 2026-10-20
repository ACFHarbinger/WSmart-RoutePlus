"""Shaking neighborhoods for the Variable Neighborhood Search solver.

Moved unchanged from :mod:`solver`: the mixin holding the five shaking
structures N_1 ... N_5 (random, worst, cluster and regret-2
ruin-and-recreate moves of increasing severity) that VNS perturbs with
before its local search descent.
"""

from typing import List

from logic.src.pipeline.learning.envs.common.expand_pool import current_expand_pool
from logic.src.policies.helpers.operators.search_heuristics.destroy_repair_llh import (
    llh_cluster_greedy,
    llh_random_greedy,
    llh_random_regret_2,
    llh_worst_regret_2,
)


class VNSShakingMixin:
    """Shaking neighborhoods N_1 ... N_5 for the VNS solver."""

    # ------------------------------------------------------------------
    # Shaking neighborhoods (N_1 ... N_5, increasing severity)
    # ------------------------------------------------------------------

    def _shake_n1(self, routes: List[List[int]]) -> List[List[int]]:
        """
        N_1: Remove 1 node randomly, greedy reinsert.

        Args:
            routes: List of routes.
            n: Number of routes.

        Returns:
            List[List[int]]: List of routes.
        """
        use_profit = self.params.profit_aware_operators
        expand_pool = current_expand_pool()

        return llh_random_greedy(
            routes,
            1,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            self.R,
            self.C,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
            profit_aware=use_profit,
            rng=self.random,
        )

    def _shake_n2(self, routes: List[List[int]]) -> List[List[int]]:
        """
        N_2: Remove 2 nodes randomly, greedy reinsert.

        Args:
            routes: List of routes.
            n: Number of routes.

        Returns:
            List[List[int]]: List of routes.
        """
        use_profit = self.params.profit_aware_operators
        expand_pool = current_expand_pool()

        return llh_random_greedy(
            routes,
            2,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            self.R,
            self.C,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
            profit_aware=use_profit,
            rng=self.random,
        )

    def _shake_n3(self, routes: List[List[int]]) -> List[List[int]]:
        """
        N_3: Worst removal of 2 nodes, regret-2 reinsert.

        Args:
            routes: List of routes.
            n: Number of routes.

        Returns:
            List[List[int]]: List of routes.
        """
        use_profit = self.params.profit_aware_operators
        expand_pool = current_expand_pool()

        return llh_worst_regret_2(
            routes,
            2,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            self.R,
            self.C,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
            profit_aware=use_profit,
            rng=self.random,
        )

    def _shake_n4(self, routes: List[List[int]]) -> List[List[int]]:
        """
        N_4: Cluster removal of 3 nodes, greedy reinsert.

        Args:
            routes: List of routes.
            n: Number of routes.

        Returns:
            List[List[int]]: List of routes.
        """
        use_profit = self.params.profit_aware_operators
        expand_pool = current_expand_pool()

        return llh_cluster_greedy(
            routes,
            3,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            self.R,
            self.C,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
            profit_aware=use_profit,
            rng=self.random,
            nodes=self.nodes,
        )

    def _shake_n5(self, routes: List[List[int]]) -> List[List[int]]:
        """
        N_5: Remove 3 nodes randomly, regret-2 reinsert.

        Args:
            routes: List of routes.
            n: Number of routes.

        Returns:
            List[List[int]]: List of routes.
        """
        use_profit = self.params.profit_aware_operators
        expand_pool = current_expand_pool()

        return llh_random_regret_2(
            routes,
            3,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            self.R,
            self.C,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
            profit_aware=use_profit,
            rng=self.random,
        )
