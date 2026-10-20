"""Low-level heuristic (LLH) pool for the Reactive Tabu Search solver.

Moved unchanged from :mod:`solver`: the mixin holding the seven
ruin-and-recreate operators (random/worst/cluster/shaw/string removal
paired with greedy or regret-2 insertion) that make up the RTS LLH pool.
"""

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


class RTSLLHOperatorsMixin:
    """LLH ruin-and-recreate operators for the RTS solver."""

    # ------------------------------------------------------------------
    # LLH pool
    # ------------------------------------------------------------------

    def _llh0(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """
        Random removal followed by greedy insertion.
        Uses random_removal to remove a sequence of nodes and greedy_profit_insertion
        to insert them back.

        Args:
            routes: The routes to evaluate.
            n: The number of nodes to remove.

        Returns:
            List[List[int]]: The routes evaluated.
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
        """
        Worst profit removal followed by regret-2 insertion.
        Uses worst_profit_removal to remove a sequence of nodes and regret_2_profit_insertion
        to insert them back.

        Args:
            routes: The routes to evaluate.
            n: The number of nodes to remove.

        Returns:
            List[List[int]]: The routes evaluated.
        """
        if self.params.profit_aware_operators:
            partial, removed = worst_profit_removal(routes, n, self.dist_matrix, self.wastes, self.R, self.C)
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
        """
        Cluster removal followed by greedy insertion.
        Uses cluster_removal to remove a sequence of nodes and greedy_profit_insertion
        to insert them back.

        Args:
            routes: The routes to evaluate.
            n: The number of nodes to remove.

        Returns:
            List[List[int]]: The routes evaluated.
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
        """
        Worst profit removal followed by greedy insertion.
        Uses worst_profit_removal to remove a sequence of nodes and greedy_profit_insertion
        to insert them back.

        Args:
            routes: The routes to evaluate.
            n: The number of nodes to remove.

        Returns:
            List[List[int]]: The routes evaluated.
        """
        if self.params.profit_aware_operators:
            partial, removed = worst_profit_removal(routes, n, self.dist_matrix, self.wastes, self.R, self.C)
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
        """
        Random removal followed by regret-2 insertion.
        Uses random_removal to remove a sequence of nodes and regret_2_profit_insertion
        to insert them back.

        Args:
            routes: The routes to evaluate.
            n: The number of nodes to remove.

        Returns:
            List[List[int]]: The routes evaluated.
        """
        partial, removed = random_removal(routes, n, self.random)
        if self.params.profit_aware_operators:
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
        """
        Shaw removal followed by regret-2 insertion.
        Uses shaw_removal to remove a sequence of nodes and regret_2_profit_insertion
        to insert them back.

        Args:
            routes: The routes to evaluate.
            n: The number of nodes to remove.

        Returns:
            List[List[int]]: The routes evaluated.
        """
        if self.params.profit_aware_operators:
            partial, removed = shaw_profit_removal(
                routes, n, self.dist_matrix, self.wastes, self.R, self.C, rng=self.random
            )
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
            partial, removed = shaw_removal(routes, n, self.dist_matrix, wastes=self.wastes)
            return regret_2_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=current_expand_pool(),
            )

    def _llh6(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """
        String removal followed by greedy insertion.
        Uses string_removal to remove a sequence of nodes and greedy_profit_insertion
        to insert them back.

        Args:
            routes: The routes to evaluate.
            n: The number of nodes to remove.

        Returns:
            List[List[int]]: The routes evaluated.
        """
        partial, removed = string_removal(routes, n, self.dist_matrix, rng=self.random)
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
