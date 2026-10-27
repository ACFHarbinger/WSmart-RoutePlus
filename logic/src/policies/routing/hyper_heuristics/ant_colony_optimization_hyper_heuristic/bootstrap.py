"""Bootstrap phase of the ACO hyper-heuristic: the initial pheromone matrix from sampled operator sequences.

Attributes:
    ACOBootstrapMixin: Standard and profit-aware bootstrap of ``HyperHeuristicACO``.

Example:
    >>> class HyperHeuristicACO(ACOBootstrapMixin, HyperHeuristicSearch): ...
"""

import random
from typing import Dict, List

import numpy as np

from logic.src.configs.policies.routing.hyper_heuristics.aco_hh import HyperHeuristicACOConfig
from logic.src.policies.helpers.operators import (
    bb_insertion,
    bb_profit_insertion,
    deep_insertion,
    deep_profit_insertion,
    farthest_insertion,
    farthest_profit_insertion,
    geni_insertion,
    geni_profit_insertion,
    greedy_insertion,
    greedy_insertion_with_blinks,
    greedy_profit_insertion,
    greedy_profit_insertion_with_blinks,
    nearest_insertion,
    nearest_profit_insertion,
    regret_2_insertion,
    regret_2_profit_insertion,
    regret_3_insertion,
    regret_3_profit_insertion,
    regret_4_insertion,
    regret_4_profit_insertion,
    savings_insertion,
    savings_profit_insertion,
)


class ACOBootstrapMixin:
    """Standard and profit-aware bootstrap of :class:`HyperHeuristicACO`."""

    # Provided by HyperHeuristicACO.__init__; annotations only.
    cfg: HyperHeuristicACOConfig
    random: random.Random
    dist_matrix: np.ndarray
    wastes: Dict[int, float]
    capacity: float
    R: float
    C: float

    def _bootstrap(
        self,
        nodes: List[int],
        effective_mandatory: List[int],
    ) -> List[List[int]]:
        """Select and apply one bootstrap insertion operator uniformly at random.

        Chooses a heuristic from the full ``recreate_repair`` portfolio with
        equal probability (pure exploration) and applies it to an empty route
        list with ``nodes`` as the unvisited pool.

        When ``cfg.profit_aware_operators`` is ``True``, the profit-aware
        variant of the chosen operator is used; otherwise the cost-minimising
        variant is used.

        Args:
            nodes: Customer node indices to bootstrap (excluding depot 0).
            effective_mandatory: Mandatory node indices that must be served.

        Returns:
            List of routes produced by the selected bootstrap heuristic.
        """
        if self.cfg.profit_aware_operators:
            return self._bootstrap_profit(nodes, effective_mandatory)
        return self._bootstrap_standard(nodes, effective_mandatory)

    def _bootstrap_standard(self, nodes: List[int], mandatory: List[int]) -> List[List[int]]:
        """Apply a uniformly-random cost-minimising insertion heuristic.

        Candidates: greedy, greedy-with-blinks, nearest, farthest, savings,
        regret-2, regret-3, regret-4, deep, GENI, branch-and-bound LDS.

        Args:
            nodes: Customer nodes to insert.
            mandatory: Mandatory node indices.

        Returns:
            List of routes.
        """
        choice = self.random.randrange(11)  # 11 standard operators

        dm = self.dist_matrix
        ws = self.wastes
        cap = self.capacity
        man = mandatory

        if choice == 0:
            return greedy_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 1:
            return greedy_insertion_with_blinks(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                mandatory_nodes=man,
                rng=self.random,
                expand_pool=False,
            )
        if choice == 2:
            return nearest_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 3:
            return farthest_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 4:
            return savings_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 5:
            return regret_2_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 6:
            return regret_3_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 7:
            return regret_4_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 8:
            return deep_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 9:
            return geni_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                mandatory_nodes=man,
                rng=self.random,
                expand_pool=False,
            )
        # choice == 10
        return bb_insertion(
            routes=[],
            removed_nodes=nodes,
            dist_matrix=dm,
            wastes=ws,
            capacity=cap,
            max_discrepancy=0,
            mandatory_nodes=man,
            expand_pool=False,
        )

    def _bootstrap_profit(self, nodes: List[int], mandatory: List[int]) -> List[List[int]]:
        """Apply a uniformly-random profit-aware insertion heuristic.

        Candidates: greedy_profit, blinks_profit, nearest_profit,
        farthest_profit, savings_profit, regret-2_profit, regret-3_profit,
        regret-4_profit, deep_profit, GENI_profit, bb_profit.

        Args:
            nodes: Customer nodes to insert.
            mandatory: Mandatory node indices.

        Returns:
            List of routes.
        """
        choice = self.random.randrange(11)  # 11 profit-aware operators

        dm = self.dist_matrix
        ws = self.wastes
        cap = self.capacity
        R = self.R
        C = self.C
        man = mandatory

        if choice == 0:
            return greedy_profit_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                R=R,
                C=C,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 1:
            return greedy_profit_insertion_with_blinks(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                R=R,
                C=C,
                mandatory_nodes=man,
                rng=self.random,
                expand_pool=False,
            )
        if choice == 2:
            return nearest_profit_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                R=R,
                C=C,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 3:
            return farthest_profit_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                R=R,
                C=C,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 4:
            return savings_profit_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                R=R,
                C=C,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 5:
            return regret_2_profit_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                R=R,
                C=C,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 6:
            return regret_3_profit_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                R=R,
                C=C,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 7:
            return regret_4_profit_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                R=R,
                C=C,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 8:
            return deep_profit_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                R=R,
                C=C,
                mandatory_nodes=man,
                expand_pool=False,
            )
        if choice == 9:
            return geni_profit_insertion(
                routes=[],
                removed_nodes=nodes,
                dist_matrix=dm,
                wastes=ws,
                capacity=cap,
                R=R,
                C=C,
                mandatory_nodes=man,
                rng=self.random,
                expand_pool=False,
            )
        # choice == 10
        return bb_profit_insertion(
            routes=[],
            removed_nodes=nodes,
            dist_matrix=dm,
            wastes=ws,
            capacity=cap,
            R=R,
            C=C,
            max_discrepancy=0,
            mandatory_nodes=man,
            expand_pool=False,
        )
