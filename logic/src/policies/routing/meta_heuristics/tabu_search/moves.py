"""Candidate moves and low-level heuristics of the tabu search.

Attributes:
    TSMoveGenerationMixin: Swap, relocate and 2-opt neighbourhoods plus the ruin-and-recreate LLHs of ``TSSolver``.

Example:
    >>> class TSSolver(TSMoveGenerationMixin, PolicyVizMixin): ...
"""

import copy
from typing import Any, List, Sequence, Tuple

from logic.src.envs.tasks.common.expand_pool import current_expand_pool
from logic.src.policies.helpers.operators.destroy_ruin import (
    cluster_removal,
    random_removal,
    worst_profit_removal,
    worst_removal,
)
from logic.src.policies.helpers.operators.intra_route_local_search.k_opt import move_2opt_intra
from logic.src.policies.helpers.operators.intra_route_local_search.relocate import move_relocate
from logic.src.policies.helpers.operators.intra_route_local_search.swap import move_swap
from logic.src.policies.helpers.operators.recreate_repair import (
    greedy_insertion,
    greedy_profit_insertion,
    regret_2_insertion,
    regret_2_profit_insertion,
)
from logic.src.policies.routing.meta_heuristics.tabu_search.ls_adapter import TSLSAdapter


class TSMoveGenerationMixin:
    """Swap, relocate and 2-opt candidate moves and the low-level heuristics of :class:`TSSolver`."""

    def _generate_candidates(self, routes: List[List[int]]) -> Sequence[Tuple[List[List[int]], Tuple[str, Any]]]:
        """
        Generate candidate neighborhood moves.

        Args:
            routes: Routes to evaluate.

        Returns:
            List[Tuple[List[List[int]], Tuple[str, Any]]]: List of new routes and move descriptors.
        """
        candidates: List[Tuple[List[List[int]], Tuple[str, Any]]] = []

        # Use candidate list if enabled
        max_candidates: float = self.params.candidate_list_size if self.params.candidate_list_enabled else float("inf")

        # Destroy-repair based neighborhoods
        if self.params.use_insertion:
            for _ in range(min(3, int(max_candidates))):
                llh_idx = self.random.randint(0, len(self._llh_pool) - 1)
                try:
                    new_routes = self._llh_pool[llh_idx](copy.deepcopy(routes), self.params.n_removal)
                    move_desc = ("destroy_repair", (llh_idx, self.iteration))
                    candidates.append((new_routes, move_desc))
                except Exception:
                    continue

        # Swap-based neighborhoods
        if self.params.use_swap and len(candidates) < max_candidates:
            candidates.extend(self._generate_swap_moves(routes, max_new=3))

        # Relocate-based neighborhoods
        if self.params.use_relocate and len(candidates) < max_candidates:
            candidates.extend(self._generate_relocate_moves(routes, max_new=3))

        # 2-opt neighborhoods
        if self.params.use_2opt and len(candidates) < max_candidates:
            candidates.extend(self._generate_2opt_moves(routes, max_new=2))

        # Shuffle to avoid bias
        self.random.shuffle(candidates)

        return candidates[: int(max_candidates)]

    def _generate_swap_moves(
        self, routes: List[List[int]], max_new: int = 5
    ) -> Sequence[Tuple[List[List[int]], Tuple[str, Any]]]:
        """
        Generate swap-based neighborhood moves.

        Args:
            routes: Routes to evaluate.
            max_new: Maximum number of new routes to generate.

        Returns:
            List[Tuple[List[List[int]], Tuple[str, Any]]]: List of new routes and move descriptors.
        """
        candidates: List[Tuple[List[List[int]], Tuple[str, Any]]] = []

        for _ in range(max_new):
            new_routes = copy.deepcopy(routes)

            # Try inter-route swap
            if len(new_routes) >= 2:
                try:
                    r1_idx = self.random.randint(0, len(new_routes) - 1)
                    r2_idx = self.random.randint(0, len(new_routes) - 1)
                    if r1_idx != r2_idx and new_routes[r1_idx] and new_routes[r2_idx]:
                        pos1 = self.random.randint(0, len(new_routes[r1_idx]) - 1)
                        pos2 = self.random.randint(0, len(new_routes[r2_idx]) - 1)
                        node1 = new_routes[r1_idx][pos1]
                        node2 = new_routes[r2_idx][pos2]

                        adapter = TSLSAdapter(new_routes, self.dist_matrix, self.wastes, self.capacity, -1e9)
                        move_swap(adapter, node1, node2, r1_idx, pos1, r2_idx, pos2)
                        new_routes = adapter.routes
                        move_desc = ("swap_inter", (node1, node2))
                        candidates.append((new_routes, move_desc))
                except Exception:
                    continue

            # Try intra-route swap
            if not candidates or self.random.random() < 0.5:
                try:
                    r_idx = self.random.randint(0, len(new_routes) - 1)
                    if len(new_routes[r_idx]) >= 2:
                        pos1 = self.random.randint(0, len(new_routes[r_idx]) - 1)
                        pos2 = self.random.randint(0, len(new_routes[r_idx]) - 1)
                        if pos1 != pos2:
                            node1 = new_routes[r_idx][pos1]
                            node2 = new_routes[r_idx][pos2]
                            adapter = TSLSAdapter(new_routes, self.dist_matrix, self.wastes, self.capacity, -1e9)
                            move_swap(adapter, node1, node2, r_idx, pos1, r_idx, pos2)
                            new_routes = adapter.routes
                            move_desc = ("swap_intra", (node1, node2))
                            candidates.append((new_routes, move_desc))
                except Exception:
                    continue

        return candidates

    def _generate_relocate_moves(
        self, routes: List[List[int]], max_new: int = 5
    ) -> Sequence[Tuple[List[List[int]], Tuple[str, Any]]]:
        """
        Generate relocate-based neighborhood moves.

        Args:
            routes: Routes to evaluate.
            max_new: Maximum number of new routes to generate.

        Returns:
            List[Tuple[List[List[int]], Tuple[str, Any]]]: List of new routes and move descriptors.
        """
        candidates: List[Tuple[List[List[int]], Tuple[str, Any]]] = []

        for _ in range(max_new):
            new_routes = copy.deepcopy(routes)

            # Try inter-route relocate
            if len(new_routes) >= 2:
                try:
                    r1_idx = self.random.randint(0, len(new_routes) - 1)
                    r2_idx = self.random.randint(0, len(new_routes) - 1)
                    if r1_idx != r2_idx and new_routes[r1_idx]:
                        pos1 = self.random.randint(0, len(new_routes[r1_idx]) - 1)
                        node = new_routes[r1_idx][pos1]
                        pos2 = self.random.randint(0, len(new_routes[r2_idx]))

                        adapter = TSLSAdapter(new_routes, self.dist_matrix, self.wastes, self.capacity, -1e9)
                        if pos2 == 0:
                            v, p_v = 0, -1
                        else:
                            v, p_v = new_routes[r2_idx][pos2 - 1], pos2 - 1
                        move_relocate(adapter, node, v, r1_idx, pos1, r2_idx, p_v)
                        new_routes = adapter.routes

                        move_desc = ("relocate_inter", (node, r1_idx, r2_idx))
                        candidates.append((new_routes, move_desc))
                except Exception:
                    continue

            # Try intra-route relocate
            if not candidates or self.random.random() < 0.5:
                try:
                    r_idx = self.random.randint(0, len(new_routes) - 1)
                    if len(new_routes[r_idx]) >= 2:
                        pos1 = self.random.randint(0, len(new_routes[r_idx]) - 1)
                        pos2 = self.random.randint(0, len(new_routes[r_idx]))
                        if pos1 != pos2:
                            node = new_routes[r_idx][pos1]
                            adapter = TSLSAdapter(new_routes, self.dist_matrix, self.wastes, self.capacity, -1e9)
                            if pos2 == 0:
                                v, p_v = 0, -1
                            else:
                                v, p_v = new_routes[r_idx][pos2 - 1], pos2 - 1
                            move_relocate(adapter, node, v, r_idx, pos1, r_idx, p_v)
                            new_routes = adapter.routes

                            move_desc = ("relocate_intra", (node, pos1, pos2))
                            candidates.append((new_routes, move_desc))
                except Exception:
                    continue

        return candidates

    def _generate_2opt_moves(
        self, routes: List[List[int]], max_new: int = 3
    ) -> Sequence[Tuple[List[List[int]], Tuple[str, Any]]]:
        """
        Generate 2-opt neighborhood moves.

        Args:
            routes: Routes to evaluate.
            max_new: Maximum number of new routes to generate.

        Returns:
            List[Tuple[List[List[int]], Tuple[str, Any]]]: List of new routes and move descriptors.
        """
        candidates: List[Tuple[List[List[int]], Tuple[str, Any]]] = []

        for _ in range(max_new):
            new_routes = copy.deepcopy(routes)

            try:
                r_idx = self.random.randint(0, len(new_routes) - 1)
                if len(new_routes[r_idx]) >= 4:
                    # For 2-opt, we need two edges (u, next_u) and (v, next_v)
                    # We'll pick random pos1, pos2
                    pos1 = self.random.randint(0, len(new_routes[r_idx]) - 2)
                    pos2 = self.random.randint(pos1 + 2, len(new_routes[r_idx]) - 1)
                    u = new_routes[r_idx][pos1]
                    v = new_routes[r_idx][pos2]

                    adapter = TSLSAdapter(new_routes, self.dist_matrix, self.wastes, self.capacity, -1e9)
                    move_2opt_intra(adapter, u, v, r_idx, pos1, r_idx, pos2)
                    new_routes = adapter.routes

                    move_desc = ("2opt", (r_idx, self.iteration))
                    candidates.append((new_routes, move_desc))
            except Exception:
                continue

        return candidates

    def _llh_random_greedy(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """
        Random-greedy local search operator.

        Args:
            routes: Routes to evaluate.
            n: Number of nodes to remove.

        Returns:
            List[List[int]]: Improved routes.
        """
        if self.params.profit_aware_operators:
            # random_profit_removal reverted to random_removal
            partial, removed = random_removal(routes, n, rng=self.random)
            return greedy_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=current_expand_pool(),
            )
        else:
            partial, removed = random_removal(routes, n, rng=self.random)
            return greedy_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=current_expand_pool(),
            )

    def _llh_worst_regret(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """
        Worst-regret (regret-2) local search operator.

        Args:
            routes: Routes to evaluate.
            n: Number of nodes to remove.

        Returns:
            List[List[int]]: Improved routes.
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

    def _llh_cluster_greedy(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """
        Cluster-greedy local search operator.

        Args:
            routes: Routes to evaluate.
            n: Number of nodes to remove.

        Returns:
            List[List[int]]: Improved routes.
        """
        if self.params.profit_aware_operators:
            # cluster_profit_removal reverted to cluster_removal
            partial, removed = cluster_removal(routes, n, self.dist_matrix, self.nodes, rng=self.random)
            return greedy_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=current_expand_pool(),
            )
        else:
            partial, removed = cluster_removal(routes, n, self.dist_matrix, self.nodes, rng=self.random)
            return greedy_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=current_expand_pool(),
            )

    def _llh_worst_greedy(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """
        Worst-greedy local search operator.

        Args:
            routes: Routes to evaluate.
            n: Number of nodes to remove.

        Returns:
            List[List[int]]: Improved routes.
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
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=current_expand_pool(),
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

    def _llh_random_regret(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """
        Random regret (regret-2) local search operator.

        Args:
            routes: Routes to evaluate.
            n: Number of nodes to remove.

        Returns:
            List[List[int]]: Improved routes.
        """
        if self.params.profit_aware_operators:
            # random_profit_removal reverted to random_removal
            partial, removed = random_removal(routes, n, rng=self.random)
            return regret_2_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=current_expand_pool(),
            )
        else:
            partial, removed = random_removal(routes, n, rng=self.random)
            return regret_2_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=current_expand_pool(),
            )
