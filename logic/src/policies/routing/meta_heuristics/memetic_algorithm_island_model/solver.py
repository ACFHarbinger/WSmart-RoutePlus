"""
Memetic Algorithm with Island Model (MA-IM) for VRPP.

Attributes:
    MemeticAlgorithmIslandModelSolver: Core solver class for MA-IM.

Example:
    >>> solver = MemeticAlgorithmIslandModelSolver(dist_matrix, wastes, capacity, R, C, params)
    >>> routes, profit, cost = solver.solve()
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.configs.policies.routing.meta_heuristics.aco_ks import KSparseACOConfig
from logic.src.configs.policies.routing.meta_heuristics.ma_im import MemeticAlgorithmIslandModelConfig
from logic.src.envs.common.expand_pool import current_expand_pool
from logic.src.policies.helpers.local_search.local_search_general import GeneralLocalSearch
from logic.src.policies.helpers.operators import (
    greedy_insertion,
    greedy_profit_insertion,
    random_removal,
)
from logic.src.policies.helpers.operators.solution_initialization.nearest_neighbor_si import build_nn_routes


class MemeticAlgorithmIslandModelSolver:
    """
    Memetic Algorithm with Island Model (MA-IM) solver for VRPP.

    Adaptation of SLC with rigorous nomenclature.

    Attributes:
        dist_matrix: Symmetric distance matrix.
        wastes: Mapping of bin IDs to waste quantities.
        capacity: Maximum vehicle collection capacity.
        R: Revenue per kg of waste.
        C: Cost per kg traveled.
        params: Algorithm-specific parameters.
        mandatory_nodes: Nodes that must be visited.
        n_nodes: Number of customer nodes.
        nodes: List of customer node indices.
        random: Random number generator.
        ls: Local search optimizer.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: MemeticAlgorithmIslandModelConfig,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """Initializes the Memetic Algorithm Island Model solver.

        Args:
            dist_matrix: Distance matrix between nodes.
            wastes: Dictionary mapping node indices to waste amounts.
            capacity: Vehicle capacity.
            R: Revenue per unit collected.
            C: Cost per unit distance.
            params: MA-IM parameters.
            mandatory_nodes: Optional list of nodes that must be visited.

        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes or []
        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))
        self.random = random.Random(self.params.seed) if self.params.seed is not None else random.Random()

        # Pre-instantiate Local Search for reuse
        aco_params = KSparseACOConfig(
            local_search_iterations=self.params.local_search_iterations,
            time_limit=self.params.time_limit,
            profit_aware_operators=self.params.profit_aware_operators,
            seed=self.params.seed,
        )
        self.ls = GeneralLocalSearch(
            dist_matrix=self.dist_matrix,
            waste=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            params=aco_params,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """Execute the Memetic Island Model algorithm.

        Returns:
            Tuple[List[List[int]], float, float]: Optimized (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.perf_counter()

        # Initialize islands: list of lists of (routes, profit)
        islands: List[List[Tuple[List[List[int]], float]]] = [self._new_island() for _ in range(self.params.n_islands)]
        stagnation: List[int] = [0] * self.params.n_islands
        island_best: List[float] = [max(p for _, p in island) for island in islands]

        # Global best
        best_routes, best_profit = self._global_best(islands)
        best_cost = self._cost(best_routes)

        for iteration in range(self.params.max_generations):
            if self.params.time_limit > 0 and time.perf_counter() - start > self.params.time_limit:
                break

            # --- Intra-island evolution: local perturbation ---
            for _i_idx, island in enumerate(islands):
                for c_idx in range(len(island)):
                    routes, profit = island[c_idx]
                    new_routes = self._mutate(routes)
                    new_profit = self._evaluate(new_routes)
                    if new_profit > profit:
                        island[c_idx] = (new_routes, new_profit)

            # --- Inter-island competition: probabilistic match ---
            island_indices = list(range(self.params.n_islands))
            self.random.shuffle(island_indices)
            for k in range(0, len(island_indices) - 1, 2):
                a_idx = island_indices[k]
                b_idx = island_indices[k + 1]

                fit_a: float = float(sum(p for _, p in islands[a_idx]))
                fit_b: float = float(sum(p for _, p in islands[b_idx]))

                total = abs(fit_a) + abs(fit_b) + 1e-9
                p_win_a = (fit_a - min(fit_a, fit_b) + 1e-9) / total

                if self.random.random() < p_win_a:
                    winner, loser = a_idx, b_idx
                else:
                    winner, loser = b_idx, a_idx

                # Weakest chromosome in losing island adopts structure from winner's best
                winner_best_routes = max(islands[winner], key=lambda x: x[1])[0]
                loser_worst_idx = int(np.argmin([p for _, p in islands[loser]]))
                child = self._recombine(islands[loser][loser_worst_idx][0], winner_best_routes)
                child_profit = self._evaluate(child)
                islands[loser][loser_worst_idx] = (child, child_profit)

            # --- Stagnation check and island regeneration ---
            for i_idx, island in enumerate(islands):
                current_best = max(p for _, p in island)
                if current_best > island_best[i_idx] + 1e-9:
                    island_best[i_idx] = current_best
                    stagnation[i_idx] = 0
                else:
                    stagnation[i_idx] += 1
                    if stagnation[i_idx] >= self.params.stagnation_limit:
                        islands[i_idx] = self._new_island()
                        stagnation[i_idx] = 0
                        island_best[i_idx] = max(p for _, p in islands[i_idx])

            # Update global best
            iter_best_routes, iter_best_profit = self._global_best(islands)
            if iter_best_profit > best_profit:
                best_routes = copy.deepcopy(iter_best_routes)
                best_profit = iter_best_profit
                best_cost = self._cost(best_routes)

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                n_teams=self.params.n_islands,
            )

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _new_island(self) -> List[Tuple[List[List[int]], float]]:
        """Create a fresh island of `island_size` chromosomes.

        Args:
            None.

        Returns:
            List[Tuple[List[List[int]], float]]: A list of chromosome-fitness pairs.
        """
        island = []
        for _ in range(self.params.island_size):
            routes = self._build_random_solution()
            profit = self._evaluate(routes)
            island.append((routes, profit))
        return island

    def _build_random_solution(self) -> List[List[int]]:
        """Order-dependent sequential construction (matches ALNS style).

        Args:
            None.

        Returns:
            List[List[int]]: A randomly constructed routing solution.
        """
        optimized_routes = build_nn_routes(
            nodes=self.nodes,
            mandatory_nodes=self.mandatory_nodes,
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
            rng=self.random,
        )
        return optimized_routes

    def _mutate(self, routes: List[List[int]]) -> List[List[int]]:
        """Mutation operator using destroy-repair.

        Args:
            routes: The routing solution to mutate.

        Returns:
            List[List[int]]: The mutated and refined routing solution.
        """
        try:
            partial, removed = random_removal(routes, self.params.n_removal, self.random)
            if self.params.profit_aware_operators:
                repaired = greedy_profit_insertion(
                    routes=partial,
                    removed_nodes=removed,
                    dist_matrix=self.dist_matrix,
                    wastes=self.wastes,
                    capacity=self.capacity,
                    R=self.R,
                    C=self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=current_expand_pool(),
                )
            else:
                repaired = greedy_insertion(
                    routes=partial,
                    removed_nodes=removed,
                    dist_matrix=self.dist_matrix,
                    wastes=self.wastes,
                    capacity=self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=current_expand_pool(),
                )
            # Apply comprehensive local search (reusing instance)
            return self.ls.optimize(repaired)
        except Exception:
            return copy.deepcopy(routes)

    def _recombine(self, loser_routes: List[List[int]], winner_routes: List[List[int]]) -> List[List[int]]:
        """Recombine loser's route with winner's best route via OX-inspired crossover.

        Extracts a random segment from the winner's flat tour and inserts
        unvisited nodes from that segment into the loser's solution.

        Args:
            loser_routes: Losing chromosome's route set.
            winner_routes: Winner's best chromosome route set.

        Returns:
            List[List[int]]: New child routing solution.
        """
        winner_flat = [n for r in winner_routes for n in r]
        loser_flat = [n for r in loser_routes for n in r]

        if len(winner_flat) < 2:
            return copy.deepcopy(loser_routes)

        # Extract a random segment from the winner
        a = self.random.randint(0, len(winner_flat) - 1)
        b = self.random.randint(a, min(a + max(1, len(winner_flat) // 3), len(winner_flat)))
        segment = winner_flat[a:b]
        segment_set = set(segment)

        # Order Crossover (OX): fill remaining from loser preserving order
        remaining = [n for n in loser_flat if n not in segment_set]
        insert_pos = min(a, len(remaining))
        child_flat = remaining[:insert_pos] + segment + remaining[insert_pos:]

        # Split into routes by capacity
        child_routes: List[List[int]] = []
        curr_route: List[int] = []
        load = 0.0
        for node in child_flat:
            waste = self.wastes.get(node, 0.0)
            if load + waste <= self.capacity:
                curr_route.append(node)
                load += waste
            else:
                if curr_route:
                    child_routes.append(curr_route)
                curr_route = [node]
                load = waste
        if curr_route:
            child_routes.append(curr_route)

        # Ensure mandatory nodes
        visited = {n for r in child_routes for n in r}
        for n in self.mandatory_nodes:
            if n not in visited:
                child_routes.append([n])

        # Apply comprehensive local search (reusing instance)
        return self.ls.optimize(child_routes)

    def _global_best(self, islands: List[List[Tuple[List[List[int]], float]]]) -> Tuple[List[List[int]], float]:
        """Return the best (routes, profit) across all islands.

        Args:
            islands: The set of all islands and their chromosomes.

        Returns:
            Tuple[List[List[int]], float]: The best routing solution and its profit.
        """
        best_p = -float("inf")
        best_r: List[List[int]] = []
        for island in islands:
            for routes, profit in island:
                if profit > best_p:
                    best_p = profit
                    best_r = routes
        return copy.deepcopy(best_r), best_p

    def _evaluate(self, routes: List[List[int]]) -> float:
        """Net profit for a set of routes.

        Args:
            routes: The routing solution to evaluate.

        Returns:
            float: The net profit (revenue - cost).
        """
        if not routes:
            return 0.0
        rev = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        return rev - self._cost(routes) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        """Total routing distance.

        Args:
            routes: The routing solution to calculate distance for.

        Returns:
            float: Total distance traveled.
        """
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total
