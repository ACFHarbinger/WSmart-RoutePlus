"""Tabu list, frequency memory, elite pool, intensification and diversification of the tabu search.

Attributes:
    TSMemoryMixin: Short- and long-term memory of ``TSSolver``.

Example:
    >>> class TSSolver(TSMemoryMixin, TSMoveGenerationMixin): ...
"""

import copy
import random
from typing import TYPE_CHECKING, Any, Callable, DefaultDict, Deque, Dict, List, Tuple

import numpy as np

from logic.src.configs.policies.routing.meta_heuristics.ts import TSConfig
from logic.src.envs.common.expand_pool import current_expand_pool
from logic.src.policies.helpers.operators.recreate_repair import (
    greedy_insertion,
    greedy_profit_insertion,
)
from logic.src.policies.helpers.operators.solution_initialization.greedy_si import build_greedy_routes


class TSMemoryMixin:
    """Short-term (tabu list) and long-term (frequency, elite pool) memory of :class:`TSSolver`."""

    # Provided by TSSolver.__init__; annotations only.
    dist_matrix: np.ndarray
    wastes: Dict[int, float]
    capacity: float
    R: float
    C: float
    params: TSConfig
    mandatory_nodes: List[int]
    nodes: List[int]
    random: random.Random
    tabu_list: Deque[Tuple[str, Any, int]]
    node_frequency: DefaultDict[int, int]
    move_frequency: DefaultDict[Tuple[str, Any], int]
    elite_solutions: List[Tuple[List[List[int]], float]]
    iteration: int
    iterations_since_improvement: int
    _llh_pool: List[Callable[[List[List[int]], int], List[List[int]]]]

    if TYPE_CHECKING:

        def _evaluate(self, routes: List[List[int]]) -> float: ...

    def _is_tabu(self, move_desc: Tuple[str, Any]) -> bool:
        """Check if a move is currently tabu.

        Args:
            move_desc: Description of the move.

        Returns:
            bool: True if the move is tabu, False otherwise.
        """
        move_type, move_attrs = move_desc
        for tabu_type, tabu_attrs, expiration in self.tabu_list:
            if tabu_type == move_type and tabu_attrs == move_attrs and expiration > self.iteration:
                return True
        return False

    def _add_to_tabu_list(self, move_desc: Tuple[str, Any]):
        """Add a move to the tabu list with appropriate tenure.

        Args:
            move_desc: Description of the move.
        """
        move_type, move_attrs = move_desc

        # Compute tabu tenure
        tenure = self._compute_dynamic_tenure() if self.params.dynamic_tenure else self.params.tabu_tenure

        expiration = self.iteration + tenure
        self.tabu_list.append((move_type, move_attrs, expiration))

        # Clean expired entries
        self._clean_tabu_list()

    def _clean_tabu_list(self):
        """Remove expired tabu entries.

        While the tabu list is not empty and the first entry has expired, remove it.
        """
        while self.tabu_list and self.tabu_list[0][2] <= self.iteration:
            self.tabu_list.popleft()

    def _compute_dynamic_tenure(self) -> int:
        """
        Compute dynamic tabu tenure based on search state.

        Shorter tenure during improvement, longer during stagnation.

        Returns:
            int: Dynamic tabu tenure.
        """
        base_tenure = self.params.tabu_tenure

        if self.iterations_since_improvement < 10:
            # Rapid improvement — shorter tenure for exploitation
            return max(self.params.min_tenure, int(base_tenure * 0.7))
        elif self.iterations_since_improvement > 50:
            # Stagnation — longer tenure for diversification
            return min(self.params.max_tenure, int(base_tenure * 1.5))
        else:
            return base_tenure

    def _update_frequency_memory(self, routes: List[List[int]]):
        """Update node frequency counters.

        Args:
            routes: Routes to update frequencies for.
        """
        for route in routes:
            for node in route:
                self.node_frequency[node] += 1

    def _update_move_frequency(self, move_desc: Tuple[str, Any]):
        """Update move frequency counters.

        Args:
            move_desc: Description of the move.
        """
        self.move_frequency[move_desc] += 1

    def _compute_frequency_penalty(self, move_desc: Tuple[str, Any]) -> float:
        """
        Compute penalty based on move frequency to encourage diversification.

        Frequently used moves receive higher penalties.

        Args:
            move_desc: Description of the move.

        Returns:
            float: Penalty value.
        """
        freq = self.move_frequency.get(move_desc, 0)
        return self.params.frequency_penalty_weight * freq

    def _update_elite_pool(self, routes: List[List[int]], profit: float):
        """
        Update pool of elite (high-quality) solutions.

        Args:
            routes: Routes to evaluate.
            profit: Profit of the routes.
        """
        # Add to elite pool
        self.elite_solutions.append((copy.deepcopy(routes), profit))

        # Sort by profit (descending)
        self.elite_solutions.sort(key=lambda x: x[1], reverse=True)

        # Keep only top-k elite solutions
        self.elite_solutions = self.elite_solutions[: self.params.elite_size]

    def _intensification_phase(self, best_routes: List[List[int]]) -> List[List[int]]:
        """
        Intensification: Return to best solution and explore neighborhood intensively.

        This encourages exploitation of promising regions.

        Args:
            best_routes: Best routes found so far.

        Returns:
            List[List[int]]: Intensified routes.
        """
        # Start from best known solution
        routes = copy.deepcopy(best_routes)

        # Apply local search intensively
        for _ in range(5):
            improved = False
            for llh in self._llh_pool:
                try:
                    new_routes = llh(copy.deepcopy(routes), self.params.n_removal)
                    new_profit = self._evaluate(new_routes)
                    if new_profit > self._evaluate(routes):
                        routes = new_routes
                        improved = True
                        break
                except Exception:
                    continue
            if not improved:
                break

        return routes

    def _diversification_phase(self, current_routes: List[List[int]]) -> List[List[int]]:
        """
        Diversification: Perturb current solution to explore new regions.

        Uses frequency-based penalties to favor rarely visited nodes.

        Args:
            current_routes: Current routes to perturb.

        Returns:
            List[List[int]]: Perturbed routes.
        """
        routes = copy.deepcopy(current_routes)

        # Remove frequently visited nodes
        all_nodes_in_routes = [node for route in routes for node in route]
        if not all_nodes_in_routes:
            return routes

        # Sort by frequency (descending)
        freq_nodes = sorted(all_nodes_in_routes, key=lambda n: self.node_frequency.get(n, 0), reverse=True)

        # Remove top frequent nodes
        n_remove = min(self.params.n_removal * 2, len(freq_nodes) // 2)
        nodes_to_remove = set(freq_nodes[:n_remove])

        # Rebuild routes without these nodes
        new_routes = []
        removed_nodes = []
        for route in routes:
            new_route = [n for n in route if n not in nodes_to_remove]
            removed_nodes.extend([n for n in route if n in nodes_to_remove])
            if new_route:
                new_routes.append(new_route)

        # Reinsert removed nodes
        if removed_nodes:
            if self.params.profit_aware_operators:
                new_routes = greedy_profit_insertion(
                    new_routes,
                    removed_nodes,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    self.R,
                    self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=current_expand_pool(),
                )
            else:
                new_routes = greedy_insertion(
                    new_routes,
                    removed_nodes,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=current_expand_pool(),
                )

        return new_routes

    def _diversification_restart(self) -> List[List[int]]:
        """
        Diversification restart: Build a new solution avoiding frequent patterns.

        This is triggered when search stagnates.

        Returns:
            List[List[int]]: New routes with diversified node selections.
        """
        # If we have elite solutions, use path relinking
        if len(self.elite_solutions) >= 2:
            return self._path_relink(self.elite_solutions[0][0], self.elite_solutions[1][0])

        # Otherwise, build a new solution with frequency penalties
        return self._build_diversified_solution()

    def _build_diversified_solution(self) -> List[List[int]]:
        """
        Build a new solution that avoids frequently used nodes.

        Penalize nodes with high frequency to encourage exploration.

        Returns:
            List[List[int]]: New routes with diversified node selections.
        """
        # Modify wastes temporarily to penalize frequent nodes
        modified_wastes = {}
        max_freq = max(self.node_frequency.values()) if self.node_frequency else 1

        for node in self.nodes:
            freq = self.node_frequency.get(node, 0)
            penalty = (freq / max_freq) * 0.3  # Up to 30% penalty
            modified_wastes[node] = self.wastes.get(node, 0.0) * (1.0 - penalty)

        # Build solution with modified wastes
        return build_greedy_routes(
            dist_matrix=self.dist_matrix,
            wastes=modified_wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
        )

    def _path_relink(self, routes1: List[List[int]], routes2: List[List[int]]) -> List[List[int]]:
        """
        Path relinking: Generate intermediate solution between two elite solutions.

        This explores trajectories between high-quality solutions.

        Args:
            routes1: First set of routes.
            routes2: Second set of routes.

        Returns:
            List[List[int]]: New routes formed by blending node selections.
        """
        # Simple implementation: blend node selections
        nodes1 = set(node for route in routes1 for node in route)
        nodes2 = set(node for route in routes2 for node in route)

        # Nodes in both solutions are kept
        common_nodes = nodes1 & nodes2

        # Randomly select from differing nodes
        diff_nodes = (nodes1 | nodes2) - common_nodes
        selected_diff = set(self.random.sample(list(diff_nodes), min(len(diff_nodes), len(diff_nodes) // 2)))

        # Build routes with selected nodes
        selected_nodes = list(common_nodes | selected_diff)

        # Use greedy insertion to rebuild
        if self.params.profit_aware_operators:
            return greedy_profit_insertion(
                [],
                selected_nodes,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=current_expand_pool(),
            )
        else:
            return greedy_insertion(
                [],
                selected_nodes,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=current_expand_pool(),
            )
