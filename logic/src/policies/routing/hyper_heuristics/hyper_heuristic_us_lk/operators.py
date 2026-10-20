"""
Operator implementations for HULK hyper-heuristic.

Wraps unstringing/stringing operators and local search moves.

Attributes:
    HULKOperators: Operator collection for HULK hyper-heuristic.

Example:
    >>> operators = HULKOperators(dist_matrix, wastes, capacity, R, C, mandatory_nodes)
    >>> operators.apply_unstring_type_i(solution, 10)

Intra-route local search moves live in ``hulk_local_search`` and are mixed into ``HULKOperators``.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.operators import (
    greedy_insertion,
    greedy_profit_insertion,
    regret_2_insertion,
    regret_2_profit_insertion,
    shaw_profit_removal,
    shaw_removal,
    string_removal,
)
from logic.src.policies.helpers.operators.generalized_insertion_and_deletion import (
    stringing_insertion,
    stringing_profit_insertion,
    unstringing_profit_removal,
    unstringing_removal,
)

from .hulk_local_search import HULKLocalSearchMixin as HULKLocalSearchMixin
from .solution import Solution


class HULKOperators(HULKLocalSearchMixin):
    """Operator collection for HULK hyper-heuristic.

    Implements unstringing operators (Type I, II, III, IV, Shaw, String)
    for removing nodes, and stringing operators for reinserting nodes.
    Includes local search moves: 2-opt, 3-opt, swap, relocate.

    Attributes:
        dist_matrix (np.ndarray): Distance matrix.
        wastes (Dict[int, float]): Waste dictionary.
        capacity (float): Vehicle capacity.
        R (float): Revenue multiplier.
        C (float): Cost multiplier.
        mandatory_nodes (List[int]): Must-visit nodes.
        expand_pool (bool): Includes globally unassigned pool for VRPP support.
        profit_aware_operators (bool): Use profit-aware versions of operators.
        rng (random.Random): Random number generator.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        mandatory_nodes: Optional[List[int]] = None,
        seed: Optional[int] = None,
        expand_pool: bool = False,
        profit_aware_operators: bool = False,
    ):
        """
        Initialize operators.

        Args:
            dist_matrix: Distance matrix.
            wastes: Waste dictionary.
            capacity: Vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            mandatory_nodes: Must-visit nodes.
            seed: Random seed.
            expand_pool: Includes globally unassigned pool for VRPP support.
            profit_aware_operators: Use profit-aware versions of operators.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.mandatory_nodes = mandatory_nodes or []
        self.expand_pool = expand_pool
        self.profit_aware_operators = profit_aware_operators
        self.rng = random.Random(seed) if seed is not None else random.Random()

    def apply_unstring_type_i(self, solution: Solution, n_remove: int) -> Tuple[Solution, List[int]]:
        """Applies Unstringing Type I removal operator.

        Args:
            solution: The current solution.
            n_remove: Number of nodes to remove.

        Returns:
            Tuple[Solution, List[int]]: The partial solution and the removed nodes.
        """
        return self._apply_unstring_wrapper(solution, n_remove, 1)

    def apply_unstring_type_ii(self, solution: Solution, n_remove: int) -> Tuple[Solution, List[int]]:
        """Applies Unstringing Type II removal operator.

        Args:
            solution: The current solution.
            n_remove: Number of nodes to remove.

        Returns:
            Tuple[Solution, List[int]]: The partial solution and the removed nodes.
        """
        return self._apply_unstring_wrapper(solution, n_remove, 2)

    def apply_unstring_type_iii(self, solution: Solution, n_remove: int) -> Tuple[Solution, List[int]]:
        """Applies Unstringing Type III removal operator.

        Args:
            solution: The current solution.
            n_remove: Number of nodes to remove.

        Returns:
            Tuple[Solution, List[int]]: The partial solution and the removed nodes.
        """
        return self._apply_unstring_wrapper(solution, n_remove, 3)

    def apply_unstring_type_iv(self, solution: Solution, n_remove: int) -> Tuple[Solution, List[int]]:
        """Applies Unstringing Type IV removal operator.

        Args:
            solution: The current solution.
            n_remove: Number of nodes to remove.

        Returns:
            Tuple[Solution, List[int]]: The partial solution and the removed nodes.
        """
        return self._apply_unstring_wrapper(solution, n_remove, 4)

    def apply_unstring_shaw(self, solution: Solution, n_remove: int) -> Tuple[Solution, List[int]]:
        """Applies Shaw removal operator.

        Args:
            solution: The current solution.
            n_remove: Number of nodes to remove.

        Returns:
            Tuple[Solution, List[int]]: The partial solution and the removed nodes.
        """
        routes = [list(r) for r in solution.routes]
        if getattr(self, "profit_aware_operators", False):
            partial, removed = shaw_profit_removal(routes, n_remove, self.dist_matrix, self.wastes, self.R, self.C)
        else:
            partial, removed = shaw_removal(routes, n_remove, self.dist_matrix)
        return Solution(partial, self.dist_matrix, self.wastes, self.capacity, self.R, self.C), removed

    def apply_unstring_string(self, solution: Solution, n_remove: int) -> Tuple[Solution, List[int]]:
        """Applies String removal operator.

        Args:
            solution: The current solution.
            n_remove: Number of nodes to remove.

        Returns:
            Tuple[Solution, List[int]]: The partial solution and the removed nodes.
        """
        routes = [list(r) for r in solution.routes]
        partial, removed = string_removal(routes, n_remove, self.dist_matrix, rng=self.rng)
        return Solution(partial, self.dist_matrix, self.wastes, self.capacity, self.R, self.C), removed

    def _apply_unstring_wrapper(
        self, solution: Solution, n_remove: int, unstring_type: int
    ) -> Tuple[Solution, List[int]]:
        """
        Wrapper for unstringing operators.

        Args:
            solution: Solution to apply the move to.
            n_remove: Number of nodes to remove.
            unstring_type: Type of unstringing to apply.

        Returns:
            Tuple[Solution, List[int]]: Tuple containing the new solution and the list of removed nodes.
        """
        routes = [list(r) for r in solution.routes]
        if getattr(self, "profit_aware_operators", False):
            new_routes, removed = unstringing_profit_removal(
                routes, n_remove, unstring_type, self.dist_matrix, self.wastes, self.R, self.C, rng=self.rng
            )
        else:
            new_routes, removed = unstringing_removal(routes, n_remove, unstring_type, self.dist_matrix, rng=self.rng)

        return Solution(new_routes, self.dist_matrix, self.wastes, self.capacity, self.R, self.C), removed

    def apply_string_repair(self, solution: Solution, removed: List[int], string_type: str) -> Solution:
        """
        Apply stringing operator to reinsert removed nodes.

        Args:
            solution: Current solution.
            removed: Nodes to reinsert.
            string_type: Type of stringing ("type_i", "type_ii", "type_iii", "type_iv").

        Returns:
            Repaired solution.
        """
        string_map = {"type_i": 1, "type_ii": 2, "type_iii": 3, "type_iv": 4}
        if string_type not in string_map:
            if string_type == "regret_2":
                return self._regret_2_repair(solution, removed)
            return self._greedy_repair(solution, removed)
        op_type = string_map[string_type]
        if getattr(self, "profit_aware_operators", False):
            routes = stringing_profit_insertion(
                [list(r) for r in solution.routes],
                removed,
                op_type,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                self.rng,
                self.expand_pool,
            )
        else:
            routes = stringing_insertion(
                [list(r) for r in solution.routes],
                removed,
                op_type,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.mandatory_nodes,
                self.rng,
                self.expand_pool,
            )
        return Solution(routes, self.dist_matrix, self.wastes, self.capacity, self.R, self.C)

    def _greedy_repair(self, solution: Solution, removed: List[int]) -> Solution:
        """
        Repair the solution using greedy insertion.

        Args:
            solution: Solution to repair.
            removed: List of removed nodes.

        Returns:
            Solution: Repaired solution.
        """
        if getattr(self, "profit_aware_operators", False):
            routes = greedy_profit_insertion(
                [list(r) for r in solution.routes],
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                expand_pool=self.expand_pool,
            )
        else:
            routes = greedy_insertion(
                [list(r) for r in solution.routes],
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.mandatory_nodes,
                expand_pool=self.expand_pool,
            )
        return Solution(routes, self.dist_matrix, self.wastes, self.capacity, self.R, self.C)

    def _regret_2_repair(self, solution: Solution, removed: List[int]) -> Solution:
        """
        Repair the solution using regret-2 insertion.

        Args:
            solution: Solution to repair.
            removed: List of removed nodes.

        Returns:
            Solution: Repaired solution.
        """
        if getattr(self, "profit_aware_operators", False):
            routes = regret_2_profit_insertion(
                [list(r) for r in solution.routes],
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                expand_pool=self.expand_pool,
            )
        else:
            routes = regret_2_insertion(
                [list(r) for r in solution.routes],
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.mandatory_nodes,
                expand_pool=self.expand_pool,
            )
        return Solution(routes, self.dist_matrix, self.wastes, self.capacity, self.R, self.C)
