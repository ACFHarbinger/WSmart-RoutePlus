"""
Local Search Base Module.

This module defines the abstract base class for local search algorithms.
It handles common initialization, neighbor lists, and provides wrappers
for atomic move operators.

This module also provides the local search implementation specifically
for Ant Colony Optimization (ACO). It focuses on 2-opt refinement.

Attributes:
    None

Example:
    >>> # Cannot be instantiated directly
    >>> class MyLS(LocalSearch): ...
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from logic.src.policies.helpers.operators.intra_route_local_search.k_opt import (
    move_2opt_intra,
    move_3opt_intra,
)
from logic.src.policies.helpers.operators.intra_route_local_search.relocate import (
    move_or_opt,
    move_relocate,
)
from logic.src.policies.helpers.operators.intra_route_local_search.swap import move_swap
from logic.src.policies.helpers.operators.inter_route_local_search.k_opt_star import move_2opt_star
from logic.src.policies.helpers.operators.inter_route_local_search.swap_star import move_swap_star


class LocalSearch(PolicyVizMixin, ABC):
    """
    Abstract base class for Local Search algorithms.
    Provides common infrastructure for neighbor lists and move operators.

    Attributes:
        d: The distance matrix between nodes.
        waste: A dictionary mapping node IDs to their waste amounts.
        Q: The maximum capacity of a vehicle.
        R: A parameter, likely related to route cost or penalty.
        C: A parameter, likely related to route cost or penalty.
        params: An object containing additional parameters for the local search.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        waste: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: Any,
        seed: Optional[int] = None,
    ):
        """
        Initialize Local Search base class.

        Args:
            dist_matrix: The distance matrix between nodes.
            waste: A dictionary mapping node IDs to their waste amounts.
            capacity: The maximum capacity of a vehicle.
            R: A parameter, likely related to route cost or penalty.
            C: A parameter, likely related to route cost or penalty.
            seed: Seed for the random number generator.
            params: An object containing additional parameters for the local search,
                    such as time_limit.
        """
        self.d = np.array(dist_matrix)
        self.waste = waste
        self.Q = capacity
        self.R = R
        self.C = C
        self.params = params
        self.random = random.Random(seed) if seed is not None else random.Random()

        # Common initialization for neighbors (used by all LS)
        n_nodes = len(dist_matrix)
        self.neighbors = {}
        for i in range(1, n_nodes):
            row = self.d[i]
            order = np.argsort(row)
            cands = []
            for c in order:
                if c not in (i, 0):
                    cands.append(c)
                    if len(cands) >= 10:
                        break
            self.neighbors[i] = cands

        self.node_map: Dict[int, Tuple[int, int]] = {}
        self.route_loads: List[float] = []
        self.routes: List[List[int]] = []

    @abstractmethod
    def optimize(self, solution: Any) -> Any:
        """
        Optimize the given solution.

        Args:
            solution: The solution to optimize.

        Returns:
            Any: The optimized solution.
        """
        pass

    def _optimize_internal(self):
        """
        Core local search loop. Assumes self.routes is populated.
        """
        self.route_loads = [self._calc_load_fresh(r) for r in self.routes]

        # Initialize node map
        self.node_map.clear()
        for ri, r in enumerate(self.routes):
            for pi, node in enumerate(r):
                self.node_map[node] = (ri, pi)

        improved = True
        it = 0
        t_start = time.perf_counter()
        while improved and it < self.params.local_search_iterations:
            if self.params.time_limit > 0 and time.perf_counter() - t_start > self.params.time_limit:
                break

            improved = False
            it += 1

            nodes = [n for n in self.neighbors if n in self.node_map]
            self.random.shuffle(nodes)

            for u in nodes:
                if self._process_node(u):
                    improved = True
                    break

            self._viz_record(
                iteration=it,
                n_routes=len(self.routes),
                improved=int(improved),
            )

    def _calc_load_fresh(self, r: List[int]) -> float:
        """
        Calculate the load of a route.

        Args:
            r: Route.

        Returns:
            float: The load of the route.
        """
        return sum(self.waste.get(x, 0) for x in r)

    def _process_node(self, u: int) -> bool:  # noqa: C901
        """
        Process a node.

        Args:
            u: Node.

        Returns:
            bool: Whether the move was successful.
        """
        u_loc = self.node_map.get(u)
        if not u_loc:
            return False
        r_u, p_u = u_loc

        for v in self.neighbors[u]:
            v_loc = self.node_map.get(v)
            if not v_loc:
                continue
            r_v, p_v = v_loc

            if self._move_relocate(u, v, r_u, p_u, r_v, p_v):
                return True
            if self._move_swap(u, v, r_u, p_u, r_v, p_v):
                return True
            if r_u != r_v:
                if self._move_2opt_star(u, v, r_u, p_u, r_v, p_v):
                    return True
                if self._move_swap_star(u, v, r_u, p_u, r_v, p_v):
                    return True
            else:
                if self._move_2opt_intra(u, v, r_u, p_u, r_v, p_v):
                    return True
                if self._move_3opt_intra(u, v, r_u, p_u, r_v, p_v, self.random):
                    return True
                if self._move_or_opt(u, self.random.choice([1, 2, 3]), r_u, p_u):
                    return True

        return False

    def _update_map(self, affected_indices: Set[int]):
        """
        Update the node map and route loads.

        Args:
            affected_indices: Set of route indices that were affected by the move.
        """
        for ri in sorted(affected_indices):
            for pi, node in enumerate(self.routes[ri]):
                self.node_map[node] = (ri, pi)
            self.route_loads[ri] = self._calc_load_fresh(self.routes[ri])

    def _get_load_cached(self, ri: int) -> float:
        """
        Get the load of a route.

        Args:
            ri: Route index.

        Returns:
            float: The load of the route.
        """
        return self.route_loads[ri]

    def _move_relocate(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """
        Perform relocate move.

        Args:
            u: Node to move.
            v: Node to move.
            r_u: Route index of node u.
            p_u: Position index of node u.
            r_v: Route index of node v.
            p_v: Position index of node v.

        Returns:
            bool: Whether the move was successful.
        """
        return move_relocate(self, u, v, r_u, p_u, r_v, p_v)

    def _move_swap(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """
        Perform swap move.

        Args:
            u: Node to move.
            v: Node to move.
            r_u: Route index of node u.
            p_u: Position index of node u.
            r_v: Route index of node v.
            p_v: Position index of node v.

        Returns:
            bool: Whether the move was successful.
        """
        return move_swap(self, u, v, r_u, p_u, r_v, p_v)

    def _move_swap_star(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """
        Perform swap star move.

        Args:
            u: Node to move.
            v: Node to move.
            r_u: Route index of node u.
            p_u: Position index of node u.
            r_v: Route index of node v.
            p_v: Position index of node v.

        Returns:
            bool: Whether the move was successful.
        """
        return move_swap_star(self, u, v, r_u, p_u, r_v, p_v)

    def _move_3opt_intra(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int, rng: random.Random) -> bool:
        """
        Perform 3-opt intra-route move.

        Args:
            u: Node to move.
            v: Node to move.
            r_u: Route index of node u.
            p_u: Position index of node u.
            r_v: Route index of node v.
            p_v: Position index of node v.
            rng: Random number generator.

        Returns:
            bool: Whether the move was successful.
        """
        return move_3opt_intra(self, u, v, r_u, p_u, r_v, p_v, rng)

    def _move_2opt_star(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """
        Perform 2-opt star move.

        Args:
            u: Node to move.
            v: Node to move.
            r_u: Route index of node u.
            p_u: Position index of node u.
            r_v: Route index of node v.
            p_v: Position index of node v.

        Returns:
            bool: Whether the move was successful.
        """
        return move_2opt_star(self, u, v, r_u, p_u, r_v, p_v)

    def _move_2opt_intra(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """
        Perform 2-opt intra-route move.

        Args:
            u: Node to move.
            v: Node to move.
            r_u: Route index of node u.
            p_u: Position index of node u.
            r_v: Route index of node v.
            p_v: Position index of node v.

        Returns:
            bool: Whether the move was successful.
        """
        return move_2opt_intra(self, u, v, r_u, p_u, r_v, p_v)

    def _move_or_opt(self, u: int, chain_len: int, r_u: int, p_u: int) -> bool:
        """
        Perform or-opt move.

        Args:
            u: Node to move (unused, derived from r_u/p_u by shared operator).
            chain_len: Length of the chain to move.
            r_u: Route index of node u.
            p_u: Position index of node u.

        Returns:
            bool: Whether the move was successful.
        """
        return move_or_opt(self, r_u, p_u, chain_len)


class ACOLocalSearch(LocalSearch):
    """
    Local Search module for K-Sparse ACO.
    Implements 2-opt local search refinement.

    Attributes:
        None

    Example:
        >>> from logic.src.policies.local_search import ACOLocalSearch
        >>> aco_local_search = ACOLocalSearch(dist_matrix, waste, capacity, R, C, params)
        >>> optimized_routes = aco_local_search.optimize(solution)
    """

    def optimize(self, solution: List[List[int]]) -> List[List[int]]:
        """
        Apply full set of local search operators to the solution.

        Args:
            solution: List of routes, where each route is a list of node indices.

        Returns:
            List of optimized routes.
        """
        # Create a deep copy of routes to modify
        self.routes = [r[:] for r in solution]

        # Run optimization
        self._optimize_internal()

        return self.routes
