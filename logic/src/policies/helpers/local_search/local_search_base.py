"""
Local Search Base Module.

This module defines the abstract base class for local search algorithms.
It handles common initialization, neighbor lists, and provides wrappers
for atomic move operators.

Attributes:
    LocalSearch: Abstract base class for local search algorithms.

Example:
    >>> # Cannot be instantiated directly
    >>> class MyLS(LocalSearch): ...

The class is split into mixins: pair evaluation and route bookkeeping live in
``local_search_pairs.py`` (:class:`LocalSearchPairMixin`), the move-operator wrappers in
``local_search_moves.py`` (:class:`LocalSearchMovesMixin`).
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from logic.src.interfaces import IAcceptanceCriterion

from .local_search_moves import LocalSearchMovesMixin
from .local_search_pairs import LocalSearchPairMixin


class LocalSearch(LocalSearchPairMixin, LocalSearchMovesMixin, ABC):
    """Abstract base class for Local Search algorithms.

    Provides common infrastructure for neighbor lists, move operators, and
    acceptance criteria management for routing optimization.

    Attributes:
        d (np.ndarray): Distance matrix.
        waste (Dict[int, float]): Dictionary mapping node IDs to their waste/demand.
        Q (float): Vehicle capacity.
        R (float): Revenue factor per unit of waste collected.
        C (float): Cost factor per unit of distance traveled.
        params (Any): Configuration parameters for the local search.
        penalty_capacity (float): Current penalty factor for capacity violations.
        random (random.Random): Local random number generator.
        acceptance_criterion (IAcceptanceCriterion): Strategy for move acceptance.
        neighbors (Dict[int, List[int]]): Granular neighbor list for each node.
        node_map (Dict[int, Tuple[int, int]]): Maps node IDs to (route_idx, pos).
        route_loads (List[float]): Cached total waste for each route.
        routes (List[List[int]]): Current set of routes in the solution.
        top_insertions (Dict[int, Dict[int, List[Tuple[float, int]]]]): Cache of
            best insertion positions for each node into each route.
        x_coords (Optional[np.ndarray]): X-coordinates for polar sector calculations.
        y_coords (Optional[np.ndarray]): Y-coordinates for polar sector calculations.
        route_sectors (Dict[int, Optional[Tuple[float, float]]]): Cached polar
            sectors (min_angle, max_angle) for each route.
        current_profit (float): Current objective value (total profit).
    """

    def __init__(
        self,
        dist_matrix: Optional[np.ndarray],
        waste: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: Any,
        mandatory_nodes: Optional[List[int]] = None,
        neighbors: Optional[Dict[int, List[int]]] = None,
        penalty_capacity: float = 1.0,
        acceptance_criterion: Optional[IAcceptanceCriterion] = None,
        node_coords: Optional[np.ndarray] = None,
    ):
        """Initialize Local Search base class.

        Args:
            dist_matrix: N x N distance matrix.
            waste: Dictionary of node waste demands.
            capacity: Vehicle load capacity.
            R: Reward factor for collected waste.
            C: Cost factor for travel distance.
            params: Hyperparameters and configuration.
            neighbors: Precomputed granular neighbors. Defaults to None.
            penalty_capacity: Initial capacity violation penalty weight. Defaults to 1.0.
            acceptance_criterion: Custom acceptance strategy. Defaults to None.
            node_coords: (N, 2) array of node coordinates.
        """
        # Store matrix only if provided (O(n^2) memory)
        self.d = np.array(dist_matrix) if dist_matrix is not None else None

        # Store coordinates for on-demand calculation (O(n) memory)
        self.node_coords = node_coords
        if node_coords is not None:
            self.x_coords = node_coords[:, 0]
            self.y_coords = node_coords[:, 1]
        else:
            self.x_coords = np.array([])
            self.y_coords = np.array([])

        self.waste = waste
        self.Q = capacity
        self.R = R
        self.C = C
        self.params = params
        self.penalty_capacity = penalty_capacity
        self.random = random.Random(params.seed) if params.seed is not None else random.Random()
        self.mandatory_nodes = set(mandatory_nodes) if mandatory_nodes else set()

        # Modular Acceptance Criterion Injection
        self.acceptance_criterion = acceptance_criterion or getattr(params, "acceptance_criterion", None)
        if self.acceptance_criterion is None:
            # Fallback for base LocalSearch (elitist hill-climbing)
            from logic.src.policies.acceptance_criteria.base.factory import (
                AcceptanceCriterionFactory,
            )

            self.acceptance_criterion = AcceptanceCriterionFactory.create(name="oi")

        if neighbors is not None:
            self.neighbors = neighbors
        elif self.d is not None:
            # Traditional O(n^2) neighbor generation
            n_nodes = len(self.d)
            self.neighbors = {}
            nb_granular = getattr(params, "nb_granular", 20)
            for i in range(1, n_nodes):
                row = self.d[i]
                order = np.argsort(row)
                cands = [c for c in order if c not in (i, 0)][:nb_granular]
                self.neighbors[i] = cands
        else:
            raise ValueError("K-Sparse optimization requires precomputed neighbors if dist_matrix is None.")

        # Initialize caches
        self.node_map: Dict[int, Tuple[int, int]] = {}
        self.route_loads: List[float] = []
        self.routes: List[List[int]] = []
        self.top_insertions: Dict[int, Dict[int, List[Tuple[float, int]]]] = {}
        self.route_sectors: Dict[int, Optional[Tuple[float, float]]] = {}
        self.current_profit: float = 0.0
        self._target_neighborhood: str = "all"

    def get_dist(self, i: int, j: int) -> float:
        """Abstraction to fetch distance from matrix or calculate on-demand.

        This follows the MMAS_exp strategy of recomputing
        values to avoid intractable matrix sizes.

        Args:
            i: The index of the first node.
            j: The index of the second node.

        Returns:
            float: The distance between the two nodes.
        """
        if self.d is not None:
            return float(self.d[i, j])
        if self.node_coords is not None:
            return float(np.linalg.norm(self.node_coords[i] - self.node_coords[j]))
        raise RuntimeError("Neither distance matrix nor coordinates available.")

    def _compute_top_insertions(self, route_idx: Optional[int] = None):
        """Modified to use get_dist abstraction.

        Args:
            route_idx: The index of the route to compute the top insertions for.
                                     If None, computes for all routes.
        """
        routed = set(self.node_map.keys())  # O(1) set lookup below
        routes_to_update = [route_idx] if route_idx is not None else range(len(self.routes))
        for r_idx in routes_to_update:
            route = self.routes[r_idx]
            if not route:
                continue
            for node in self.neighbors:
                if node in routed:  # O(1) instead of O(route_length)
                    continue
                if node not in self.top_insertions:
                    self.top_insertions[node] = {}

                insertion_costs = []
                for pos in range(len(route) + 1):
                    prev = route[pos - 1] if pos > 0 else 0
                    nxt = route[pos] if pos < len(route) else 0
                    # Use get_dist to allow O(n) execution
                    delta = self.get_dist(prev, node) + self.get_dist(node, nxt) - self.get_dist(prev, nxt)
                    insertion_costs.append((delta, pos))

                insertion_costs.sort(key=lambda x: x[0])
                self.top_insertions[node][r_idx] = insertion_costs[:3]

    def _cost(self, routes: List[List[int]]) -> float:
        """Calculate total cost of all routes.

        Args:
            routes: List of routes, where each route is a list of node IDs.

        Returns:
            float: Total distance traveled across all routes, including returns to depot.
        """
        total = 0.0
        for r in routes:
            if not r:
                continue
            total += self.get_dist(0, r[0])
            for i in range(len(r) - 1):
                total += self.get_dist(r[i], r[i + 1])
            total += self.get_dist(r[-1], 0)
        return total

    @abstractmethod
    def optimize(self, solution: Any) -> Any:
        """Optimize the given solution.

        Args:
            solution: The initial solution to optimize.

        Returns:
            Any: The optimized solution.
        """
        pass

    def _optimize_internal(
        self,
        target_neighborhood: Optional[str] = None,
        active_nodes: Optional[Set[int]] = None,
        initial_profit: Optional[float] = None,
    ):
        """
        Core local search loop. Assumes self.routes is populated.

        Args:
            target_neighborhood: If provided, only applies the specified neighborhood operator.
                               If None or "all", applies all available operators.
            active_nodes: If provided, restricts search to moves involving at least one active node.
                        This enables O(1) localization per iteration (FILO).
            initial_profit: Optional absolute profit of the starting solution.
        """
        self.route_loads = [self._calc_load_fresh(r) for r in self.routes]
        if initial_profit is not None:
            self.current_profit = initial_profit
        else:
            # Fallback slow calculation (should be avoided in tight loops)
            self.current_profit = (
                sum(self.waste.get(n, 0) * self.R for r in self.routes for n in r) - self._cost(self.routes) * self.C
            )

        # Initialize node map and sector cache
        self.node_map.clear()
        self.route_sectors.clear()
        for ri, r in enumerate(self.routes):
            for pi, node in enumerate(r):
                self.node_map[node] = (ri, pi)

        # Store target neighborhood for _process_pair to filter operators
        self._target_neighborhood = target_neighborhood if target_neighborhood else "all"

        # Initialize Top-3 Insertion Cache for O(1) SWAP* evaluation (Vidal 2022)
        if any(n not in self.node_map for n in self.neighbors):
            self._compute_top_insertions()

        # Store active nodes set for localization (FILO)
        self._active_nodes = active_nodes

        improved = True
        pairs = [(u, v) for u in self.neighbors for v in self.neighbors[u]]  # Build the full candidate pair list
        it = 0
        t_start = time.perf_counter()
        max_ls_iters = getattr(self.params, "local_search_iterations", 500)
        time_limit = getattr(self.params, "time_limit", 0.0)
        while improved and it < max_ls_iters:
            if time_limit > 0 and time.perf_counter() - t_start > time_limit:
                break

            improved = False
            it += 1
            self.random.shuffle(pairs)
            for u, v in pairs:
                if self._active_nodes is not None and u not in self._active_nodes and v not in self._active_nodes:
                    continue
                if self._process_pair(u, v):
                    improved = True
                    break

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=it,
                n_routes=len(self.routes),
                improved=int(improved),
            )

    def _calc_load_fresh(self, r: List[int]) -> float:
        """Calculate total load of a route.

        Args:
            r: The sequence of node IDs in the route.

        Returns:
            float: Sum of waste for all nodes in the route.
        """
        return sum(self.waste.get(x, 0) for x in r)

    def _should_try_operator(self, op_name: str) -> bool:
        """Check if the operator should be tried based on target_neighborhood filter.

        Args:
            op_name: Name of the operator (e.g., 'relocate', 'swap').

        Returns:
            bool: True if the operator should be executed, False otherwise.
        """
        target = getattr(self, "_target_neighborhood", "all")
        return target == "all" or target == op_name
