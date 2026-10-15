r"""Solution Construction Module for K-Sparse ACO.

This module implements the solution construction phase of the MMAS algorithm.
Ants construct solutions using pure roulette-wheel proportional selection
based on pheromone levels and heuristic information.

Attributes:
    SolutionConstructor: Constructs a single solution (route set) for an ant.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.ant_colony_optimization_k_sparse.construction import SolutionConstructor
    >>> constructor = SolutionConstructor(...)
    >>> routes = constructor.construct()

Reference:
    Hale, D. "Investigation of Ant Colony Optimization Implementation
    Strategies For Low-Memory Operating Environments", 2021.
    Uses MMAS_exp transition rule (no local updates, no q0 exploitation).
"""

import random
from typing import Any, Dict, List, Optional, Set

import numpy as np

from logic.src.configs.policies.aco_ks import KSparseACOConfig
from .pheromones import SparsePheromoneTau


class SolutionConstructor:
    """Constructs a single solution (route set) for an ant.

    Uses the k-sparse pheromone matrix and heuristic values for selection.

    Attributes:
        node_coords: Array of node coordinates.
        wastes: Node fill levels.
        capacity: Vehicle capacity.
        pheromone: Sparse pheromone matrix.
        candidate_lists: K-sparse candidate neighbor lists.
        nodes: List of customer nodes.
        params: Algorithm parameters.
        tau_0: Initial pheromone level.
        R: Revenue multiplier.
        C: Cost multiplier.
        mandatory_nodes: Nodes that must be visited.
        random: Thread-safe random number generator.
    """

    def __init__(
        self,
        node_coords: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        pheromone: SparsePheromoneTau,
        candidate_lists: Dict[int, List[int]],
        nodes: List[int],
        params: KSparseACOConfig,
        tau_0: float,
        R: float = 0.0,
        C: float = 1.0,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """Initializes the Solution Constructor for ACO ants.

        Args:
            node_coords (np.ndarray): Array of node coordinates.
            wastes (Dict[int, float]): Mapping of bin IDs to waste quantities.
            capacity (float): Maximum vehicle collection capacity.
            pheromone (SparsePheromoneTau): Sparse pheromone matrix.
            candidate_lists (Dict[int, List[int]]): K-nearest neighbor lists.
            nodes (List[int]): List of all nodes (excluding depot).
            params (KSparseACOConfig): Algorithm-specific parameters.
            tau_0 (float): Initial pheromone value.
            R (float): Revenue multiplier.
            C (float): Cost multiplier.
            mandatory_nodes (Optional[List[int]]): Nodes that must be visited.
        """
        self.node_coords = node_coords
        self.wastes = wastes
        self.capacity = capacity
        self.pheromone = pheromone
        self.candidate_lists = candidate_lists
        self.nodes = nodes
        self.params = params
        self.tau_0 = tau_0
        self.R = R
        self.C = C
        self.mandatory_nodes = set(mandatory_nodes) if mandatory_nodes else set()
        self.random = random.Random(params.seed) if params.seed is not None else random.Random()

    def construct(self) -> List[List[int]]:
        """Construct a solution using the MMAS proportional transition rule.

        This method uses pure exploration (roulette-wheel selection) without
        any local pheromone updates or exploitation bias (q0). All pheromone
        updates occur globally after construction.

        Returns:
            List of routes, each a list of node indices.
        """
        unvisited: Set[int] = set(self.nodes)
        mandatory_unvisited = set(self.mandatory_nodes)
        routes: List[List[int]] = []

        # We continue as long as there are mandatory nodes OR if we want to visit more
        # In VRPP mode, if mandatory_unvisited is empty, we only continue if profitable nodes exist.
        while unvisited:
            if not mandatory_unvisited and not self._any_profitable_nodes(unvisited):
                break

            route: List[int] = []
            load = 0.0
            current = 0  # Start from depot

            while unvisited:
                # Find feasible next nodes
                feasible = self._get_feasible_nodes(unvisited, mandatory_unvisited, load, current)

                if not feasible:
                    self._cleanup_unvisited(unvisited, mandatory_unvisited)
                    break

                # Select next node using MMAS proportional rule
                next_node = self._select_next_node(current, sorted(feasible))

                # No local pheromone update in MMAS (removed ACS mechanics)

                route.append(next_node)
                load += self.wastes.get(next_node, 0)
                unvisited.remove(next_node)
                if next_node in mandatory_unvisited:
                    mandatory_unvisited.remove(next_node)
                current = next_node

            if route:
                routes.append(route)

        return routes

    def _cleanup_unvisited(self, unvisited: Set[int], mandatory_unvisited: Set[int]) -> None:
        """Remove nodes that can never fit in any route to avoid infinite loops.

        Args:
            unvisited: Set of nodes to potentially clean up.
            mandatory_unvisited: Set of mandatory nodes to clean up.

        Returns:
            None.
        """
        still_constructible = False
        for j in list(unvisited):
            if self.wastes.get(j, 0) <= self.capacity:
                still_constructible = True
            else:
                unvisited.remove(j)
                if j in mandatory_unvisited:
                    mandatory_unvisited.remove(j)

        if not still_constructible:
            unvisited.clear()

    def _dist(self, i: int, j: int) -> np.floating[Any]:
        """Compute exact Euclidean distance on-demand.

        Args:
            i: Index of the first node.
            j: Index of the second node.

        Returns:
            Euclidean distance between nodes i and j.
        """
        return np.linalg.norm(self.node_coords[i] - self.node_coords[j])

    def _any_profitable_nodes(self, unvisited: Set[int]) -> bool:
        """Check if there are any profitable nodes remaining.

        Args:
            unvisited: Set of unvisited nodes.

        Returns:
            bool: True if there are profitable nodes, False otherwise.
        """
        if not self.params.vrpp:
            return True

        for j in sorted(unvisited):
            revenue = self.wastes.get(j, 0) * self.R
            dist_j = self._dist(0, j)
            if revenue > (dist_j * 2) * self.C:
                return True
        return False

    def _get_feasible_nodes(
        self, unvisited: Set[int], mandatory_unvisited: Set[int], load: float, current: int
    ) -> List[int]:
        """Get a list of feasible nodes that can be added to the current route.

        Args:
            unvisited: Set of unvisited nodes.
            mandatory_unvisited: Set of mandatory unvisited nodes.
            load: Current load of the vehicle.
            current: Current node.

        Returns:
            List of feasible nodes.
        """
        feasible = []
        use_profit_check = self.params.vrpp and self.params.profit_aware_operators
        for j in sorted(unvisited):
            if load + self.wastes.get(j, 0) <= self.capacity:
                if j in mandatory_unvisited:
                    feasible.append(j)
                elif use_profit_check:
                    revenue = self.wastes.get(j, 0) * self.R
                    marginal_cost = (self._dist(current, j) + self._dist(j, 0) - self._dist(current, 0)) * self.C
                    if revenue > marginal_cost:
                        feasible.append(j)
                else:
                    feasible.append(j)
        return feasible

    def _select_next_node(self, current: int, feasible: List[int]) -> int:
        """
        Select next node using MMAS proportional (roulette-wheel) selection.

        This is pure exploration without exploitation bias. The probability
        of selecting node j is proportional to:
            P(j) = [tau(current, j)^alpha] * [eta(current, j)^beta]

        Candidate lists are used to bias selection toward nearby neighbors
        when available, following the k-sparse optimization strategy.

        Args:
            current: Current node index.
            feasible: List of feasible next nodes.

        Returns:
            Selected next node index.
        """
        # Prefer candidates if available (k-nearest neighbors)
        candidates_in_feasible = [j for j in self.candidate_lists.get(current, []) if j in feasible]
        selection_pool = candidates_in_feasible if candidates_in_feasible else feasible

        if not selection_pool:
            return -1  # Failsafe

        # 1. Vectorized Distance Computation
        pool_array = np.array(selection_pool)
        current_coord = self.node_coords[current]
        target_coords = self.node_coords[pool_array]

        # Calculate all distances in C-backend simultaneously
        diffs = target_coords - current_coord
        dists = np.linalg.norm(diffs, axis=1)

        # 2. Vectorized Eta Computation (avoid div by zero)
        etas = np.zeros_like(dists)
        valid_mask = dists > 0
        etas[valid_mask] = 1.0 / dists[valid_mask]

        # 3. Fetch Taus (requires Python iteration due to Sparse dict structure)
        taus = np.array([self.pheromone.get(current, j) for j in selection_pool])

        # 4. Vectorized Probability Calculation
        probs = (taus**self.params.alpha) * (etas**self.params.beta)
        total = np.sum(probs)

        if total <= 0:
            return self.random.choice(selection_pool)

        # Vectorized Roulette Wheel
        r = self.random.uniform(0, total)
        cumsum = np.cumsum(probs)
        chosen_idx = np.searchsorted(cumsum, r)

        # searchsorted can occasionally return out of bounds due to float precision
        chosen_idx = min(chosen_idx, len(selection_pool) - 1)

        return selection_pool[chosen_idx]
