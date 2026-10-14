"""Evaluator for Chance-Constrained Optimization (COP) subproblems.

Provides TSP cost precomputation and feasibility checking for the ESDP solver.

Attributes:
    COPEvaluator (class): Evaluator for routing costs and feasibility.

Example:
    >>> evaluator = COPEvaluator(dist_matrix, num_nodes=5)
    >>> cost = evaluator.get_route_cost(frozenset([1, 2]))
    >>> actions = evaluator.get_feasible_actions(state=(1, 2, 0), capacity=1.0, L=4)
"""

import itertools
from typing import Dict, FrozenSet, List, Tuple

import numpy as np


class COPEvaluator:
    """Evaluates and caches optimal routing costs for all possible subsets of nodes.

    Since V <= 10, exact evaluation using Held-Karp dynamic programming is O(V^2 2^V),
    which takes less than a second to cache all 2^V subsets.

    Attributes:
        dist_matrix (np.ndarray): Distance matrix between nodes.
        num_nodes (int): Total number of nodes in the graph.
        cost_cache (Dict[FrozenSet[int], float]): Cache mapping node subsets to TSP costs.
    """

    def __init__(self, dist_matrix: np.ndarray, num_nodes: int):
        """
        Initializes the COP evaluator.

        Args:
            dist_matrix: Distance matrix between nodes.
            num_nodes: Total number of nodes in the graph.
        """
        self.dist_matrix = dist_matrix
        self.num_nodes = num_nodes
        self.cost_cache: Dict[FrozenSet[int], float] = {}
        self._precompute_all_subsets()

        """Precomputes the TSP tour cost (starting and ending at depot 0)
        for every possible subset of customer nodes {1, ..., V-1}.

        Algorithm: Held-Karp dynamic programming.
        Complexity: O(V^2 2^V).
        """
        customers = list(range(1, self.num_nodes))

        # Base case: subset is empty -> cost is 0
        self.cost_cache[frozenset()] = 0.0

        # We will compute TSP recursively or iteratively for all subsets.
        # Held-Karp DP: C(S, j) = min_{i in S, i!=j} (C(S\{j}, i) + dist(i, j))
        # where C(S, j) is the min cost path visiting subset S ending at node j.
        # This requires storing (Subset, EndNode).

        hk_cache = {}
        # Base cases: path visiting only 1 customer ending at that customer
        for c in customers:
            s_set = frozenset([c])
            hk_cache[(s_set, c)] = self.dist_matrix[0, c]

            # The TSP cost from 0 -> c -> 0
            self.cost_cache[s_set] = float(self.dist_matrix[0, c] + self.dist_matrix[c, 0])

        # DP over subset sizes
        for size in range(2, self.num_nodes):
            for subset in itertools.combinations(customers, size):
                S = frozenset(subset)
                best_tour_cost = float("inf")

                for j in S:
                    S_minus_j = S - frozenset([j])
                    best_path_to_j = float("inf")
                    for i in S_minus_j:
                        cost = hk_cache[(S_minus_j, i)] + self.dist_matrix[i, j]
                        if cost < best_path_to_j:
                            best_path_to_j = cost

                    hk_cache[(S, j)] = best_path_to_j

                    # Update TSP completion (return to depot)
                    tour_cost = best_path_to_j + self.dist_matrix[j, 0]
                    if tour_cost < best_tour_cost:
                        best_tour_cost = tour_cost

                self.cost_cache[S] = float(best_tour_cost)

    def get_route_cost(self, subset: FrozenSet[int]) -> float:
        """Return pre-computed TSP cost for the selected node subset.

        Args:
            subset (FrozenSet[int]): Set of node indices.

        Returns:
            float: Pre-computed TSP cost.
        """
        return self.cost_cache.get(subset, float("inf"))

    def get_feasible_actions(self, discrete_state: Tuple[int, ...], capacity: float, L: int) -> List[FrozenSet[int]]:
        """Return all subsets of nodes whose total weight <= vehicle capacity.

        Since we operate in discretized fill levels, actual_weight approx = (level / (L-1)).

        Args:
            discrete_state (Tuple[int, ...]): Current discrete state tuple.
            capacity (float): Vehicle capacity.
            L (int): Number of discrete fill levels per bin.

        Returns:
            List[FrozenSet[int]]: List of feasible visit subsets.
        """
        customers = list(range(1, self.num_nodes))
        valid_actions: List[FrozenSet[int]] = [frozenset()]  # doing nothing is always valid

        # The weight of a bin i (1-indexed) is given by discrete_state[i-1] / (L-1)
        # assuming the 'capacity' is normalized or the fill level represents percentage
        # We assume max capacity is 1.0 in terms of normalized fill percent across the vehicle.

        for size in range(1, self.num_nodes):
            for subset in itertools.combinations(customers, size):
                S = frozenset(subset)
                # Ensure load fits in vehicle
                # 'capacity' refers to the vehicle's capacity in units (or %).
                # If fill level is [0,1], we sum them and check against capacity.
                total_load = sum(discrete_state[i - 1] / max(1, (L - 1)) for i in S)
                if total_load <= capacity + 1e-6:
                    valid_actions.append(S)

        return valid_actions
