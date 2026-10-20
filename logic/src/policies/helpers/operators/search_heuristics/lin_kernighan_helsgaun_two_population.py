"""Tour population management with diversity tracking for LKH-2."""

from __future__ import annotations

from typing import List, Set, Tuple

import numpy as np


class TourPopulation:
    """
    Fixed-size population of locally-optimal tours for LKH-2.

    Maintains:
    - A list of (tour, cost) pairs, at most ``max_size`` members.
    - The best solution (lowest cost) seen so far.
    - Diversity tracking via edge-set Hamming distance.

    Insertion policy:
    - A new tour is accepted if it is better than the current worst member
      **or** if its minimum edge-Hamming distance to all current members
      exceeds ``diversity_threshold * n`` (i.e., it is sufficiently novel).
    - When the population is full and the new tour is accepted, the worst
      member is evicted.

    Attributes:
        max_size (int): Maximum population size.
        diversity_threshold (float): Minimum fractional Hamming distance.
        members (List[Tuple[List[int], float]]): (tour, cost) pairs.
        edge_sets (List[Set[Tuple[int, int]]]): Sorted-pair edge sets.
        best_tour (List[int]): Best solution found.
        best_cost (float): Best cost found.

    Example:
        >>> pop = TourPopulation(max_size=10)
        >>> pop.try_insert(tour, cost)
    """

    def __init__(self, max_size: int = 10, diversity_threshold: float = 0.05):
        """
        Args:
            max_size: Maximum population size.
            diversity_threshold: Minimum fractional Hamming distance (0..1)
                from all existing members for a new tour to be accepted on
                diversity grounds.

        """
        self.max_size = max_size
        self.diversity_threshold = diversity_threshold
        self.members: List[Tuple[List[int], float]] = []  # (tour, cost)
        self.edge_sets: List[Set[Tuple[int, int]]] = []  # sorted-pair edge sets
        self.best_tour: List[int] = []
        self.best_cost: float = float("inf")

    # ------------------------------------------------------------------
    # Edge-set helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_edge_set(tour: List[int]) -> Set[Tuple[int, int]]:
        """
        Convert a closed tour to a set of sorted (i,j) edge pairs.

        Args:
            tour: Closed tour node sequence.

        Returns:
            Set[Tuple[int, int]]: Set of edges, where each edge is a sorted tuple.
        """
        es: Set[Tuple[int, int]] = set()
        for k in range(len(tour) - 1):
            a, b = tour[k], tour[k + 1]
            es.add((min(a, b), max(a, b)))
        return es

    def _hamming_distance(self, es_new: Set[Tuple[int, int]], n: int) -> float:
        """
        Minimum fractional Hamming distance from ``es_new`` to any existing member.

        Hamming(T1, T2) / n ∈ [0, 1], where Hamming counts edges in the
        symmetric difference.

        Args:
            es_new: Edge set of the new tour.
            n: Number of nodes in the problem instance.

        Returns:
            float: Minimum fractional Hamming distance to current population.
        """
        if not self.edge_sets:
            return 1.0
        min_dist = 1.0
        for es in self.edge_sets:
            sym_diff = len(es_new.symmetric_difference(es))
            dist = sym_diff / max(n, 1)
            min_dist = min(min_dist, dist)
        return min_dist

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def try_insert(self, tour: List[int], cost: float) -> bool:
        """
        Attempt to insert a new tour into the population.

        Args:
            tour: Closed node sequence.
            cost: Tour length (raw, unpenalized).

        Returns:
            bool: True if the tour was inserted; False otherwise.
        """
        n = len(tour) - 1
        es_new = self._to_edge_set(tour)

        # Update global best
        if cost < self.best_cost - 1e-9:
            self.best_cost = cost
            self.best_tour = tour[:]

        if len(self.members) < self.max_size:
            # Population not yet full: always accept
            self.members.append((tour[:], cost))
            self.edge_sets.append(es_new)
            return True

        # Population full: accept if better than worst or sufficiently diverse
        worst_idx = max(range(len(self.members)), key=lambda i: self.members[i][1])
        worst_cost = self.members[worst_idx][1]

        hamming = self._hamming_distance(es_new, n)
        is_better_than_worst = cost < worst_cost - 1e-9
        is_diverse = hamming >= self.diversity_threshold

        if is_better_than_worst or is_diverse:
            self.members[worst_idx] = (tour[:], cost)
            self.edge_sets[worst_idx] = es_new
            return True
        return False

    def sample_parents(self, rng: np.random.Generator, n_parents: int = 2) -> List[Tuple[List[int], float]]:
        """
        Sample parent tours from the population for crossover.

        Selection picks the maximum-diversity pair from the population:
        - Draw a candidate pool via fitness-proportionate sampling (lower cost
          → higher probability).
        - From that pool, select the pair that maximises their mutual
          edge-Hamming distance, encouraging structural diversity in offspring.

        Args:
            rng: NumPy Generator.
            n_parents: Number of parents to sample.

        Returns:
            List[Tuple[List[int], float]]: List of (tour, cost) pairs.
        """
        if len(self.members) <= n_parents:
            return list(self.members)

        if n_parents == 2 and len(self.members) >= 3:
            # Fitness-proportionate candidate pool (sample min(2*pop, all) members)
            costs = np.array([c for _, c in self.members])
            max_c = costs.max()
            weights = max_c - costs + 1e-9
            weights /= weights.sum()
            pool_size = min(len(self.members), max(4, 2 * n_parents))
            pool_indices = list(rng.choice(len(self.members), size=pool_size, replace=False, p=weights))

            # Select pair with maximum Hamming distance from the candidate pool
            best_pair = (pool_indices[0], pool_indices[1])
            best_dist = 0.0
            for ii in range(len(pool_indices)):
                for jj in range(ii + 1, len(pool_indices)):
                    idx_i = pool_indices[ii]
                    idx_j = pool_indices[jj]
                    es_i = self.edge_sets[idx_i]
                    es_j = self.edge_sets[idx_j]
                    n_nodes = len(self.members[idx_i][0]) - 1
                    dist = len(es_i.symmetric_difference(es_j)) / max(n_nodes, 1)
                    if dist > best_dist:
                        best_dist = dist
                        best_pair = (idx_i, idx_j)

            return [self.members[i] for i in best_pair]

        # General case: fitness-proportionate weighted sample
        costs = np.array([c for _, c in self.members])
        max_c = costs.max()
        weights = max_c - costs + 1e-9
        weights /= weights.sum()
        indices = rng.choice(len(self.members), size=n_parents, replace=False, p=weights)
        return [self.members[i] for i in indices]

    @property
    def size(self) -> int:
        """Current number of members in the population.

        Returns:
            int: Population size.
        """
        return len(self.members)

    @property
    def diversity(self) -> float:
        """Mean pairwise fractional Hamming distance across population.

        Returns:
            float: Population diversity.
        """
        if len(self.members) < 2:
            return 1.0
        n = len(self.members[0][0]) - 1
        total = 0.0
        count = 0
        for i in range(len(self.edge_sets)):
            for j in range(i + 1, len(self.edge_sets)):
                total += len(self.edge_sets[i].symmetric_difference(self.edge_sets[j])) / max(n, 1)
                count += 1
        return total / max(count, 1)
