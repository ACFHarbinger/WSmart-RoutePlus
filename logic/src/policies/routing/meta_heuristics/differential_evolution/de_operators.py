r"""Continuous Differential Evolution operators for the DE solver.

Moved unchanged from :mod:`solver`: the mixin with the vectorized mutation
index generation, exponential and binomial crossovers, differential
mutation and bounce-back boundary handling (plus the legacy scalar
variants kept for reference).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from logic.src.configs.policies.routing.meta_heuristics.de import DEConfig


class DEOperatorsMixin:
    r"""Continuous DE mutation, crossover and boundary-handling operators."""

    # Provided by DESolver.__init__; annotations only.
    rng: np.random.RandomState
    params: DEConfig
    n_nodes: int

    def _generate_mutation_indices(self, pop_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pure NumPy implementation of 3 mutually exclusive random indices for all i.

        Ensures r1[i] != r2[i] != r3[i] != i for all i in [0, pop_size) using
        offset-based modulo arithmetic, avoiding sequential Python loops.

        Args:
            pop_size: The size of the population.

        Returns:
            Tuple of three NumPy arrays (r1, r2, r3).
        """
        # Generate 3 distinct offsets in [1, pop_size-1] for each individual i.
        # By adding these offsets to i modulo pop_size, we guarantee r_j != i.
        # By ensuring offsets are unique per i, we guarantee r1 != r2 != r3.
        num_offsets = pop_size - 1
        offset_perms = np.argsort(self.rng.rand(pop_size, num_offsets), axis=1) + 1

        # Select first 3 random offsets
        offsets = offset_perms[:, :3]

        # Compute final indices using broadcasting and modulo
        rows = np.arange(pop_size)[:, None]
        r_indices = (rows + offsets) % pop_size

        return r_indices[:, 0], r_indices[:, 1], r_indices[:, 2]

    def _vectorized_exponential_crossover(self, target_pop: np.ndarray, mutant_pop: np.ndarray) -> np.ndarray:
        """Fully vectorized implementation of Exponential Crossover.

        Replaces the sequential population loop with a single matrix operation
        using 2D Boolean masks and probabilistic sequence length calculation.

        Args:
            target_pop: The current population matrix.
            mutant_pop: The mutated population matrix.

        Returns:
            The trial population matrix.
        """
        NP, D = target_pop.shape
        CR = self.params.crossover_rate

        # 1. Select starting indices n for each individual
        n_starts = self.rng.randint(0, D, size=NP)

        # 2. Simulate the do...while loop with a Geometric Stop
        # P(L=k) = CR^(k-1) * (1-CR) for k in [1, D]
        # In NP, this is equivalent to generating masks from random values.
        # Stop probability is (1 - CR).
        L = self.rng.geometric(1 - CR, size=NP)
        L = np.clip(L, 1, D)

        # 3. Create the crossover mask
        # For each i, mutant contributes indices [n[i], n[i] + L[i]) modulo D.
        idx_grid = np.arange(D)
        # Calculate circular distance: (j - n_starts[i]) % D
        dist = (idx_grid - n_starts[:, None]) % D
        mask = dist < L[:, None]

        return np.where(mask, mutant_pop, target_pop)

    def _apply_boundary_handling(self, vector: np.ndarray, base_vector: np.ndarray) -> np.ndarray:
        """Apply bounce-back boundary handling to preserve genetic diversity.

        Args:
            vector: The continuous vector to bound.
            base_vector: The reference vector (mutant base or target parent).

        Returns:
            Bounded vector strictly within [-1.0, 1.0].
        """
        # Lower bound violation
        low_mask = vector < -1.0
        if np.any(low_mask):
            # x_new = base + rand * (boundary - base)
            vector[low_mask] = base_vector[low_mask] + self.rng.rand(np.sum(low_mask)) * (-1.0 - base_vector[low_mask])  # type: ignore[call-overload]

        # Upper bound violation
        high_mask = vector > 1.0
        if np.any(high_mask):
            vector[high_mask] = base_vector[high_mask] + self.rng.rand(np.sum(high_mask)) * (  # type: ignore[call-overload]
                1.0 - base_vector[high_mask]
            )

        return vector

    def _exponential_crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """Perform exponential crossover (DE/rand/1/exp).

        Args:
            target: Target vector.
            mutant: Mutant vector.

        Returns:
            The trial vector.
        """
        trial = target.copy()
        n = self.rng.randint(0, self.n_nodes)
        L = 0
        while True:
            trial[n] = mutant[n]
            n = (n + 1) % self.n_nodes
            L += 1
            if self.n_nodes <= L or self.rng.rand() >= self.params.crossover_rate:
                break
        return trial

    def _binomial_crossover(self, target: np.ndarray, mutant: np.ndarray, CR: float) -> np.ndarray:
        """Binomial crossover operator with domain enforcement.

        Args:
            target: Target vector (x_i).
            mutant: Mutant vector (v_i).
            CR: Crossover probability [0, 1].

        Returns:
            Trial vector (u_i), clipped to domain [-1.0, 1.0].
        """
        trial = np.copy(target)

        # Ensure at least one dimension comes from mutant
        j_rand = self.rng.randint(0, self.n_nodes)

        for j in range(self.n_nodes):
            if j == j_rand or self.rng.random() < CR:
                trial[j] = mutant[j]

        # Enforce domain bounds to prevent vector explosion
        return np.clip(trial, -1.0, 1.0)
