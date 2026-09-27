"""
Particle class for the Particle Swarm Optimization Memetic Algorithm (PSOMA).

Attributes:
    PSOMAParticle: Single-particle class mapping continuous space to VRPP routes.

Example:
    >>> # Example instantiation assumes params and split_solver are defined
    >>> # clients = [1, 2, 3, 4]
    >>> # particle = PSOMAParticle(clients, params, split_solver)
"""

from typing import List, Optional, Tuple

import numpy as np

from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.split import LinearSplit
from logic.src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.params import (
    PSOMAParams,
)


class PSOMAParticle:
    """
    Continuous particle mapping to VRPP via ROV and Split.

    Attributes:
        n_clients (int): Number of clients.
        clients (np.ndarray): Array of client IDs.
        X (np.ndarray): Continuous position vector.
        V (np.ndarray): Continuous velocity vector.
        giant_tour (np.ndarray): Discrete permutation obtained via ROV.
        routes (List[List[int]]): Decoded phenotype routes.
        profit (float): Objective value.
        pbest_X (np.ndarray): Personal best continuous position.
        pbest_giant_tour (np.ndarray): Personal best discrete permutation.
        pbest_profit (float): Personal best objective value.
    """

    def __init__(
        self,
        clients: List[int],
        params: PSOMAParams,
        split_solver: LinearSplit,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initializes a continuous PSOMA particle.

        Args:
            clients: List of node indices (excluding depot).
            params: PSOMA hyperparameters (bounds, velocities).
            split_solver: Initialized LinearSplit decoder instance.
            rng: Seeded generator for the initial position and velocity (a fresh one if None).
        """
        self.n_clients = len(clients)
        self.clients = np.array(clients)

        # Continuous state
        rng = rng if rng is not None else np.random.default_rng()
        self.X = rng.uniform(params.x_min, params.x_max, self.n_clients)
        self.V = rng.uniform(params.v_min, params.v_max, self.n_clients)

        # Discrete state mapping
        self.mapping_indices = np.argsort(self.X)
        self.giant_tour = self.clients[self.mapping_indices]

        # Phenotype (Routes) and Fitness
        self.routes, self.profit = split_solver.split(self.giant_tour.tolist())

        # Personal bests
        self.pbest_X = np.copy(self.X)
        self.pbest_giant_tour = np.copy(self.giant_tour)
        self.pbest_profit = self.profit
        self.pbest_mapping_indices = np.copy(self.mapping_indices)

    def _rov_rule(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ranked-Order-Value mapping. Maps the continuous vector to a permutation.

        Args:
            X (np.ndarray): Continuous position vector.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - np.ndarray: Discrete permutation of client IDs.
                - np.ndarray: Mapping indices.
        """
        mapping = np.argsort(X)
        tour = self.clients[mapping]
        return tour, mapping
