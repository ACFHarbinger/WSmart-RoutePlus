r"""Random Key encoding and decoding for the Memetic Differential Evolution solver.

Moved unchanged from :mod:`solver`: the mixin that seeds and initializes the
continuous population and converts between discrete VRPP routes and
continuous Random Key vectors.
"""

from __future__ import annotations

from typing import List

import numpy as np

from logic.src.policies.helpers.operators.solution_initialization.greedy_si import build_greedy_routes


class DERandomKeyMixin:
    r"""Random Key population initialization, encoding and decoding for DE."""

    def _initialize_population(self) -> np.ndarray:
        """Initialize population as continuous Random Key vectors.

        Returns:
            Population matrix of shape (pop_size, n_nodes) in [-1.0, 1.0].
        """
        population = np.zeros((self.pop_size, self.n_nodes))

        # Generate base greedy solution
        greedy_routes = build_greedy_routes(
            dist_matrix=self.dist_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.py_rng,
        )
        greedy_vector = self._encode_routes(greedy_routes)

        # Seed first 10% of population (minimum 1) with Gaussian deviations
        n_seeded = max(1, self.pop_size // 10)
        for i in range(n_seeded):
            # Storn & Price (1997) suggest Gaussian noise for preliminary solutions
            noise = self.rng.normal(0, 0.1, size=self.n_nodes)
            population[i] = np.clip(greedy_vector + noise, -1.0, 1.0)

        # Initialize remaining population uniformly at random
        for i in range(n_seeded, self.pop_size):
            population[i] = self.rng.uniform(-1.0, 1.0, size=self.n_nodes)

        return population

    def _encode_routes(self, routes: List[List[int]]) -> np.ndarray:
        """Encode discrete routes into a continuous Random Key vector with noise injection.

        Args:
            routes: Discrete routing solution.

        Returns:
            Continuous vector of shape (n_nodes,) in [-1.0, 1.0].
        """
        # Initialize all nodes with random negative values (unvisited representation)
        # Distributing across [-1.0, -0.01] prevents genetic uniformity
        new_vector = self.rng.uniform(-1.0, -0.01, size=self.n_nodes)

        # Process each route independently to maintain route structure
        for route in routes:
            if not route:
                continue

            n_stops = len(route)

            # Generate linearly spaced base values in safe positive domain [0.1, 0.9]
            # Avoiding extreme boundaries [0.0, 1.0] gives noise room to operate
            base_values = np.linspace(0.1, 0.9, n_stops)

            # Inject uniform noise to prevent identical encodings
            # Noise magnitude [-0.05, 0.05] is small enough to preserve sequence order
            # but large enough to maintain population diversity
            noise = self.rng.uniform(-0.05, 0.05, size=n_stops)

            # Assign combined base + noise to visited nodes
            for idx, node in enumerate(route):
                vector_idx = node - 1  # Convert 1-indexed node to 0-indexed array position
                # Clip to [0.0, 1.0] to handle edge cases where noise pushes values out
                new_vector[vector_idx] = np.clip(base_values[idx] + noise[idx], 0.0, 1.0)

        return new_vector

    def _decode_vector(self, vector: np.ndarray) -> List[List[int]]:
        """Decode continuous Random Key vector into discrete VRPP routes.

        Args:
            vector: Continuous vector of shape (n_nodes,) in [-1.0, 1.0].

        Returns:
            Discrete routing solution satisfying capacity constraints.
        """
        # Step 1: Node selection
        selected_nodes = []
        for i in range(self.n_nodes):
            node_id = i + 1  # Convert 0-indexed to 1-indexed
            if vector[i] >= 0.0 or node_id in self.mandatory_nodes:
                selected_nodes.append((vector[i], node_id))

        if not selected_nodes:
            return []

        # Step 2: Sort by Random Key values (ascending)
        selected_nodes.sort(key=lambda x: x[0])
        sorted_nodes = [node_id for _, node_id in selected_nodes]

        # Step 3: Route packing with capacity constraints
        routes = []
        current_route = []
        current_load = 0.0

        for node in sorted_nodes:
            node_waste = self.wastes.get(node, 0.0)

            # Check if adding node would violate capacity
            if current_load + node_waste <= self.capacity:
                current_route.append(node)
                current_load += node_waste
            else:
                # Start new route if current route is not empty
                if current_route:
                    routes.append(current_route)
                # Start fresh with this node
                current_route = [node]
                current_load = node_waste

        # Add final route if not empty
        if current_route:
            routes.append(current_route)

        return routes
