"""Individual representation for the (μ,κ,λ) Evolution Strategy.

Attributes:
    Individual: Data structure for ES individuals with age tracking.

Example:
    >>> ind = Individual(routes=[[1, 2, 0]])
"""

import copy
from typing import List

import numpy as np


class Individual:
    """
    An individual in the ES population representing a routing solution.

    In accordance with self-adaptive ES principles, the individual carries both
    the object variables (the routes) and the strategy parameters (the
    mutation_strength).

    Attributes:
        routes: The discrete solution representation (set of routes).
        mutation_strength: The self-adaptive 'step-size'.
        fitness: The objective value (net profit).
        age: The number of generations this individual has survived.
    """

    def __init__(
        self,
        routes: List[List[int]],
        fitness: float = -np.inf,
        age: int = 1,
        mutation_strength: float = 3.0,
    ):
        """
        Initialize a new Individual.

        Args:
            routes: Initial routing sequences.
            fitness: Initial fitness value.
            age: Starting age (default 1).
            mutation_strength: Initial mutation intensity (discrete σ).

        Returns:
            None.
        """
        self.routes = copy.deepcopy(routes)
        self.fitness = fitness
        self.age = age
        self.mutation_strength = mutation_strength

    def copy(self) -> "Individual":
        """
        Creates a deep copy of the individual for independent variation.

        Returns:
            A new Individual instance with identical attributes.
        """
        return Individual(
            routes=copy.deepcopy(self.routes),
            fitness=self.fitness,
            age=self.age,
            mutation_strength=self.mutation_strength,
        )
