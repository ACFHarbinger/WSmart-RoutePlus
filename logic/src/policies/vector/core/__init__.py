"""Hybrid Genetic Search (HGS) core components.

This package contains the modular building blocks for the Hybrid Genetic Search
metaheuristic, optimized for vectorized execution on GPU hardware. It includes
population management, genetic crossover operators, and diversity metrics.

Attributes:
    VectorizedPopulation: Class managing a diverse pool of elite solutions.
    vectorized_ordered_crossover: GPU-optimized ordered crossover (OX1).
    calc_broken_pairs_distance: Diversity metric based on edge preservation.

Example:
    >>> from logic.src.policies.vector.core import VectorizedPopulation
    >>> pop = VectorizedPopulation(size=50, device="cuda")
"""

from .crossover import vectorized_ordered_crossover
from .evaluation import calc_broken_pairs_distance
from .population import VectorizedPopulation

__all__ = [
    "VectorizedPopulation",
    "vectorized_ordered_crossover",
    "calc_broken_pairs_distance",
]
