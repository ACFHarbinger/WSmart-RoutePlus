"""
Search sub-package export for Simulated Annealing Neighborhood Search (SANS).

The deterministic relocate passes moved to
``logic.src.policies.helpers.operators.improvement_descent.relocate_descent``.

Attributes:
    local_search: Apply one randomly chosen LAC perturbation to a solution.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.search import local_search
"""

from .random_search import local_search

__all__ = ["local_search"]
