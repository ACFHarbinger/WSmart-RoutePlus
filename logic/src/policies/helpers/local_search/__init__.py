"""
Local Search Package.

This package provides implementations of various local search algorithms for
vehicle routing problems, including:
- ACOLocalSearch: Ant Colony Optimization based local search.
- HGSLocalSearch: Hulman-Gobbi Simplex (HGS) based local search.
- LocalSearch: Base class for all local search implementations.
- SweepLocalSearch: Named first-improvement operator sweeps (route improvers).

Attributes:
    ACOLocalSearch: ACO-based local search implementation.
    HGSLocalSearch: HGS-based local search implementation.
    LocalSearch: Abstract base class for local search algorithms.
    SweepLocalSearch: Operator sweeps on the LocalSearch state.

Example:
    >>> from logic.src.policies.helpers.local_search import HGSLocalSearch
    >>> from logic.src.policies.wrappers.base import PolicyConfig
    >>> config = PolicyConfig()
    >>> hgs = HGSLocalSearch(config)
    >>> solution = hgs.search(best_solution, current_cost)
    >>> print(f"Improved cost: {solution.cost}")
"""

from .local_search_aco import ACOLocalSearch
from .local_search_base import LocalSearch
from .local_search_hgs import HGSLocalSearch
from .local_search_sweeps import SweepLocalSearch

__all__ = ["ACOLocalSearch", "HGSLocalSearch", "LocalSearch", "SweepLocalSearch"]
