"""
Auxiliary modules for the Hybrid Genetic Search (HGS) policy.

Attributes:
    run_hgs: Entry point function for running the HGS solver.
    Individual: Class representing an individual in the population.
    HGSParams: Configuration parameters for the HGS algorithm.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.hybrid_genetic_search import HGSParams
    >>> params = HGSParams()
"""

from .dispatcher import run_hgs
from .individual import Individual
from .params import HGSParams

__all__ = ["run_hgs", "Individual", "HGSParams"]
