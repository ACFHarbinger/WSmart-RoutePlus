"""
Auxiliary modules for the Hybrid Genetic Search (HGS) policy.

Attributes:
    run_hgs: Entry point function for running the HGS solver.
    Individual: Class representing an individual in the population.
    HGSSolver: Main solver class for HGS.

"""

from .dispatcher import run_hgs
from .hgs import HGSSolver
from .individual import Individual

__all__ = ["HGSSolver", "run_hgs", "Individual"]
