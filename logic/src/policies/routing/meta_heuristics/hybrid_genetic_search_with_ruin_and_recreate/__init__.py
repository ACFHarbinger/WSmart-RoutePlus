"""
Hybrid Genetic Search with Ruin-and-Recreate (HGS-RR) policy module.

This module combines the evolutionary framework of HGS with the adaptive
destroy/repair operators from ALNS (Ruin-and-Recreate paradigm).

Reference:
    Pisinger, D., & Ropke, S. (2019). Large neighborhood search.
    In Handbook of metaheuristics (pp. 99-127). Springer, Cham.

Attributes:
    HGSRRSolver: High-performance hybrid evolutionary solver.
    HGSRRConfig: Configuration parameters for the HGS-RR solver.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.hybrid_genetic_search_with_ruin_and_recreate import HGSRRSolver
    >>> solver = HGSRRSolver(dist_matrix, wastes, capacity, R, C, params)
"""

from .hgs_rr import HGSRRSolver
from logic.src.configs.policies.hgs_rr import HGSRRConfig

__all__ = ["HGSRRSolver", "HGSRRConfig"]
