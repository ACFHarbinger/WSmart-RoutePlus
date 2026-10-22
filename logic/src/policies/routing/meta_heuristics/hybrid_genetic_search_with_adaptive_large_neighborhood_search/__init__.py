"""
Hybrid Genetic Search with Adaptive Large Neighborhood Search (HGS-ALNS).

Combines HGS evolutionary operators with ALNS-based education phase for
intensive local search optimization.

Attributes:
    HGSALNSSolver: Core solver class.
    HGSALNSConfig: Configuration parameters.

"""

from logic.src.configs.policies.routing.meta_heuristics.hgs_alns import HGSALNSConfig

from .hgs_alns import HGSALNSSolver

__all__ = ["HGSALNSSolver", "HGSALNSConfig"]
