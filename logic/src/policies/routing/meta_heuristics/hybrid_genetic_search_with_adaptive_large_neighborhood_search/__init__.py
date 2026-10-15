"""
Hybrid Genetic Search with Adaptive Large Neighborhood Search (HGS-ALNS).

Combines HGS evolutionary operators with ALNS-based education phase for
intensive local search optimization.

Attributes:
    HGSALNSSolver: Core solver class.
    HGSALNSConfig: Configuration parameters.

Example:
    >>> from .hgs_alns import HGSALNSSolver
"""

from .hgs_alns import HGSALNSSolver
from logic.src.configs.policies.hgs_alns import HGSALNSConfig

__all__ = ["HGSALNSSolver", "HGSALNSConfig"]
