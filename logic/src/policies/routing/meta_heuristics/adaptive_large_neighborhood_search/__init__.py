r"""Auxiliary modules for the Adaptive Large Neighborhood Search (ALNS) policy.

Attributes:
    ALNSParams: Dataclass for ALNS configuration.
    run_alns: Main dispatcher for ALNS.
    run_alns_ortools: OR-Tools based ALNS implementation.
    run_alns_package: Package-based ALNS implementation.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search import ALNSParams
    >>> params = ALNSParams()
"""

from .alns_package import run_alns_package
from .dispatcher import run_alns
from .ortools_wrapper import run_alns_ortools
from .params import ALNSParams

__all__ = ["ALNSParams", "run_alns", "run_alns_ortools", "run_alns_package"]
