"""
Reactive Tabu Search (RTS) solver for VRPP.

Attributes:
    RTSSolver: The main solver class.
    RTSParams: Configuration parameters dataclass.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.reactive_tabu_search import RTSSolver, RTSParams
    >>> params = RTSParams()
    >>> solver = RTSSolver(params)
    >>> solution = solver.solve()
    >>> print(solution)
"""

from .params import RTSParams
from .solver import RTSSolver

__all__ = ["RTSSolver", "RTSParams"]
