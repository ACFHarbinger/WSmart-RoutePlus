"""
Reactive Tabu Search (RTS) solver for VRPP.

Attributes:
    RTSSolver: The main solver class.
    RTSConfig: Configuration parameters dataclass.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.reactive_tabu_search import RTSSolver, RTSConfig
    >>> params = RTSConfig()
    >>> solver = RTSSolver(params)
    >>> solution = solver.solve()
    >>> print(solution)
"""

from logic.src.configs.policies.rts import RTSConfig

from .solver import RTSSolver

__all__ = ["RTSSolver", "RTSConfig"]
