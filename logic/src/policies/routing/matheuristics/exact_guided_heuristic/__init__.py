r"""TCF → ALNS → BPC → SP-merge pipeline package.

Attributes:
----------
ExactGuidedHeuristicPolicy   Policy adapter (registers as "exact_guided_heuristic" in the solver registry).
ExactGuidedHeuristicParams   Configuration dataclass with ``alpha`` quality/speed dial.
run_pipeline                 Low-level orchestrator — can be used as a drop-in replacement
                             for ``_run_gurobi_optimizer``.

Example:
--------
Drop-in replacement for the bare SWC-TCF solver::

    from <package>.exact_guided_heuristic import run_pipeline, ExactGuidedHeuristicParams

    params = ExactGuidedHeuristicParams(alpha=0.5, time_limit=120)
    flat_route, profit, cost = run_pipeline(
        bins, dist_matrix, env, values, binsids, mandatory,
        n_vehicles=2, params=params,
    )

Policy registration (automatic on import)::

    from <package>.pipeline import ExactGuidedHeuristicPolicy   # registers "exact_guided_heuristic"
    policy = ExactGuidedHeuristicPolicy()
"""

from .dispatcher import run_pipeline
from .params import ExactGuidedHeuristicParams
from .policy_egh import ExactGuidedHeuristicPolicy
from .route_pool import RoutePool, VRPPRoute

__all__ = [
    "ExactGuidedHeuristicPolicy",
    "ExactGuidedHeuristicParams",
    "RoutePool",
    "VRPPRoute",
    "run_pipeline",
]
