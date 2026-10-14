r"""ALNS Dispatcher module.

Dispatches to different ALNS implementations based on configuration.

Attributes:
    run_alns: High-level entry point for ALNS.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search.dispatcher import run_alns
    >>> routes, profit, cost = run_alns(dist, wastes, 50.0, 1.0, 1.0, {"engine": "custom"})
"""

from .alns import ALNSSolver
from .alns_package import run_alns_package
from .ortools_wrapper import run_alns_ortools
from .params import ALNSParams


def run_alns(dist_matrix, wastes, capacity, R, C, values, mandatory_nodes=None, recorder=None, *args):
    """Main ALNS entry point with dispatching to different algorithm variants.

    Args:
        dist_matrix: Distance matrix.
        wastes: Bin wastes.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        values: Dictionary of parameters and config.
        mandatory_nodes: List of mandatory node indices.
        recorder: Optional telemetry recorder.
        args: Additional arguments (ignored or passed through).

    Returns:
        Tuple[List[List[int]], float, float]: Best routes, profit, and cost.
    """
    variant = values.get("variant") or values.get("engine") or "custom"

    if variant == "package":
        return run_alns_package(dist_matrix, wastes, capacity, R, C, values)
    elif variant == "ortools":
        return run_alns_ortools(dist_matrix, wastes, capacity, R, C, values)

    # Default: Custom internal ALNS solver
    params = ALNSParams.from_config(values)
    solver = ALNSSolver(dist_matrix, wastes, capacity, R, C, params, mandatory_nodes, recorder=recorder)
    return solver.solve()
