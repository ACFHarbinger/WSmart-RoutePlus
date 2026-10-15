r"""ALNS Dispatcher module.

Dispatches to different ALNS implementations based on configuration.

Attributes:
    run_alns: High-level entry point for ALNS.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search.dispatcher import run_alns
    >>> routes, profit, cost = run_alns(dist, wastes, 50.0, 1.0, 1.0, {"engine": "custom"})
"""

import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from logic.src.configs.policies.alns import ALNSConfig

from .alns import ALNSSolver
from .alns_package import run_alns_package
from .ortools_wrapper import run_alns_ortools


def run_alns(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    values: Optional[Union[ALNSConfig, Dict[str, Any]]] = None,
    mandatory_nodes: Optional[List[int]] = None,
    recorder: Optional[Any] = None,
    *args: Any,
) -> Tuple[List[List[int]], float, float]:
    """Main ALNS entry point with dispatching to different algorithm variants.

    Args:
        dist_matrix: Distance matrix.
        wastes: Bin wastes.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        values: ALNSConfig dataclass or dictionary of parameters and config.
        mandatory_nodes: List of mandatory node indices.
        recorder: Optional telemetry recorder.
        args: Additional arguments (ignored or passed through).

    Returns:
        Tuple[List[List[int]], float, float]: Best routes, profit, and cost.
    """
    if isinstance(values, ALNSConfig):
        cfg = values
        variant = cfg.engine
    elif isinstance(values, dict):
        variant = values.get("variant") or values.get("engine") or "custom"
        valid_keys = {f.name for f in dataclasses.fields(ALNSConfig)}
        filtered = {k: v for k, v in values.items() if k in valid_keys}
        cfg = ALNSConfig(**filtered)
    elif values is None:
        cfg = ALNSConfig()
        variant = cfg.engine
    else:
        cfg = values
        variant = getattr(cfg, "engine", "custom")

    if variant == "package":
        vals = values if isinstance(values, dict) else dataclasses.asdict(cfg)
        return run_alns_package(dist_matrix, wastes, capacity, R, C, vals, recorder=recorder)
    elif variant == "ortools":
        vals = values if isinstance(values, dict) else dataclasses.asdict(cfg)
        return run_alns_ortools(dist_matrix, wastes, capacity, R, C, vals)

    # Default: Custom internal ALNS solver
    solver = ALNSSolver(dist_matrix, wastes, capacity, R, C, cfg, mandatory_nodes, recorder=recorder)
    return solver.solve()
