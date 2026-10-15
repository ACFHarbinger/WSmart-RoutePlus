"""
HGS Dispatcher module.

Dispatches to different HGS implementations based on configuration.

Attributes:
    run_hgs: Main entry point for running HGS.

Example:
    >>> routes, profit, cost = run_hgs(dist_matrix, wastes, capacity, R, C, values)
"""

import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from logic.src.configs.policies.hgs import HGSConfig

from .hgs import HGSSolver
from .pyvrp_wrapper import solve_pyvrp


def run_hgs(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    values: Optional[Union[HGSConfig, Dict[str, Any]]] = None,
    mandatory_nodes: Optional[List[int]] = None,
    x_coords: Optional[np.ndarray] = None,
    y_coords: Optional[np.ndarray] = None,
    *args: Any,
) -> Tuple[List[List[int]], float, float]:
    """
    Main HGS entry point with dispatching logic.

    Args:
        dist_matrix: Distance matrix.
        wastes: Bin wastes.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        values: HGSConfig dataclass or dictionary of parameters and config.
        mandatory_nodes: List of local node indices that MUST be visited.
        x_coords: Optional x-coordinates.
        y_coords: Optional y_coords.
        args: Additional positional arguments (ignored).

    Returns:
        Tuple[List[List[int]], float, float]: Best routes, profit, and cost.
    """
    if isinstance(values, HGSConfig):
        cfg = values
    elif isinstance(values, dict):
        valid_keys = {f.name for f in dataclasses.fields(HGSConfig)}
        filtered = {k: v for k, v in values.items() if k in valid_keys}
        cfg = HGSConfig(**filtered)
    elif values is None:
        cfg = HGSConfig()
    else:
        cfg = values

    if getattr(cfg, "engine", "custom") == "pyvrp":
        vals = values if isinstance(values, dict) else dataclasses.asdict(cfg)
        return solve_pyvrp(dist_matrix, wastes, capacity, R, C, vals)

    if len(dist_matrix) <= 1:
        return [], 0.0, 0.0

    if len(dist_matrix) == 2:
        d = wastes.get(1, 0)
        if d <= capacity:
            # Fix 5: Calculate simple profit = revenue - cost
            cost = dist_matrix[0][1] + dist_matrix[1][0]
            revenue = d * R
            total_cost = C * cost
            profit = revenue - total_cost
            return [[1]], profit, total_cost
        else:
            return [], 0.0, 0.0

    solver = HGSSolver(dist_matrix, wastes, capacity, R, C, cfg, mandatory_nodes, x_coords, y_coords)
    return solver.solve()
