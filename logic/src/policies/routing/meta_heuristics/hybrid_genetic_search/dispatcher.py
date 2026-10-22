"""
HGS Dispatcher module.

Dispatches to different HGS implementations based on configuration.

Attributes:
    run_hgs: Main entry point for running HGS.

Example:
    >>> routes, profit, cost = run_hgs(problem, values)
"""

import dataclasses
from typing import Any, Dict, Optional, Union

from logic.src.configs.policies.routing.meta_heuristics.hgs import HGSConfig
from logic.src.policies.routing.base.sub_problem import SolveResult, SubProblem

from .hgs import HGSSolver
from .pyvrp_wrapper import solve_pyvrp


def run_hgs(
    problem: SubProblem,
    cfg: Optional[Union[HGSConfig, Dict[str, Any]]] = None,
) -> SolveResult:
    """
    Main HGS entry point with dispatching logic.

    P-30: takes the consolidated sub-problem value object plus the typed
    config; coordinates come from ``problem.x_coords``/``problem.y_coords``.

    Args:
        problem: The routing sub-problem (distances, wastes, capacity,
            revenue, cost unit, mandatory nodes, optional coordinates).
        cfg: HGSConfig dataclass or dictionary of parameters and config.

    Returns:
        SolveResult: Best routes, profit, and cost.
    """
    if isinstance(cfg, HGSConfig):
        config = cfg
    elif isinstance(cfg, dict):
        valid_keys = {f.name for f in dataclasses.fields(HGSConfig)}
        filtered = {k: v for k, v in cfg.items() if k in valid_keys}
        config = HGSConfig(**filtered)
    elif cfg is None:
        config = HGSConfig()
    else:
        config = cfg

    dist_matrix = problem.dist_matrix

    if getattr(config, "engine", "custom") == "pyvrp":
        vals = cfg if isinstance(cfg, dict) else dataclasses.asdict(config)
        return solve_pyvrp(dist_matrix, problem.wastes, problem.capacity, problem.revenue, problem.cost_unit, vals)

    if len(dist_matrix) <= 1:
        return SolveResult([], 0.0, 0.0)

    if len(dist_matrix) == 2:
        d = problem.wastes.get(1, 0)
        if d <= problem.capacity:
            # Fix 5: Calculate simple profit = revenue - cost
            cost = dist_matrix[0][1] + dist_matrix[1][0]
            revenue = d * problem.revenue
            total_cost = problem.cost_unit * cost
            profit = revenue - total_cost
            return SolveResult([[1]], profit, total_cost)
        else:
            return SolveResult([], 0.0, 0.0)

    solver = HGSSolver(problem, config)
    return solver.solve()
