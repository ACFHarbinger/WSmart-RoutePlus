r"""ALNS Dispatcher module.

Dispatches to different ALNS implementations based on configuration.

Attributes:
    run_alns: High-level entry point for ALNS.

Example:
    >>> from logic.src.policies.routing.base.sub_problem import build_subproblem
    >>> from logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search.dispatcher import run_alns
    >>> problem = build_subproblem(dist, wastes, 50.0, 1.0, 1.0)
    >>> routes, profit, cost = run_alns(problem, {"engine": "custom"})
"""

import dataclasses
from typing import Any, Dict, Optional, Union

from logic.src.configs.policies.routing.meta_heuristics.alns import ALNSConfig
from logic.src.policies.routing.base.sub_problem import SolveResult, SubProblem

from .alns import ALNSSolver
from .alns_package import run_alns_package
from .ortools_wrapper import run_alns_ortools


def run_alns(
    problem: SubProblem,
    cfg: Optional[Union[ALNSConfig, Dict[str, Any]]] = None,
    trace: Optional[Any] = None,
    trace_parent_id: Optional[str] = None,
) -> SolveResult:
    """Main ALNS entry point with dispatching to different algorithm variants.

    Args:
        problem: The routing sub-problem (distance matrix, wastes, capacity,
            revenue, cost unit, mandatory nodes, optional recorder).
        cfg: ALNSConfig dataclass or dictionary of parameters and config.
            ``engine`` selects the variant: "custom" (default), "package"
            (external ``alns`` package) or "ortools".
        trace: Optional execution trace to record search events into.
        trace_parent_id: Optional parent event id for trace events.

    Returns:
        SolveResult: Best routes, profit, and cost.
    """
    if isinstance(cfg, ALNSConfig):
        config = cfg
        variant = cfg.engine
    elif isinstance(cfg, dict):
        variant = cfg.get("variant") or cfg.get("engine") or "custom"
        valid_keys = {f.name for f in dataclasses.fields(ALNSConfig)}
        filtered = {k: v for k, v in cfg.items() if k in valid_keys}
        config = ALNSConfig(**filtered)
    elif cfg is None:
        config = ALNSConfig()
        variant = config.engine
    else:
        config = cfg
        variant = getattr(cfg, "engine", "custom")

    if variant == "package":
        vals = cfg if isinstance(cfg, dict) else dataclasses.asdict(config)
        return SolveResult(*run_alns_package(problem, vals))
    elif variant == "ortools":
        vals = cfg if isinstance(cfg, dict) else dataclasses.asdict(config)
        return SolveResult(*run_alns_ortools(problem, vals))

    # Default: Custom internal ALNS solver
    solver = ALNSSolver(problem, config)
    effective_trace = trace if trace is not None else getattr(problem, "trace", None)
    effective_parent_id = (
        trace_parent_id if trace_parent_id is not None else getattr(problem, "trace_parent_id", None)
    )
    return SolveResult(*solver.solve(trace=effective_trace, parent_id=effective_parent_id))
