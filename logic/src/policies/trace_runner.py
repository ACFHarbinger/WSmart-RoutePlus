"""Run one registered routing policy and write an execution trace.

Schema issue #168 proves the file shape on ALNS. The ALNS search records
destroy, repair, acceptance, and incumbent events when a trace is passed.
Other routing ids are named in the document and marked ``not_instrumented``;
they are not solved here.
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from logic.src.configs.policies.acceptance_criteria import AcceptanceConfig, OnlyImprovingConfig
from logic.src.configs.policies.routing.meta_heuristics.alns import ALNSConfig
from logic.src.policies.routing.base.sub_problem import SolveResult, build_subproblem
from logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search.dispatcher import (
    run_alns,
)
from logic.src.tracking.execution_trace import ExecutionTrace, solution_record

PROBLEMS = ("ptp", "cmptp", "cmptptc", "cmptptwtc", "cvrp")
TRACED_ROUTING = "alns"


def make_instance(graph_size: int, seed: int) -> Tuple[np.ndarray, Dict[int, float], np.ndarray]:
    """Euclidean instance on ``graph_size`` nodes, depot at index 0.

    The generator is local to ``seed``. It does not touch global NumPy state.
    """
    if graph_size < 3:
        raise ValueError("graph_size must be at least 3")
    generator = np.random.default_rng(seed)
    coords = generator.random((graph_size, 2))
    delta = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((delta ** 2).sum(axis=-1))
    np.fill_diagonal(dist, 0.0)
    wastes = {index: float(generator.random()) + 0.1 for index in range(1, graph_size)}
    return dist, wastes, coords


def _acceptance(name: str) -> AcceptanceConfig:
    if name == "oi":
        return AcceptanceConfig(method="oi", params=OnlyImprovingConfig())
    if name == "bmc":
        return AcceptanceConfig(method="bmc")
    raise ValueError(f"trace_policy acceptance must be oi or bmc, got {name!r}")


def solve_alns(
    dist: np.ndarray,
    wastes: Dict[int, float],
    seed: int,
    acceptance: str,
    max_iterations: int,
) -> SolveResult:
    """Solve with the ALNS dispatcher. ``seed`` fixes the solver RNG."""
    problem = build_subproblem(dist, wastes, capacity=10.0, revenue=1.0, cost_unit=1.0)
    config = ALNSConfig(
        seed=seed,
        max_iterations=max_iterations,
        acceptance_criterion=_acceptance(acceptance),
        time_limit=30.0,
    )
    return run_alns(problem, config)


def build_alns_trace(
    problem: str,
    graph_size: int,
    seed: int,
    acceptance: str = "oi",
    packing: Optional[str] = None,
    route_improvement: Optional[str] = None,
    max_iterations: int = 20,
) -> Tuple[ExecutionTrace, SolveResult]:
    """Solve ALNS once and record the search on ``trace``.

    The solver receives the trace. A call with the same arguments and no trace
    uses :func:`solve_alns` and must return the same tours and profit.
    """
    if problem not in PROBLEMS:
        raise ValueError(f"unknown problem {problem!r}")
    dist, wastes, coords = make_instance(graph_size, seed)
    trace = ExecutionTrace(
        problem=problem,
        policies={
            "packing": packing,
            "routing": TRACED_ROUTING,
            "route_improvement": route_improvement,
            "acceptance": acceptance,
        },
        seed=seed,
        instance={
            "n_nodes": graph_size,
            "depot": [float(coords[0, 0]), float(coords[0, 1])],
            "capacity": 10.0,
        },
    )
    input_id = trace.add(
        name="instance",
        kind="input",
        arguments={"n_nodes": graph_size, "n_customers": graph_size - 1},
    )
    subproblem = build_subproblem(
        dist,
        wastes,
        capacity=10.0,
        revenue=1.0,
        cost_unit=1.0,
        trace=trace,
        trace_parent_id=input_id,
    )
    config = ALNSConfig(
        seed=seed,
        max_iterations=max_iterations,
        acceptance_criterion=_acceptance(acceptance),
        time_limit=30.0,
    )
    result = run_alns(subproblem, config)
    final = solution_record(result.routes, objective=result.profit)
    trace.add(
        name="routes",
        kind="output",
        parent_id=input_id,
        changed_solution=True,
        after=final,
    )
    trace.set_final(result.routes, objective=result.profit)
    return trace, result


def routes_equal(left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]) -> bool:
    """True when both route lists contain the same integer tours."""
    return [list(route) for route in left] == [list(route) for route in right]


def trace_summary(trace: ExecutionTrace) -> Dict[str, Any]:
    """Small dict the CLI can print while debugging a run. Not the file format."""
    document = trace.to_dict()
    return {
        "schema_version": document["schema_version"],
        "events": len(document["events"]),
        "final_hash": None if document["final"] is None else document["final"]["unchanged_hash"],
    }
