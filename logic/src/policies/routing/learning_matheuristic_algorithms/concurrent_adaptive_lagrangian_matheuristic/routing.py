"""
Routing subproblem worker (the "RS side" in the Lagrangian decomposition).

Given a fixed selection S_t for period t, the RS evaluates the true routing
cost by invoking an LKH-3-style tour improver, and pushes the result through
the shared infrastructure:

    1. Updates the InsertionCostOracle (gated EMA).
    2. Writes x_R[:, t] to LagrangianState.
    3. Submits the period's subgradient to the LagrangianCoordinator, which
       (for the EMA tracker) performs an async per-period multiplier update.
    4. Returns a summary dict for telemetry / bandit reward.

In this codebase the per-period RS is TPKS itself (it returns a tour and a
cost) -- we treat the LKH-3-style local search as an optional inner loop
that may or may not exist in the environment.  If an `lkh3_improver`
callable is supplied, we use it to refine the tour before computing the
insertion-cost oracle updates.  Otherwise the TPKS-returned tour is used as
the final answer.

Concurrency
-----------
The per-period workers are designed to run concurrently via a thread pool
(concurrent.futures.ThreadPoolExecutor).  Gurobi's TPKS solve releases the
GIL for most of its wall-clock time, and all writes to shared state go
through locks.

Attributes:
    RoutingResult: Result of a single period's routing subproblem evaluation.
    evaluate_period: Evaluates a single period's routing subproblem, updates oracle + coordinator.

Example:
    >>> evaluate_period(
    ...     selection_result=SelectionResult(
    ...         period=0,
    ...         selection=[1, 2, 3],
    ...         tour=[0, 1, 2, 3, 0],
    ...         lagrangian_objective=10.0,
    ...         raw_objective=10.0,
    ...         routing_cost_estimate=6.0,
    ...         lifted_penalties={},
    ...         pareto_penalties={},
    ...     ),
    ...     dist_matrix=np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]]),
    ...     n_bins=3,
    ...     V_column=np.array([1, 2, 3]),
    ...     lambdas_column=np.array([0.1, 0.2, 0.3]),
    ...     oracle=InsertionCostOracle(np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])),
    ...     coordinator=LagrangianCoordinator(n_periods=1),
    ...     upper_bound=10.0,
    ...     lkh3_improver=None,
    ... )
    RoutingResult(period=0, tour=[0, 1, 2, 3, 0], tour_cost=6.0, selection=[1, 2, 3], quality_ratio=1.0, insertion_costs_for_unselected={}, lagrangian_value_contrib=9.4, accepted_by_oracle='accepted_improving', multiplier_step=0.1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .lagrangian import LagrangianCoordinator
from .oracle import InsertionCostOracle
from .selection import SelectionResult


@dataclass
class RoutingResult:
    """
    Result of a single period's routing subproblem evaluation.

    Attributes:
        period: The time period (0-indexed).
        tour: The full tour including the depot (e.g., [0, 3, 1, 0]).
        tour_cost: The cost of the tour.
        selection: The set of nodes visited in the tour (excluding the depot).
        quality_ratio: Ratio of the current tour cost to the incumbent cost (>= 1.0 if improving).
        insertion_costs_for_unselected: Dictionary mapping unselected node indices to their insertion costs.
        lagrangian_value_contrib: The Lagrangian value contribution from this period (prize - cost).
        accepted_by_oracle: Indicates whether the solution was accepted by the oracle ("accepted_improving", "accepted_partial", "rejected").
        multiplier_step: The multiplier step size determined by the coordinator.
    """

    period: int
    tour: List[int]
    tour_cost: float
    selection: List[int]
    quality_ratio: float  # tour_cost / incumbent_cost (>= 1 unless new incumbent)
    insertion_costs_for_unselected: Dict[int, float]
    lagrangian_value_contrib: float
    accepted_by_oracle: str  # {"accepted_improving", "accepted_partial", "rejected"}
    multiplier_step: float


def evaluate_period(
    *,
    selection_result: SelectionResult,
    dist_matrix: np.ndarray,
    n_bins: int,
    V_column: np.ndarray,
    lambdas_column: np.ndarray,
    oracle: InsertionCostOracle,
    coordinator: LagrangianCoordinator,
    upper_bound: float,
    lkh3_improver: Optional[Callable[[List[int], np.ndarray], Tuple[List[int], float]]] = None,
) -> RoutingResult:
    """
    Evaluate a single period's routing subproblem, update oracle + coordinator.

    Args:
        selection_result : SelectionResult
            Output of the corresponding knapsack solve for this period.
        dist_matrix : (N+1, N+1)
            Distance matrix including the depot.
        n_bins : int  (= N, without depot)
        V_column : (N,)  lookahead values for this period (raw, not corrected).
        lambdas_column : (N,)  current lambda[:, period].
        oracle : InsertionCostOracle
        coordinator : LagrangianCoordinator
        upper_bound : float  (best known primal; used for Polyak stepsize).
        lkh3_improver : optional callable that takes (tour, dist_matrix) ->
                    (improved_tour, improved_cost).

    Returns:
        RoutingResult: Result of the routing subproblem evaluation.
    """
    period = selection_result.period
    tour = list(selection_result.tour)
    cost = float(selection_result.routing_cost_estimate)

    # Optional LKH-3 polishing.
    if lkh3_improver is not None and len(tour) >= 4:
        try:
            improved_tour, improved_cost = lkh3_improver(tour, dist_matrix)
            if improved_cost < cost - 1e-9:
                tour = improved_tour
                cost = improved_cost
        except Exception:
            # Polishing is optional; on any failure we keep the TPKS tour.
            pass

    selection = [n for n in tour if n != 0]
    selection = sorted(set(selection))

    # Compute the Lagrangian contribution of this period:
    #   sum_{i in S_t} V[i, t] - lambda[i, t]     (routing side of the split)
    # minus the routing cost (using the per-period cost_unit C).  The caller
    # absorbs the cost_unit; we report contribution in terms of prize here.
    selected_idx = np.array([b - 1 for b in selection if 1 <= b <= n_bins], dtype=int)
    prize_contrib = float(np.sum(V_column[selected_idx] - lambdas_column[selected_idx])) if selected_idx.size else 0.0
    lagr_contrib = prize_contrib - cost

    # Build the x_R column and push to coordinator.
    x_R_col = np.zeros(n_bins, dtype=float)
    x_R_col[selected_idx] = 1.0
    coordinator.set_routing_selection(period=period, x_R_column=x_R_col)

    # Compute insertion costs for UNSELECTED bins.
    unselected = [b for b in range(1, n_bins + 1) if b not in selection]
    insertion_costs = oracle.batch_insertion_costs(dist_matrix=dist_matrix, tour=tour, candidates=unselected)

    # Update the oracle (gated).
    incumbent_cost = oracle.get_incumbent(period).cost
    quality_ratio = cost / incumbent_cost if np.isfinite(incumbent_cost) else 1.0
    outcome = oracle.update_from_routing(
        period=period,
        tour=tour,
        tour_cost=cost,
        selection=selection,
        insertion_costs_for_unselected=insertion_costs,
    )

    # Submit to the Lagrangian coordinator.
    _, step = coordinator.submit_period_result(
        period=period,
        lagrangian_value_contrib=lagr_contrib,
        tour_quality_ratio=quality_ratio,
        upper_bound=upper_bound,
    )

    return RoutingResult(
        period=period,
        tour=tour,
        tour_cost=cost,
        selection=selection,
        quality_ratio=quality_ratio,
        insertion_costs_for_unselected=insertion_costs,
        lagrangian_value_contrib=lagr_contrib,
        accepted_by_oracle=outcome,
        multiplier_step=step,
    )
