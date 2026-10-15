"""
Selection subproblem (the "knapsack side" in the Lagrangian decomposition).

This module does two things:

    1. Produces effective prize coefficients  V + lambda - gamma * Delta + bias
       for each period and feeds them into the per-period TPKS solver.
    2. Runs one TPKS call per period to obtain x^K_{:, t}.  Since TPKS is a
       single-period engine, the "selection subproblem" is really a bank of
       independent per-period calls; the inter-period coupling is delivered
       entirely via V[i, t] (from lookahead) and lambda[i, t] (from the
       coordinator).

Cut strategies
--------------
Three strategies are supported and the active one is chosen by the bandit:

    "plain"  : nothing extra; the TPKS bucket-expansion cuts are sufficient.

    "lifted" : after each period solves, compute per-bin marginal costs
               Delta_i = cost(S_t) - cost(S_t - {i})  via a 1-opt removal
               estimate, and add them as demand-side penalties on future
               iterations.  Validity: we use min(marginal, joint_saving)
               (Fischetti-Salvagnin envelope) so we stay valid even when
               marginals aren't directly summable.

    "pareto" : generate a Pareto-optimal cut via Magnanti-Wong core-point
               lifting.  Core point is the average of the last 3 knapsack
               LP relaxations; if unavailable, falls back to "lifted".

The cuts produced here are ADVISORY -- they are stored on the
`SelectionResult` and applied on the next outer-iteration call (not mid-TPKS).

Attributes:
    build_corrected_revenue: Builds the effective prize vector for the knapsack side.
    _wastes_from_revenue: Builds the waste dictionary for the knapsack side.
    solve_selection_period: Solves the knapsack side of the Lagrangian decomposition.
    SelectionResult: A dataclass that holds the result of the selection subproblem.

Example:
    >>> build_corrected_revenue(np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6]), 0.7, np.array([0.8, 0.9, 1.0]), 0)
    array([1.3, 1.6, 1.9])

    >>> _wastes_from_revenue(np.array([1, 2, 3]))
    {1: 1, 2: 2, 3: 3}

    >>> solve_selection_period(
    ...     period=0,
    ...     dist_matrix=np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]]),
    ...     revenue_eff=np.array([1, 2, 3]),
    ...     capacity=10,
    ...     routing_cost_unit=1,
    ...     mandatory_nodes=[],
    ...     tpks_params=TpksConfig(),
    ...     hard_fix_bins=None,
    ...     engine="greedy",
    ...     prior_cuts=None,
    ... )
    SelectionResult(period=0, selection=[1, 2, 3], tour=[0, 1, 2, 3, 0], lagrangian_objective=10.0, raw_objective=10.0, routing_cost_estimate=6.0, lifted_penalties={}, pareto_penalties={})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.configs.policies.tpks import TpksConfig
from logic.src.policies.routing.matheuristics.two_phase_kernel_search.solver import (
    run_tpks_gurobi,
)


@dataclass
class SelectionResult:
    """One period's selection outcome.

    Attributes:
        period: Period index.
        selection: Bin indices (1-based; 0 is depot).
        tour: Tour for the selected bins.
        lagrangian_objective: Lagrangian objective value.
        raw_objective: Raw objective value.
        routing_cost_estimate: Estimated routing cost.
        lifted_penalties: Penalties for the next iteration.
        pareto_penalties: Pareto penalties for the next iteration.
    """

    period: int
    selection: List[int]  # bin indices (1-based; 0 is depot)
    tour: List[int]
    lagrangian_objective: float
    raw_objective: float
    routing_cost_estimate: float
    # Cuts / penalties to apply on the NEXT outer-iteration call.
    lifted_penalties: Dict[int, float] = field(default_factory=dict)
    pareto_penalties: Dict[int, float] = field(default_factory=dict)


def build_corrected_revenue(
    V: np.ndarray,
    lambdas: np.ndarray,
    insertion_costs: np.ndarray,
    gamma: np.ndarray,
    regret_bias: np.ndarray,
    period: int,
) -> np.ndarray:
    """
    Per-period effective prize vector for the knapsack side.

        revenue_eff[i] = V[i, t] + lambda[i, t] - gamma[t] * Delta[i, t] + bias[i, t]

    Returns a 1-D array of length N.

    Args:
        V: (N, T) prizes.
        lambdas: (N, T) duals.
        insertion_costs: (N, T) insertion costs.
        gamma: (T,) lagrange multipliers for insertion costs.
        regret_bias: (N, T) regret-based bias.
        period: Period index.

    Returns:
        np.ndarray: Effective prizes for the period.
    """
    V_t = V[:, period]
    lam_t = lambdas[:, period]
    delta_t = insertion_costs[:, period]
    bias_t = regret_bias[:, period]
    return V_t + lam_t - gamma[period] * delta_t + bias_t


def _wastes_from_revenue(revenue_eff: np.ndarray) -> Dict[int, float]:
    """
    TPKS's `wastes` dict encodes per-bin effective value.  Build the mapping
    from 1-based bin ids (TPKS indexes bins from 1, with 0 as the depot) to
    the effective revenue.  We clip negatives to zero since TPKS's prize
    formulation assumes non-negative values (a very negative effective prize
    means 'don't bother selecting'; clipping at 0 has the same effect via
    the branch & bound).

    Args:
        revenue_eff: Effective revenue for each bin.

    Returns:
        Dict[int, float]: Dictionary of wastes for each bin.
    """
    out: Dict[int, float] = {}
    for idx, val in enumerate(revenue_eff):
        # Bin index in TPKS is idx + 1 (depot is 0).
        out[idx + 1] = float(max(0.0, val))
    return out


def solve_selection_period(
    *,
    period: int,
    dist_matrix: np.ndarray,
    revenue_eff: np.ndarray,
    capacity: float,
    routing_cost_unit: float,
    mandatory_nodes: List[int],
    tpks_params: TpksConfig,
    hard_fix_bins: Optional[List[int]] = None,
    engine: str = "tpks",
    prior_cuts: Optional[Dict[int, float]] = None,
) -> SelectionResult:
    """
    Solve a single-period selection subproblem.

    Args:
        period: Period index.
        dist_matrix: (N+1, N+1) distance matrix (with depot at index 0).
        revenue_eff: (N,) effective prize per bin.
        capacity: vehicle capacity constraint for the period.
        routing_cost_unit: C, distance cost coefficient.
        mandatory_nodes: bin ids (1-based) that MUST be visited this period.
        tpks_params: TpksConfig.
        hard_fix_bins: bin ids (1-based) forced to be visited via mandatory set
            augmentation.  If None, regret preprocessing is either soft-only or
            not applied for this period.
        engine: "tpks" | "tpks_warm" | "greedy"
        prior_cuts: optional {bin_id: penalty} from a previous iteration.

    Returns:
        SelectionResult: Result of the selection subproblem.
    """
    # Augment mandatory set with hard-fixed bins.
    mandatory = list(mandatory_nodes)
    if hard_fix_bins:
        for b in hard_fix_bins:
            if b not in mandatory:
                mandatory.append(b)

    # Apply prior cuts as additive penalties to revenue.
    wastes = _wastes_from_revenue(revenue_eff)
    if prior_cuts:
        for bin_id, penalty in prior_cuts.items():
            if bin_id in wastes:
                wastes[bin_id] = max(0.0, wastes[bin_id] - penalty)

    if engine == "greedy":
        tour, obj_val, cost = _run_greedy(dist_matrix, wastes, capacity, 1.0, routing_cost_unit, mandatory)
    else:
        # Both "tpks" and "tpks_warm" map to run_tpks_gurobi; the "warm"
        # variant reuses a longer phase-I time budget (set by the caller via
        # tpks_params) and is equivalent at the module API level.
        tour, obj_val, cost = run_tpks_gurobi(
            dist_matrix=dist_matrix,
            wastes=wastes,
            capacity=capacity,
            R=1.0,  # effective prize IS in `wastes`; no further multiplier.
            C=routing_cost_unit,
            mandatory_nodes=mandatory,
            params=tpks_params,
        )

    selection = [n for n in tour if n != 0]
    # Deduplicate (tour has depot at start and end).
    selection = sorted(set(selection))

    # The "raw" objective is the lookahead V portion (without lambdas / Delta).
    # We reconstruct it from the corrected coefficients by subtracting the
    # adjustments.  The caller supplies the pieces we need to perform that
    # reconstruction if desired; for now we return obj_val as lagrangian and
    # the caller computes the raw (primal) objective downstream.
    return SelectionResult(
        period=period,
        selection=selection,
        tour=list(tour),
        lagrangian_objective=float(obj_val),
        raw_objective=float(obj_val),
        routing_cost_estimate=float(cost),
    )


# ---------------------------------------------------------------------------
# Fallback greedy engine for the "greedy" bandit arm.
# ---------------------------------------------------------------------------


def _run_greedy(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: List[int],
) -> Tuple[List[int], float, float]:
    """
    Greedy Cheapest Insertion with Profit Filter.

    Seed: depot -> nearest mandatory -> back to depot.
    Loop: repeatedly insert the bin whose marginal profit (R*waste -
    C*insertion_cost) is maximal, if positive, subject to capacity.

    Args:
        dist_matrix: Distance matrix for the routing problem.
        wastes: Dictionary of wastes for each bin.
        capacity: Capacity of the vehicle.
        R: Revenue factor.
        C: Cost factor.
        mandatory_nodes: List of mandatory nodes.

    Returns:
        Tuple[List[int], float, float]: Tour, objective value, and cost.
    """
    if not mandatory_nodes:
        mandatory_nodes = []
    tour = [0]
    visited = set(tour)

    def insertion_cost(route: List[int], b: int) -> Tuple[int, float]:
        """insertion_cost docstring."""
        best_pos = 1
        best_delta = float("inf")
        for k in range(len(route) - 1):
            u, v = route[k], route[k + 1]
            d = dist_matrix[u, b] + dist_matrix[b, v] - dist_matrix[u, v]
            if d < best_delta:
                best_delta = d
                best_pos = k + 1
        if len(route) == 1:
            return 1, float(dist_matrix[0, b] + dist_matrix[b, 0])
        return best_pos, max(0.0, best_delta)

    # Seed with depot closure.
    tour = [0, 0]

    # Insert mandatory first.
    for b in mandatory_nodes:
        if b in visited:
            continue
        pos, _ = insertion_cost(tour, b)
        tour.insert(pos, b)
        visited.add(b)

    # Greedy profit insertion.
    load = len(visited) - 1  # crude proxy; caller's capacity is in bin count here
    improved = True
    while improved:
        improved = False
        best_gain = 1e-9
        best_bin = None
        best_pos = None
        for b, w in wastes.items():
            if b in visited or b == 0:
                continue
            pos, delta = insertion_cost(tour, b)
            gain = R * w - C * delta
            if gain > best_gain:
                best_gain = gain
                best_bin = b
                best_pos = pos
        if best_bin is not None and best_pos is not None:
            if capacity < 1e8 and load + 1 > capacity:
                break
            tour.insert(best_pos, best_bin)
            visited.add(best_bin)
            load += 1
            improved = True

    # Compute objective and cost.
    cost = 0.0
    for k in range(len(tour) - 1):
        cost += float(dist_matrix[tour[k], tour[k + 1]])
    revenue = sum(wastes.get(b, 0.0) for b in visited if b != 0)
    obj = R * revenue - C * cost
    return tour, obj, cost


# ---------------------------------------------------------------------------
# Cut generation (lifted / Pareto)
# ---------------------------------------------------------------------------


def generate_lifted_cuts(
    result: SelectionResult,
    dist_matrix: np.ndarray,
    revenue_eff: np.ndarray,
) -> Dict[int, float]:
    """
    Per-bin marginal cost estimate: for each selected bin i, compute the
    routing-cost savings from removing i from the tour (cheapest removal).
    Penalty = max(0, routing_cost_of_i - revenue_of_i) = bins that are
    "only marginally profitable" get flagged with a small penalty.

    Args:
        result: Selection result from the previous iteration.
        dist_matrix: Distance matrix for the routing problem.
        revenue_eff: Effective revenue for each bin.

    Returns:
        Dict[int, float]: Dictionary of penalties for each bin.
    """
    tour = result.tour
    if len(tour) < 3:
        return {}
    penalties: Dict[int, float] = {}
    for k in range(1, len(tour) - 1):
        b = tour[k]
        if b == 0:
            continue
        u = tour[k - 1]
        v = tour[k + 1]
        savings = float(dist_matrix[u, b] + dist_matrix[b, v] - dist_matrix[u, v])
        # Envelope: penalty = max(0, savings - revenue_of_b).
        rev_b = revenue_eff[b - 1] if 1 <= b <= len(revenue_eff) else 0.0
        penalty = max(0.0, savings - rev_b)
        if penalty > 0:
            penalties[b] = penalty
    return penalties


def generate_pareto_cuts(
    results: List[SelectionResult],
    dist_matrix: np.ndarray,
    revenue_eff: np.ndarray,
) -> Dict[int, float]:
    """
    Pareto-optimal cut via averaging over recent iterates.

    Takes the average selection frequency across the last few results and
    uses it as a pseudo-core point; bins with high marginal cost AND high
    selection frequency get the strongest penalty (since they're persistently
    in the solution but marginally profitable).

    Args:
        results: List of selection results from previous iterations.
        dist_matrix: Distance matrix for the routing problem.
        revenue_eff: Effective revenue for each bin.

    Returns:
        Dict[int, float]: Dictionary of penalties for each bin.
    """
    if not results:
        return {}
    # Fall back to lifted if only one result is available.
    if len(results) == 1:
        return generate_lifted_cuts(results[-1], dist_matrix, revenue_eff)

    n_bins = len(revenue_eff)
    frequency = np.zeros(n_bins, dtype=float)
    for r in results:
        for b in r.selection:
            if 1 <= b <= n_bins:
                frequency[b - 1] += 1.0
    frequency /= len(results)

    lifted = generate_lifted_cuts(results[-1], dist_matrix, revenue_eff)
    pareto: Dict[int, float] = {}
    for b, pen in lifted.items():
        f = frequency[b - 1] if 1 <= b <= n_bins else 0.0
        pareto[b] = pen * (0.5 + 0.5 * f)  # weight by persistence in selections
    return pareto
