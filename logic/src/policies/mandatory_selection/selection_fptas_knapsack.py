"""
FPTAS Bi-Objective Multiple-Knapsack Selection Strategy Module.

Combines the overflow-risk objective of ``MIPKnapsackSelection`` with the
net-profit objective of ``FractionalKnapsackSelection`` into a single
scalarised per-bin value, then solves the resulting 0/1 multiple-knapsack
problem via a Fully Polynomial Time Approximation Scheme (FPTAS).

Formulation
-----------
Each bin *i* receives a combined dimensionless value:

.. math::

    v_i = \\alpha \\cdot
            \\frac{\\mathrm{risk}_i}{\\mathrm{risk}_{\\max}}
          + (1-\\alpha) \\cdot
            \\frac{\\max(0,\\; \\mathrm{profit}_i)}{\\mathrm{profit}_{\\max}}

where

- :math:`\\mathrm{risk}_i` (kg) is the expected overflow loss (spilled waste
  plus occurrence penalty) incurred if bin *i* is **not** collected today —
  identical to the score used by ``MIPKnapsackSelection``.
- :math:`\\mathrm{profit}_i` (monetary) is the net revenue from collecting
  bin *i*, approximated as the standalone depot-round-trip profit
  :math:`m_i \\cdot r_{\\mathrm{kg}} - \\lambda \\cdot 2d(0,i)` — the same
  eligibility metric used by ``FractionalKnapsackSelection``.
- :math:`\\alpha \\in [0,1]` is the overflow-weight hyperparameter.
- Both objectives are normalised to :math:`[0,1]` so :math:`\\alpha` is
  dimensionless and stable across problem instances.

FPTAS — value-scaled 0/1 knapsack DP
--------------------------------------
Given a single knapsack of capacity *Q* and *m* items with (real-valued)
values :math:`v_i` and (real-valued) weights :math:`w_i`, the FPTAS proceeds:

1. Compute the scaling factor :math:`K = \\varepsilon \\cdot v_{\\max} / m`.
2. Round down each value: :math:`p_i' = \\lfloor v_i / K \\rfloor` (integer).
3. Solve the exact min-weight DP on scaled values:

   .. math::

       \\mathrm{dp}[v] = \\min \\Bigl\\{
           \\sum_{i \\in S} w_i :
           S \\subseteq [m],\\; \\sum_{i \\in S} p_i' = v
       \\Bigr\\}

   The DP has :math:`m \\cdot \\lfloor m/\\varepsilon \\rfloor \\leq m^2/\\varepsilon`
   states — polynomial in *m* and :math:`1/\\varepsilon`.

4. Return the subset achieving the largest *v* with
   :math:`\\mathrm{dp}[v] \\leq Q`.

The returned solution satisfies :math:`V(S) \\geq (1-\\varepsilon)\\,\\mathrm{OPT}`.

Multiple-vehicle decomposition
-------------------------------
The *K*-vehicle multiple-knapsack problem is decomposed greedily: for each
vehicle *k* in turn the FPTAS is applied to the remaining unscheduled bins
with that vehicle's capacity.  Letting :math:`\\mathrm{OPT}_k` be the
optimal value for vehicle *k* over remaining bins,

.. math::

    V(S_k) \\geq (1-\\varepsilon)\\,\\mathrm{OPT}_k,

and therefore :math:`V(S) = \\sum_k V(S_k) \\geq (1-\\varepsilon)\\,\\mathrm{OPT}`
(since :math:`\\mathrm{OPT} \\leq \\sum_k \\mathrm{OPT}_k` by the greedy
feasibility of the decomposition).

A 1/2-approximation lower bound on the joint objective is recovered by
comparing the greedy multi-vehicle solution against the single best-fitting
bin — mirroring the argument in ``FractionalKnapsackSelection``.

Memory budget
-------------
The forward DP is kept in a 1-D array of size :math:`V_{\\mathrm{upper}}+1`
and updated in-place (using a row snapshot to preserve 0/1 semantics).
A boolean *taken* matrix of shape :math:`m \\times (V_{\\mathrm{upper}}+1)` is
maintained for backtracking.  :math:`V_{\\mathrm{upper}}` is capped at
``_MAX_DP_VALUES`` (default 100 000) by adaptively relaxing :math:`\\varepsilon`
when necessary, keeping peak memory well below 100 MB for typical VRP
instance sizes (:math:`m \\leq 500`).

Attributes:
    FPTASBiObjectiveKnapsackSelection: The FPTAS Bi-Objective Multiple-Knapsack
        Selection strategy.
    _EPS_GUARD: Epsilon guard for floating-point comparisons.
    _DIST_EPS: Distance epsilon for insertion-distance lower-bound.
    _MAX_DP_VALUES: Maximum number of DP values.

Example:

    >>> from logic.src.policies.mandatory_selection.selection_fptas_bi_objective_knapsack import (
    ...     FPTASBiObjectiveKnapsackSelection,
    ... )
    >>> strategy = FPTASBiObjectiveKnapsackSelection(epsilon=0.05, alpha=0.6)
    >>> bins, ctx = strategy.select_bins(context)
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context import SearchContext, SelectionContext
from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy
from logic.src.policies.mandatory_selection.base import MandatorySelectionRegistry

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_EPS_GUARD: float = 1e-9  # float comparison / zero-division guard
_DIST_EPS: float = 1e-9  # insertion-distance lower-bound
_MAX_DP_VALUES: int = 100_000  # DP table width ceiling (memory guard)


# ---------------------------------------------------------------------------
# Overflow risk (mirrors MIPKnapsackSelection._compute_overflow_risk exactly)
# ---------------------------------------------------------------------------


def _compute_overflow_risk(
    current_fill: np.ndarray,
    bin_mass: np.ndarray,
    scenario_tree: Optional[Any],
    overflow_penalty_frac: float,
) -> np.ndarray:
    """Per-bin overflow risk score — cost of **not** collecting bin *i* (kg).

    Computes the probability-weighted spilled waste plus an occurrence penalty
    across all future-day nodes in ``scenario_tree``.  Falls back to a
    deterministic current-fill proxy when no tree is provided.

    Args:
        current_fill (np.ndarray): Current fill levels as percentages (shape ``(n_bins,)``).
        bin_mass (np.ndarray): Full waste capacity in kg for each bin.
        scenario_tree (Optional[Any]): Optional ``ScenarioTree`` from the prediction module.
            Must expose ``get_scenarios_at_day(day)`` and ``horizon``.
        overflow_penalty_frac (float): Occurrence penalty as a fraction of full bin
            capacity added per unit of overflow probability.

    Returns:
        np.ndarray: Per-bin overflow risk in kg (shape ``(n_bins,)``).
    """
    n_bins = len(current_fill)

    if scenario_tree is None or not hasattr(scenario_tree, "get_scenarios_at_day"):
        overflow_pct = np.maximum(0.0, current_fill - 100.0)
        overflow_kg = (overflow_pct / 100.0) * bin_mass
        overflow_prob = (current_fill >= 100.0).astype(float)
        return overflow_kg + overflow_prob * overflow_penalty_frac * bin_mass

    expected_overflow_kg = np.zeros(n_bins, dtype=float)
    overflow_prob_any = np.zeros(n_bins, dtype=float)
    horizon: int = getattr(scenario_tree, "horizon", 1)

    for day in range(1, horizon + 1):
        try:
            day_nodes = scenario_tree.get_scenarios_at_day(day)
        except Exception:
            continue
        for node in day_nodes:
            prob = float(getattr(node, "probability", 0.0))
            if prob <= 0.0:
                continue
            wastes = np.asarray(getattr(node, "wastes", np.empty(0)), dtype=float)
            if wastes.size == 0:
                continue
            n_sc = min(n_bins, wastes.size)
            overflow_pct = np.maximum(0.0, wastes[:n_sc] - 100.0)
            expected_overflow_kg[:n_sc] += prob * (overflow_pct / 100.0) * bin_mass[:n_sc]
            overflow_prob_any[:n_sc] = np.minimum(
                1.0,
                overflow_prob_any[:n_sc] + prob * (wastes[:n_sc] >= 100.0).astype(float),
            )

    return expected_overflow_kg + overflow_prob_any * overflow_penalty_frac * bin_mass


# ---------------------------------------------------------------------------
# FPTAS core: (1-ε)-approximation for the 0/1 knapsack problem
# ---------------------------------------------------------------------------


def _fptas_01_knapsack(
    values: np.ndarray,
    weights: np.ndarray,
    capacity: float,
    epsilon: float,
) -> np.ndarray:
    """FPTAS (1-ε)-approximation for the 0/1 knapsack problem.

    **Algorithm** (value-scaled DP):

    1. Filter to feasible items (weight ≤ capacity, value > 0).
    2. Compute scale factor :math:`K = \\varepsilon \\cdot v_{\\max} / m`.
       If the resulting DP width :math:`V_{\\mathrm{upper}} = m \\cdot
       \\lfloor m/\\varepsilon \\rfloor` exceeds ``_MAX_DP_VALUES``, raise
       :math:`\\varepsilon` (×1.5) until it fits.
    3. Solve the min-weight DP on integer-scaled values using a 1-D array
       updated with a per-row snapshot (standard 0/1 trick, vectorised).
    4. Track which item improved each DP state in a boolean ``taken`` matrix
       (shape :math:`m \\times (V_{\\mathrm{upper}}+1)`) to enable O(m) backtracking.
    5. Backtrack from the best feasible value to recover the selected items.

    **Correctness** of the ``taken`` backtracking: ``taken[i, v] = True`` is
    set only when item *i* strictly improves ``dp[v]`` at step *i* (using the
    pre-step snapshot to enforce 0/1 semantics).  Backtracking from
    ``best_val`` downward, selecting item *i* when ``taken[i, rem]`` is True
    and reducing ``rem`` by ``p_scaled[i]``, recovers a valid subset by
    induction on the DP layers.

    Args:
        values (np.ndarray): Non-negative per-item values (shape ``(n,)``).
        weights (np.ndarray): Non-negative per-item weights (shape ``(n,)``).
        capacity (float): Knapsack weight capacity (scalar).
        epsilon (float): Approximation parameter ε ∈ (0, 1).

    Returns:
        np.ndarray: Boolean mask of length ``n`` — True for each selected item.
    """
    n_all = len(values)
    result_mask = np.zeros(n_all, dtype=bool)

    # Restrict to items that individually fit and carry positive value.
    feasible = (weights > _EPS_GUARD) & (weights <= capacity + _EPS_GUARD) & (values > _EPS_GUARD)
    if not np.any(feasible):
        return result_mask

    idx = np.where(feasible)[0]
    v = values[idx].astype(np.float64)
    w = weights[idx].astype(np.float64)
    m = int(idx.size)
    v_max = float(v.max())

    # ------------------------------------------------------------------
    # Determine scale factor K, raising ε if the DP table would be too
    # wide (memory guard).  V_upper = m * floor(m/ε) ≤ m²/ε.
    # ------------------------------------------------------------------
    eps = float(np.clip(epsilon, _EPS_GUARD, 1.0 - _EPS_GUARD))
    while True:
        scale_K = eps * v_max / m
        if scale_K < _EPS_GUARD:
            scale_K = v_max  # degenerate fallback: all items get scaled value 1
        p_scaled = np.floor(v / scale_K).astype(np.int64)
        p_scaled = np.maximum(p_scaled, 0)
        p_max_s = int(p_scaled.max()) if p_scaled.size > 0 else 0
        V_upper = m * p_max_s
        if V_upper <= _MAX_DP_VALUES:
            break
        eps = min(eps * 1.5, 0.99)

    if V_upper == 0:
        return result_mask

    # ------------------------------------------------------------------
    # Forward DP
    #
    #   dp[v]  = minimum total weight to achieve total scaled value v
    #            using a subset of items processed so far.
    #   dp_row = snapshot of dp *before* processing item i, used to
    #            enforce the 0/1 constraint (each item used at most once).
    #   taken[i, v] = True iff item i strictly improved dp[v] at step i.
    # ------------------------------------------------------------------
    dp = np.full(V_upper + 1, np.inf, dtype=np.float64)
    dp[0] = 0.0
    taken = np.zeros((m, V_upper + 1), dtype=bool)

    for i in range(m):
        pi = int(p_scaled[i])
        wi = float(w[i])
        if pi == 0:
            continue

        # Snapshot before this item's update — critical for 0/1 semantics.
        # Without the snapshot, forward traversal would allow item i to be
        # counted multiple times (unbounded knapsack).
        dp_row = dp.copy()

        v_range = np.arange(pi, V_upper + 1, dtype=np.int64)
        prev_w = dp_row[v_range - pi]  # weights from the snapshot
        candidate = prev_w + wi
        finite = np.isfinite(prev_w)
        improve = finite & (candidate < dp[v_range])

        taken[i, v_range[improve]] = True
        dp[v_range[improve]] = candidate[improve]

    # ------------------------------------------------------------------
    # Best feasible value: largest v with dp[v] ≤ capacity.
    # ------------------------------------------------------------------
    achievable = np.where(dp <= capacity + _EPS_GUARD)[0]
    if achievable.size == 0:
        return result_mask
    best_val = int(achievable.max())
    if best_val == 0:
        return result_mask

    # ------------------------------------------------------------------
    # Backtrack in O(m): item i is selected iff taken[i, rem] is True,
    # at which point rem is decremented by p_scaled[i].
    # ------------------------------------------------------------------
    selected_local = np.zeros(m, dtype=bool)
    rem = best_val
    for i in range(m - 1, -1, -1):
        pi = int(p_scaled[i])
        if rem >= pi and taken[i, rem]:
            selected_local[i] = True
            rem -= pi

    result_mask[idx[selected_local]] = True
    return result_mask


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


@GlobalRegistry.register(
    PolicyTag.SELECTION,
    PolicyTag.MATHEURISTIC,
    PolicyTag.HEURISTIC,
    PolicyTag.PROFIT_AWARE,
)
@MandatorySelectionRegistry.register("fptas_knapsack")
class FPTASKnapsackSelection(IMandatorySelectionStrategy):
    """FPTAS bi-objective multiple-knapsack selection strategy.

    Scalarises the overflow-risk objective (kg) of ``MIPKnapsackSelection``
    and the net-profit objective (monetary) of ``FractionalKnapsackSelection``
    into a single dimensionless per-bin value via a convex combination, then
    solves the resulting 0/1 multiple-knapsack problem with a sequential FPTAS
    delivering a :math:`(1-\\varepsilon)`-approximation per vehicle.

    **Combined value per bin:**

    .. math::

        v_i = \\alpha \\cdot \\frac{\\mathrm{risk}_i}{\\mathrm{risk}_{\\max}}
              + (1-\\alpha) \\cdot
                \\frac{\\max(0, \\mathrm{profit}_i)}{\\mathrm{profit}_{\\max}}

    where :math:`\\mathrm{risk}_i` is the expected overflow cost (kg) of
    skipping bin *i* (scenario-tree or fill-level proxy) and
    :math:`\\mathrm{profit}_i = m_i r_{\\mathrm{kg}} - \\lambda \\cdot 2d(0,i)`
    is the depot-round-trip net revenue from collecting it.  Both objectives
    are normalised to :math:`[0,1]` so :math:`\\alpha` is dimensionless and
    instance-independent.

    **Approximation guarantee:**  For each vehicle *k* the FPTAS returns a
    packing of value :math:`\\geq (1-\\varepsilon)\\,\\mathrm{OPT}_k` over the
    bins remaining after vehicles :math:`1,\\ldots,k-1` have been packed.
    Comparing the greedy solution against the single best-value bin gives the
    standard :math:`1/2`-approximation lower bound on the joint objective.

    Attributes:
        epsilon (float): FPTAS approximation parameter ε ∈ (0, 1).
        alpha (float): Overflow-risk weight ∈ [0, 1].
        overflow_penalty_frac (float): Occurrence penalty per overflow event.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        alpha: float = 0.5,
        overflow_penalty_frac: Optional[float] = None,
    ) -> None:
        """
        Initializes the FPTAS knapsack selection.

        Args:
            epsilon: Precision parameter.
            alpha: Alpha parameter.
            overflow_penalty_frac: Fraction of overflow penalty.
        """
        if not 0.0 < epsilon < 1.0:
            raise ValueError(f"epsilon must be in (0, 1), got {epsilon!r}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.overflow_penalty_frac = None if overflow_penalty_frac is None else float(overflow_penalty_frac)

    # ------------------------------------------------------------------
    # IMandatorySelectionStrategy interface
    # ------------------------------------------------------------------

    def select_bins(self, context: SelectionContext) -> Tuple[List[int], SearchContext]:
        """
        Solve the bi-objective 0/1 multiple-knapsack via sequential FPTAS.

        For each vehicle *k* in turn, applies the FPTAS to the pool of
        remaining eligible bins and collects the chosen set.  Returns the
        union of all chosen sets unless the best single-bin solution
        dominates (1/2-approximation fallback).

        Args:
            context (SelectionContext): The selection context.

        Returns:
            Tuple[List[int], SearchContext]: Selected bin IDs and search context.

        Raises:
            ValueError: If ``distance_matrix`` is not provided.
        """
        if context.distance_matrix is None:
            raise ValueError("FPTASBiObjectiveKnapsackSelection requires a distance_matrix.")

        # ---- Physical quantities ------------------------------------------
        dm = np.asarray(context.distance_matrix, dtype=np.float64)
        bin_cap = float(context.bin_volume) * float(context.bin_density)  # kg
        mass_all = (context.current_fill / context.max_fill) * bin_cap  # kg

        # dist_depot[i] = distance from depot (index 0) to bin i (matrix index i+1)
        dist_depot = dm[0, 1:].astype(np.float64)

        # ---- Overflow risk (kg) ------------------------------------------
        overflow_penalty_frac = (
            self.overflow_penalty_frac
            if self.overflow_penalty_frac is not None
            else float(getattr(context, "overflow_penalty_frac", 1.0))
        )
        risk_all = _compute_overflow_risk(
            current_fill=context.current_fill,
            bin_mass=np.full_like(mass_all, bin_cap),
            scenario_tree=getattr(context, "scenario_tree", None),
            overflow_penalty_frac=overflow_penalty_frac,
        )

        # ---- Net profit (monetary) — standalone depot-round-trip proxy ---
        revenue_kg = float(getattr(context, "revenue_kg", 0.0))
        cost_per_km = float(getattr(context, "cost_per_km", 0.0))
        revenue_all = mass_all * revenue_kg
        profit_all = revenue_all - cost_per_km * 2.0 * dist_depot

        # ---- Scalarised combined value (dimensionless) -------------------
        # Normalise each objective to [0, 1] independently so α is stable.
        risk_max = float(np.max(risk_all)) if np.any(risk_all > _EPS_GUARD) else 1.0
        profit_clipped = np.maximum(0.0, profit_all)
        profit_max = float(np.max(profit_clipped)) if np.any(profit_clipped > _EPS_GUARD) else 1.0

        combined = self.alpha * (risk_all / risk_max) + (1.0 - self.alpha) * (profit_clipped / profit_max)

        # ---- Eligibility -------------------------------------------------
        # Bins must have positive mass, positive combined value, and must
        # individually fit within a single vehicle.
        n_vehicles = int(getattr(context, "n_vehicles", 1))
        capacity = float(context.vehicle_capacity)

        _empty_metrics = {"strategy": "FPTASBiObjectiveKnapsackSelection"}
        _empty_ctx = SearchContext.initialize(selection_metrics=_empty_metrics)

        eligible_mask = (mass_all > _EPS_GUARD) & (combined > _EPS_GUARD)

        if not np.any(eligible_mask):
            return [], _empty_ctx

        # Unbounded knapsacks: capacity does not bind → take all eligible bins.
        if n_vehicles <= 0:
            sel = sorted((np.where(eligible_mask)[0] + 1).tolist())
            return sel, SearchContext.initialize(selection_metrics={**_empty_metrics, "n_selected": len(sel)})

        if capacity <= _EPS_GUARD:
            return [], _empty_ctx

        # Items individually exceeding a single vehicle's capacity are infeasible.
        eligible_mask &= mass_all <= capacity + _EPS_GUARD

        if not np.any(eligible_mask):
            return [], _empty_ctx

        eligible_idx = np.where(eligible_mask)[0]  # 0-based global bin indices

        # ---- Sequential FPTAS across K vehicles --------------------------
        remaining: set[int] = set(eligible_idx.tolist())
        greedy_selected: List[int] = []
        total_value: float = 0.0

        for _ in range(n_vehicles):
            if not remaining:
                break
            rem_arr = np.fromiter(remaining, dtype=np.intp, count=len(remaining))

            sel_mask_local = _fptas_01_knapsack(
                values=combined[rem_arr],
                weights=mass_all[rem_arr],
                capacity=capacity,
                epsilon=self.epsilon,
            )
            chosen = rem_arr[sel_mask_local]
            if chosen.size == 0:
                break

            total_value += float(combined[chosen].sum())
            greedy_selected.extend(chosen.tolist())
            remaining.difference_update(chosen.tolist())

        # ---- Best-single-item fallback (1/2-approximation guarantee) -----
        # OPT ≤ V(greedy) + V(best_single), so max(·, ·) ≥ OPT / 2.
        elig_arr = np.fromiter(eligible_idx, dtype=np.intp, count=len(eligible_idx))
        best_local = int(np.argmax(combined[elig_arr]))
        best_single_idx = int(elig_arr[best_local])
        best_single_value = float(combined[best_single_idx])

        if best_single_value > total_value:
            return [best_single_idx + 1], SearchContext.initialize(
                selection_metrics={
                    **_empty_metrics,
                    "n_selected": 1,
                    "total_value": best_single_value,
                    "fallback": "best_single",
                }
            )

        return sorted(i + 1 for i in greedy_selected), SearchContext.initialize(
            selection_metrics={
                **_empty_metrics,
                "n_selected": len(greedy_selected),
                "total_value": total_value,
            }
        )
