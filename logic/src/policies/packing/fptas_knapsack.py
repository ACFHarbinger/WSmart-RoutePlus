"""

DP helpers live in ``fptas_knapsack_dp``; this module keeps ``FPTASKnapsackSelection`` and re-exports those names.
FPTAS Bi-Objective Multiple-Knapsack Packing Algorithm Module.

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
        Packing algorithm.
    _EPS_GUARD: Epsilon guard for floating-point comparisons.
    _DIST_EPS: Distance epsilon for insertion-distance lower-bound.
    _MAX_DP_VALUES: Maximum number of DP values.

Example:

    >>> from logic.src.policies.packing.fptas_bi_objective_knapsack import (
    ...     FPTASBiObjectiveKnapsackSelection,
    ... )
    >>> strategy = FPTASBiObjectiveKnapsackSelection(epsilon=0.05, alpha=0.6)
    >>> bins, ctx = strategy.pack(context)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from logic.src.enums import PolicyTag
from logic.src.interfaces.context import PackingContext, SearchContext
from logic.src.interfaces.packing import IPackingAlgorithm
from logic.src.policies.packing.base import PackingRegistry
from logic.src.utils.registry import ALL

from .fptas_knapsack_dp import (
    _EPS_GUARD as _EPS_GUARD,
)
from .fptas_knapsack_dp import (
    _compute_overflow_risk as _compute_overflow_risk,
)
from .fptas_knapsack_dp import (
    _fptas_01_knapsack as _fptas_01_knapsack,
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Overflow risk (mirrors MIPKnapsackSelection._compute_overflow_risk exactly)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# FPTAS core: (1-ε)-approximation for the 0/1 knapsack problem
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


@ALL.register(
    PolicyTag.PACKING,
    PolicyTag.MATHEURISTIC,
    PolicyTag.HEURISTIC,
    PolicyTag.PROFIT_AWARE,
)
@PackingRegistry.register("fptas_knapsack")
class FPTASKnapsackSelection(IPackingAlgorithm):
    """FPTAS bi-objective multiple-knapsack packing algorithm.

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
    # IPackingAlgorithm interface
    # ------------------------------------------------------------------

    def pack(self, context: PackingContext) -> Tuple[List[int], SearchContext]:
        """
        Solve the bi-objective 0/1 multiple-knapsack via sequential FPTAS.

        For each vehicle *k* in turn, applies the FPTAS to the pool of
        remaining eligible bins and collects the chosen set.  Returns the
        union of all chosen sets unless the best single-bin solution
        dominates (1/2-approximation fallback).

        Args:
            context: The selection context.

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
