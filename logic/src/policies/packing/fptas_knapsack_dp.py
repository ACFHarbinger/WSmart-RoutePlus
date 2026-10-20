"""Overflow-risk scoring and FPTAS 0/1 knapsack DP.

Moved from ``logic.src.policies.packing.fptas_knapsack``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


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
        current_fill: Current fill levels as percentages (shape ``(n_bins,)``).
        bin_mass: Full waste capacity in kg for each bin.
        scenario_tree: Optional ``ScenarioTree`` from the prediction module.
            Must expose ``get_scenarios_at_day(day)`` and ``horizon``.
        overflow_penalty_frac: Occurrence penalty as a fraction of full bin
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
        values: Non-negative per-item values (shape ``(n,)``).
        weights: Non-negative per-item weights (shape ``(n,)``).
        capacity: Knapsack weight capacity (scalar).
        epsilon: Approximation parameter ε ∈ (0, 1).

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


_EPS_GUARD: float = 1e-9  # float comparison / zero-division guard

_DIST_EPS: float = 1e-9  # insertion-distance lower-bound

_MAX_DP_VALUES: int = 100_000  # DP table width ceiling (memory guard)
