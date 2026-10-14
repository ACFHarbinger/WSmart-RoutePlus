"""
Insertion-cost oracle: shared valuation layer between the knapsack (selection)
and routing subproblems.

For each bin i and period t, maintains:

    Delta[i, t] = cheapest insertion cost of i into period t's best-known tour.

This is NOT a valid lower bound on marginal routing cost (insertions interact),
so it is only used as an LP coefficient correction, not as a cut.

Gated EMA updates
-----------------
When a routing worker returns an improving tour for period t, we update
Delta[:, t] by EMA.  When it returns a non-improving tour (within threshold),
we use a much smaller EMA weight.  Tours worse than the threshold are
REJECTED entirely -- they poison the lookahead by pushing high insertion
costs from a bad local minimum into the knapsack.

This matches the discussion:

    Only update the EMA for bin i if the routing subproblem returns a
    solution that is an improvement or within X% of the incumbent, to
    prevent the Lookahead from being poisoned by temporary sub-optimal
    routes.

Attributes:
---------
    PeriodIncumbent: Best-known tour and cost for a single period.
    InsertionCostOracle: Shared, thread-safe insertion-cost table.

Example:
--------
    >>> oracle = InsertionCostOracle(n_bins=10, horizon=3)
    >>> oracle.update_from_routing(
    ...     period=0,
    ...     tour=[0, 1, 2, 0],
    ...     tour_cost=100.0,
    ...     selection=[1, 2],
    ...     insertion_costs_for_unselected={3: 10.0, 4: 20.0},
    ... )
    'accepted_improving'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List

import numpy as np


@dataclass
class PeriodIncumbent:
    """Best-known tour and cost for a single period.

    Attributes:
    ----------
    tour: List[int]
        Tour for the period (depot-first-last).  List of bin indices.
    cost: float
        Total travel cost for this tour.
    selection: frozenset
        Set of bin indices in the tour.
    """

    tour: List[int] = field(default_factory=list)
    cost: float = float("inf")
    selection: frozenset = field(default_factory=frozenset)


class InsertionCostOracle:
    """
    Shared, thread-safe insertion-cost table.

    Attributes:
    ----------
    n_bins : int
        Number of optional bins (indices 1..N; index 0 is the depot).
    horizon : int
        Number of periods.
    alpha : float
        EMA smoothing factor in (0, 1].  Higher = more reactive.
        Applied to improving updates.
    quality_threshold : float
        A new observation is fully accepted if RS tour cost <= incumbent * 1.0,
        partially accepted (at alpha/4) if RS tour cost <= incumbent * threshold,
        rejected otherwise.  E.g. 1.05 = accept within 5% of incumbent.
    delta : np.ndarray
        Delta values for each bin and period.
    incumbents : List[PeriodIncumbent]
        Incumbent solutions for each period.
    _has_observation : np.ndarray
        Whether the Delta value for each bin and period has been observed.
    _lock : Lock
        Lock for thread safety.
    """

    def __init__(
        self,
        n_bins: int,
        horizon: int,
        alpha: float = 0.3,
        quality_threshold: float = 1.05,
        default_delta: float = 0.0,
    ):
        """Init docstring.

        Args:
            n_bins (int): Number of optional bins (indices 1..N; index 0 is the depot).
            horizon (int): Number of periods.
            alpha (float): EMA smoothing factor in (0, 1].  Higher = more reactive.
                Applied to improving updates.
            quality_threshold (float): A new observation is fully accepted if RS tour cost <= incumbent * 1.0,
                partially accepted (at alpha/4) if RS tour cost <= incumbent * threshold,
                rejected otherwise.  E.g. 1.05 = accept within 5% of incumbent.
            default_delta (float): Initial Delta value before any observations.  Use a large-but-finite
                number so the knapsack sees some cost penalty at startup.
        """
        self.n_bins = n_bins
        self.horizon = horizon
        self.alpha = alpha
        self.quality_threshold = quality_threshold

        # Delta[i, t] for i in [0, n_bins), t in [0, horizon)
        self.delta = np.full((n_bins, horizon), default_delta, dtype=float)
        self._has_observation = np.zeros((n_bins, horizon), dtype=bool)

        # Per-period incumbents (for gating).
        self.incumbents: List[PeriodIncumbent] = [PeriodIncumbent() for _ in range(horizon)]

        self._lock = Lock()

    # ------------------------------------------------------------------
    # Read-only views (for the knapsack / selection subproblem)
    # ------------------------------------------------------------------

    def snapshot(self) -> np.ndarray:
        """Return a copy of the current Delta table (N, T).

        Returns:
            np.ndarray: Copy of the Delta table.
        """
        with self._lock:
            return self.delta.copy()

    def get_incumbent(self, period: int) -> PeriodIncumbent:
        """Get the incumbent solution for a given period.

        Args:
            period (int): Period to get the incumbent for.

        Returns:
            PeriodIncumbent: Incumbent solution for the given period.
        """
        with self._lock:
            inc = self.incumbents[period]
            # Shallow copy -- tour is a list, selection is a frozenset.
            return PeriodIncumbent(tour=list(inc.tour), cost=inc.cost, selection=inc.selection)

    # ------------------------------------------------------------------
    # Write path (called from routing workers)
    # ------------------------------------------------------------------

    def update_from_routing(
        self,
        period: int,
        tour: List[int],
        tour_cost: float,
        selection: List[int],
        insertion_costs_for_unselected: Dict[int, float],
    ) -> str:
        """
        Ingest a routing subproblem result.

        Args:
        ----------
        period : int
        tour : list of bin indices (depot at start/end; the tour for period t)
        tour_cost : float
        selection : list of bin indices visited in this tour (== tour minus depots)
        insertion_costs_for_unselected : dict {bin_id: cheapest-insertion-cost}
            The RS worker computes these cheaply after finishing the tour
            (one O(|tour|) scan per candidate bin).  We only need them for
            bins NOT in the current selection, since bins in the selection
            have "insertion cost" = 0 tautologically.

        Returns:
        -------
        outcome : {"accepted_improving", "accepted_partial", "rejected"}
            For logging / diagnostics.
        """
        with self._lock:
            inc = self.incumbents[period]

            if tour_cost < inc.cost - 1e-9:
                # Strict improvement.  Overwrite incumbent, full EMA update.
                self.incumbents[period] = PeriodIncumbent(
                    tour=list(tour), cost=tour_cost, selection=frozenset(selection)
                )
                weight = self.alpha
                outcome = "accepted_improving"
            elif inc.cost == float("inf") or tour_cost <= inc.cost * self.quality_threshold:
                # First observation, or within quality threshold -> partial update.
                if inc.cost == float("inf"):
                    self.incumbents[period] = PeriodIncumbent(
                        tour=list(tour), cost=tour_cost, selection=frozenset(selection)
                    )
                weight = self.alpha / 4.0
                outcome = "accepted_partial"
            else:
                # Reject: bad local minimum, don't contaminate the oracle.
                return "rejected"

            # Update Delta for every bin the RS gave us a cost for.
            # Bin ids are 1-based (matching TPKS convention); convert to row idx.
            for bin_id, cost in insertion_costs_for_unselected.items():
                row = bin_id - 1
                if 0 <= row < self.n_bins:
                    if not self._has_observation[row, period]:
                        self.delta[row, period] = cost
                        self._has_observation[row, period] = True
                    else:
                        self.delta[row, period] = (1 - weight) * self.delta[row, period] + weight * cost

            # Bins that ARE selected have effective Delta = 0 (already in tour).
            for bin_id in selection:
                row = bin_id - 1
                if 0 <= row < self.n_bins:
                    target = 0.0
                    if not self._has_observation[row, period]:
                        self.delta[row, period] = target
                        self._has_observation[row, period] = True
                    else:
                        self.delta[row, period] = (1 - weight) * self.delta[row, period] + weight * target

            return outcome

    # ------------------------------------------------------------------
    # Helpers for routing workers
    # ------------------------------------------------------------------

    @staticmethod
    def cheapest_insertion_cost(dist_matrix: np.ndarray, tour: List[int], bin_id: int) -> float:
        """Compute the cheapest insertion cost of `bin_id` into `tour`:

            min over edges (u, v) in tour of  d(u, bin_id) + d(bin_id, v) - d(u, v)

        O(|tour|).  Returns 0.0 if tour is empty or has only the depot.

        Args:
            dist_matrix (np.ndarray): Distance matrix.
            tour (List[int]): Tour.
            bin_id (int): Bin index.

        Returns:
            float: Cheapest insertion cost.
        """
        if len(tour) < 2:
            return 0.0
        best = float("inf")
        for k in range(len(tour) - 1):
            u, v = tour[k], tour[k + 1]
            cost = dist_matrix[u, bin_id] + dist_matrix[bin_id, v] - dist_matrix[u, v]
            if cost < best:
                best = cost
        return max(0.0, best)

    @staticmethod
    def batch_insertion_costs(
        dist_matrix: np.ndarray,
        tour: List[int],
        candidates: List[int],
    ) -> Dict[int, float]:
        """Vectorised batch variant of `cheapest_insertion_cost` over candidates.

        Args:
            dist_matrix (np.ndarray): Distance matrix.
            tour (List[int]): Tour.
            candidates (List[int]): Candidate bin indices.

        Returns:
            Dict[int, float]: Dictionary mapping bin indices to their cheapest insertion costs.
        """
        if len(tour) < 2 or not candidates:
            return {c: 0.0 for c in candidates}
        # Build a (|edges|,) vector of d(u,v) and (|edges|, |cand|) of d(u, c) + d(c, v).
        u_arr = np.asarray(tour[:-1], dtype=int)
        v_arr = np.asarray(tour[1:], dtype=int)
        cand_arr = np.asarray(candidates, dtype=int)

        d_uv = dist_matrix[u_arr, v_arr]  # (E,)
        d_uc = dist_matrix[u_arr[:, None], cand_arr[None, :]]  # (E, C)
        d_cv = dist_matrix[cand_arr[None, :], v_arr[:, None]]  # (E, C)
        insertion = d_uc + d_cv - d_uv[:, None]  # (E, C)
        insertion = np.maximum(insertion, 0.0)
        best_per_cand = insertion.min(axis=0)  # (C,)
        return dict(zip(candidates, best_per_cand.tolist(), strict=False))
