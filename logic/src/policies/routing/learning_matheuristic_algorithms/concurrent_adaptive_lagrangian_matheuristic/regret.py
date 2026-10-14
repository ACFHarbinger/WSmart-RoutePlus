"""
Adaptive regret-based preprocessing.

Uses the per-bin early-regret values (rho[i, 0] + rho[i, 1]) from the
lookahead module to identify bins that are overflow-bound -- i.e., bins
where deferring past period 0 or 1 strictly loses expected revenue
because they are about to saturate.

Escalation policy (adaptive):

    phase = "soft" (initial):
        Bias the knapsack's objective coefficients on top-decile early-regret
        bins upward by `soft_bias_coefficient`.  No variable fixing.  This
        nudges the selection toward early visits without constraining the
        feasible region.

    phase = "hard" (after `escalation_patience` iterations of primal
                    stagnation):
        Forcibly fix the top-fraction of early-regret bins to be visited in
        periods [0, hard_fix_max_periods).  Implemented as a hard lower-bound
        constraint on the selection variables of those periods.

The module does NOT itself modify a Gurobi model; it produces a `RegretPlan`
that the selection-subproblem builder applies when constructing the Master.

Attributes:
----------
    soft_bias : np.ndarray of shape (N, T)
        Additive bias to apply to the knapsack's objective coefficients.
        Zero for bins / periods not in the high-regret set.
    hard_fix : Dict[int, List[int]]
        Map from bin_id -> list of periods among which exactly one must be
        chosen (sum of x_K[bin, periods] >= 1).  Empty when phase == "soft".
    phase : str
        "soft" or "hard".
    high_regret_bin_ids : Set[int]
        Bins flagged by the regret preprocessing (used by downstream
        diagnostics / reward shaping).

Example:
    >>> # Assume early_regret is computed from the lookahead module.
    >>> N = 100
    >>> T = 5
    >>> regret_params = RegretParams(enabled=True, ...)
    >>> preproc = RegretPreprocessor(regret_params, n_bins=N, horizon=T)
    >>> early_regret = np.random.uniform(0, 100, size=N)
    >>> plan = preproc.build_plan(early_regret)
    >>> print(plan.phase)
    "soft"
    >>> # Hard phase after enough primal stagnation
    >>> for _ in range(10):
    ...     preproc.observe_iteration(primal_improved=False)
    >>> plan = preproc.build_plan(early_regret)
    >>> print(plan.phase)
    "hard"
    >>> # plan.hard_fix might be {42: [0, 1, 2, 3, 4], ...}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

import numpy as np

from .params import RegretParams


@dataclass
class RegretPlan:
    """
    Concrete instructions for the selection-subproblem builder.

    Attributes:
        soft_bias : np.ndarray of shape (N, T)
            Additive bias to apply to the knapsack's objective coefficients.
            Zero for bins / periods not in the high-regret set.
        hard_fix : Dict[int, List[int]]
            Map from bin_id -> list of periods among which exactly one must be
            chosen (sum of x_K[bin, periods] >= 1).  Empty when phase == "soft".
        phase : str
            "soft" or "hard".
        high_regret_bin_ids : Set[int]
            Bins flagged by the regret preprocessing (used by downstream
            diagnostics / reward shaping).
    """

    soft_bias: np.ndarray
    hard_fix: Dict[int, List[int]]
    phase: str
    high_regret_bin_ids: Set[int] = field(default_factory=set)


class RegretPreprocessor:
    """
    Maintains escalation state across outer iterations.

    Attributes:
        params : RegretParams
            Regret parameters.
        n_bins : int
            Number of bins.
        horizon : int
            Horizon.
        _phase : str
            Current phase ("soft" or "hard").
        _iters_since_improvement : int
            Number of iterations since last improvement.

    Usage
    -----
    preproc = RegretPreprocessor(params, n_bins=N, horizon=T)

    for outer_iter in range(...):
        plan = preproc.build_plan(early_regret=tables.early_regret)
        # ... run matheuristic with plan ...
        preproc.observe_iteration(primal_improved=bool(obj_improved))
    """

    def __init__(self, params: RegretParams, n_bins: int, horizon: int):
        """__init__ docstring.

        Args:
            params: Regret parameters.
            n_bins: Number of bins.
            horizon: Horizon.

        Returns:
            None
        """
        self.params = params
        self.n_bins = n_bins
        self.horizon = horizon
        self._phase = "soft"
        self._iters_since_improvement = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def phase(self) -> str:
        """Returns the current phase.

        Returns:
            str: Current phase ("soft" or "hard").
        """
        return self._phase

    def build_plan(self, early_regret: np.ndarray) -> RegretPlan:
        """
        Produce a plan for the current outer iteration.

        Args:
            early_regret : np.ndarray of shape (N,)
                Sum of rho[i, 0] + rho[i, 1] from the lookahead tables.

        Returns:
            RegretPlan: Plan for the current outer iteration.
        """
        if not self.params.enabled or self.n_bins == 0:
            return RegretPlan(
                soft_bias=np.zeros((self.n_bins, self.horizon), dtype=float),
                hard_fix={},
                phase="soft",
                high_regret_bin_ids=set(),
            )

        N = self.n_bins
        T = self.horizon

        # Identify the top-fraction bins by early regret.
        # Use the phase's threshold: soft always operates on the top decile;
        # hard uses the params-configured fraction (default 10%).
        k_soft = max(1, int(np.ceil(0.10 * N)))
        k_hard = max(1, int(np.ceil(self.params.hard_fix_top_fraction * N)))

        # Guard: if no bin has strictly positive regret, return an empty plan.
        if not np.any(early_regret > 0):
            return RegretPlan(
                soft_bias=np.zeros((N, T), dtype=float),
                hard_fix={},
                phase=self._phase,
                high_regret_bin_ids=set(),
            )

        # Rank bins by early regret descending.
        order = np.argsort(-early_regret)

        soft_bias = np.zeros((N, T), dtype=float)
        hard_fix: Dict[int, List[int]] = {}
        flagged: Set[int] = set()

        # --- Soft contribution (always active while there is regret) ---
        soft_set = [i for i in order[:k_soft] if early_regret[i] > 0]
        for i in soft_set:
            # Apply the bias only to the earliest periods that drive the regret
            # -- i.e., periods where V[i, t] is highest.  Without V here (kept
            # outside this module for separation of concerns), we default to
            # periods 0 and 1, which is what "early_regret" aggregates over.
            periods_to_boost = [0] + ([1] if T >= 2 else [])
            for t in periods_to_boost:
                soft_bias[i, t] += self.params.soft_bias_coefficient
            flagged.add(int(i))

        # --- Hard contribution (only in hard phase) ---
        if self._phase == "hard":
            hard_periods = list(range(min(self.params.hard_fix_max_periods, T)))
            if not hard_periods:
                hard_periods = [0]
            hard_set = [i for i in order[:k_hard] if early_regret[i] > 0]
            for i in hard_set:
                hard_fix[int(i)] = list(hard_periods)
                flagged.add(int(i))

        return RegretPlan(
            soft_bias=soft_bias,
            hard_fix=hard_fix,
            phase=self._phase,
            high_regret_bin_ids=flagged,
        )

    def observe_iteration(self, primal_improved: bool) -> None:
        """Update escalation state at the end of an outer iteration.

        Args:
            primal_improved: Whether the primal objective improved in the last iteration.

        Returns:
            None
        """
        if primal_improved:
            self._iters_since_improvement = 0
            # Relax back to soft if we're climbing again (optional; keeps the
            # method work-conserving against oscillating instances).
            self._phase = "soft"
        else:
            self._iters_since_improvement += 1
            if self._phase == "soft" and self._iters_since_improvement >= self.params.escalation_patience:
                self._phase = "hard"

    def force_phase(self, phase: str) -> None:
        """Manual override, for testing / ablations.

        Args:
            phase: Phase to force ("soft" or "hard").

        Returns:
            None
        """
        assert phase in ("soft", "hard")
        self._phase = phase
