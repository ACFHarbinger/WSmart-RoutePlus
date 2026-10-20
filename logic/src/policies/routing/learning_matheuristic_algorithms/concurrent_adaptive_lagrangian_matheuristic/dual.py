"""
Dual-bound tracking strategies for the Lagrangian coordinator.

Two strategies are supported, selectable via DualBoundParams.strategy:

    "ema":     Best-so-far Lagrangian value, with a per-period EMA of the
               Lagrangian dual contributions.  Updates to period t's EMA are
               GATED: only accepted if the routing subproblem returned a tour
               within a quality threshold of the period incumbent.  This
               prevents the lookahead from being poisoned by a single
               transient bad RS result, and isolates contamination across
               periods (a bad RS in period 3 cannot degrade period 1's EMA).

    "bundle":  Proximal bundle method.  Maintains a bundle of subgradients
               and solves a small regularised QP at each update to choose a
               stabilising step.  Stronger dual, heavier per-update cost.

Both strategies expose the same interface:
    - submit(period, lambdas, subgrad_i_t, tour_quality_ratio) -> None
    - current_dual_bound() -> float
    - suggest_step(current_lambdas, aggregate_subgrad) -> np.ndarray
    - serious_step(lambdas) / null_step() -> informational callbacks

Attributes:
    DualBoundTracker: Interface for dual-bound tracking.
    EMADualBoundTracker: EMA-based dual-bound tracker.
    BundleEntry: Entry in the proximal bundle.
    ProximalBundleDualBoundTracker: Proximal bundle dual-bound tracker.
    build_dual_bound_tracker: Builder function for dual-bound trackers.

Example:
    >>> tracker = EMADualBoundTracker(DualBoundParams(), LagrangianParams(), 1, 1)
    >>> tracker.submit(0, np.array([1]), np.array([1]), 1, 1)
    True
    >>> tracker.current_dual_bound()
    1.0
    >>> tracker.suggest_step(np.array([1]), np.array([1]))
    (array([0.9999]), 0.01)
    >>> tracker.serious_step(np.array([1]))
    >>> tracker.null_step()


The tracker interface and the proximal bundle method live in ``proximal_bundle`` and are re-exported here. The EMA tracker stays in this module.
"""

from __future__ import annotations

from threading import Lock
from typing import Optional, Tuple

import numpy as np

from .params import DualBoundParams, LagrangianParams
from .proximal_bundle import (
    BundleEntry as BundleEntry,
)
from .proximal_bundle import (
    DualBoundTracker as DualBoundTracker,
)
from .proximal_bundle import (
    ProximalBundleDualBoundTracker as ProximalBundleDualBoundTracker,
)

# ---------------------------------------------------------------------------
# EMA tracker
# ---------------------------------------------------------------------------


class EMADualBoundTracker(DualBoundTracker):
    """
    Best-so-far Lagrangian value plus gated per-period EMA of subgradients.

    The EMA is ONLY updated for period t when the associated RS tour quality
    ratio is <= `ema_quality_threshold`.  This implements the refinement
    agreed during planning.

    Attributes:
        params: Dual bound parameters
        lag_params: Lagrangian parameters
        n_bins: Number of bins
        horizon: Horizon
        _ema: Per-period EMA of the subgradient
        _has_observation: Whether the EMA has been observed for each period
        _best_lagrangian: Running best Lagrangian value
        _period_contribs: Lagrangian value contributions from each period
        _lock: Lock for thread safety
    """

    def __init__(
        self,
        dual_params: DualBoundParams,
        lag_params: LagrangianParams,
        n_bins: int,
        horizon: int,
    ):
        """Initializes the EMA dual bound tracker.

        Args:
            dual_params: Parameters for the tracker.
            lag_params: Lagrangian-specific parameters.
            n_bins: Number of customer bins.
            horizon: Planning horizon.
        """
        self.params = dual_params
        self.lag_params = lag_params
        self.n_bins = n_bins
        self.horizon = horizon

        # Per-period EMA of the subgradient.
        self._ema = np.zeros((n_bins, horizon), dtype=float)
        self._has_observation = np.zeros(horizon, dtype=bool)

        # Running best Lagrangian value (valid lower bound since Lagrangian is
        # a minorant of the optimum).
        self._best_lagrangian = -np.inf
        self._period_contribs = np.zeros(horizon, dtype=float)

        self._lock = Lock()

    # ----- DualBoundTracker interface -----

    def submit(
        self,
        period: int,
        lambdas: np.ndarray,
        subgrad: np.ndarray,
        lagrangian_value_contrib: float,
        tour_quality_ratio: float,
    ) -> bool:
        """Submits a per-period subgradient for EMA aggregation.

        Args:
            period: Period index.
            lambdas: Multipliers at which subgradient was evaluated.
            subgrad: Per-period subgradient.
            lagrangian_value_contrib: Period's contribution to Lagrangian objective.
            tour_quality_ratio: Ratio of RS tour cost to period incumbent.

        Returns:
            bool: True if the submission was accepted (gated by quality ratio).
        """
        with self._lock:
            gated = tour_quality_ratio <= self.params.ema_quality_threshold
            if gated:
                alpha = self.params.ema_alpha
                if not self._has_observation[period]:
                    self._ema[:, period] = subgrad
                    self._has_observation[period] = True
                else:
                    self._ema[:, period] = (1 - alpha) * self._ema[:, period] + alpha * subgrad
                self._period_contribs[period] = lagrangian_value_contrib
                total = float(np.sum(self._period_contribs))
                if total > self._best_lagrangian:
                    self._best_lagrangian = total
                return True
            return False

    def current_dual_bound(self) -> float:
        """Returns the current best Lagrangian lower bound.

        Returns:
            float: Best dual bound found.
        """
        with self._lock:
            return self._best_lagrangian if np.isfinite(self._best_lagrangian) else float("-inf")

    def suggest_step(self, current_lambdas: np.ndarray, aggregate_subgrad: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Polyak stepsize:  alpha_k = mu_k * (UB - L_k) / ||g_k||^2

        UB is supplied by the coordinator (best primal).  Here we take it to
        be max(|L_k|, 1) * 2 as a neutral placeholder; the coordinator will
        usually override by calling :meth:`polyak_step` directly.

        Args:
            current_lambdas: Current multipliers.
            aggregate_subgrad: Aggregate subgradient.

        Returns:
            Tuple[np.ndarray, float]: New lambda vector and step size.
        """
        g = aggregate_subgrad
        g_norm_sq = float(np.dot(g.ravel(), g.ravel()))
        if g_norm_sq < 1e-12:
            return current_lambdas, 0.0
        ub_proxy = max(abs(self._best_lagrangian), 1.0) * 2.0
        gap = max(0.0, ub_proxy - self._best_lagrangian)
        mu = self.lag_params.polyak_mu_default
        step = mu * gap / g_norm_sq
        new_lambdas = current_lambdas + step * g
        return new_lambdas, step

    # ----- Public helpers -----

    def polyak_step(
        self,
        current_lambdas: np.ndarray,
        aggregate_subgrad: np.ndarray,
        upper_bound: float,
        mu: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """Explicit Polyak step using a caller-supplied UB (best primal).

        Args:
            current_lambdas: Current multipliers.
            aggregate_subgrad: Aggregate subgradient.
            upper_bound: Upper bound (best primal).
            mu: Polyak step parameter (default: lag_params.polyak_mu_default).

        Returns:
            Tuple[np.ndarray, float]: New lambda vector and step size.
        """
        g = aggregate_subgrad
        g_norm_sq = float(np.dot(g.ravel(), g.ravel()))
        if g_norm_sq < 1e-12:
            return current_lambdas, 0.0
        gap = max(0.0, upper_bound - self._best_lagrangian)
        mu = mu if mu is not None else self.lag_params.polyak_mu_default
        step = mu * gap / g_norm_sq
        new_lambdas = current_lambdas + step * g
        return new_lambdas, step

    def ema_snapshot(self) -> np.ndarray:
        """Returns a snapshot of the current EMA subgradients.

        Returns:
            np.ndarray: Copy of the internal EMA array.
        """
        with self._lock:
            return self._ema.copy()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_dual_bound_tracker(
    dual_params: DualBoundParams,
    lag_params: LagrangianParams,
    n_bins: int,
    horizon: int,
) -> DualBoundTracker:
    """Factory function to build the requested DualBoundTracker strategy.

    Args:
        dual_params: Parameters specifying the strategy.
        lag_params: Lagrangian parameters.
        n_bins: Number of customer bins.
        horizon: Planning horizon.

    Returns:
        DualBoundTracker: The instantiated tracker.
    """
    strat = dual_params.strategy.lower()
    if strat == "ema":
        return EMADualBoundTracker(dual_params, lag_params, n_bins, horizon)
    if strat == "bundle":
        return ProximalBundleDualBoundTracker(dual_params, lag_params, n_bins, horizon)
    raise ValueError(f"Unknown dual-bound strategy '{dual_params.strategy}' (expected 'ema' or 'bundle').")
