"""
Lagrangian multiplier state and per-period async update logic.

The coupling is x^K_{i,t} = x^R_{i,t}; we relax this with multipliers
lambda[i, t].  The knapsack sees effective prize  V + lambda (minus the
insertion-cost oracle correction, applied elsewhere); the routing sees
effective prize  V - lambda.

Update logic
------------
When a routing worker finishes period t and returns x^R_{:,t}, the coordinator
(running on the same thread as that worker -- option 1b, no dedicated
coordinator thread) does the following:

    1. Read the most recent x^K_{:,t} from the shared selection state.
    2. Compute local subgradient g[:, t] = x^K[:, t] - x^R[:, t].
    3. Forward the subgradient to the DualBoundTracker.
    4. Ask the tracker to suggest a step; update lambdas[:, t] in place.

This is "per-period async": lambdas[:, t'] for t' != t are untouched in the
same call.  Outer-iteration synchronous updates are also supported for the
bundle strategy, which needs a consistent multiplier vector for its QP.

Attributes:
    LagrangianState: Shared, thread-safe state for the Lagrangian coordinator.
    LagrangianCoordinator: Manages the Lagrangian state and multiplier updates.

Example:
    >>> lagrangian_state = LagrangianState(n_bins=2, horizon=3)
    >>> lagrangian_coordinator = LagrangianCoordinator(lagrangian_state, tracker, lag_params)
    >>> lagrangian_coordinator.set_knapsack_selection(x_K)
    >>> lagrangian_coordinator.set_routing_selection(0, x_R_column)
    >>> lagrangian_coordinator.submit_period_result(0, lagrangian_value_contrib, tour_quality_ratio, upper_bound)
    (True, 0.5)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Optional, Tuple

import numpy as np

from .dual import (
    DualBoundTracker,
    EMADualBoundTracker,
    ProximalBundleDualBoundTracker,
)
from .params import LagrangianParams


@dataclass
class LagrangianState:
    """Shared, thread-safe state for the Lagrangian coordinator.

    Attributes:
    -----------
    n_bins : int
        The number of bins.
    horizon : int
        The horizon.
    lambdas : np.ndarray
        The Lagrangian multipliers.
    x_K : np.ndarray
        The solution to the knapsack subproblem.
    x_R : np.ndarray
        The solution to the routing subproblem.
    gamma : np.ndarray
        The trust weight on insertion-cost oracle, annealed over iterations.
    """

    x_K: np.ndarray = field(init=False)  # (N, T), from the knapsack side
    x_R: np.ndarray = field(init=False)  # (N, T), from the routing side

    # Per-period gamma (trust weight on insertion-cost oracle), annealed over
    # iterations.  Allowed to differ per-period so we can ramp one period at
    # a time if its RS is stable earlier than others.
    gamma: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Initializes state tensors after dataclass initialization.

        Returns:
        --------
        None
        """
        self.lambdas = np.zeros((self.n_bins, self.horizon), dtype=float)
        self.x_K = np.zeros((self.n_bins, self.horizon), dtype=float)
        self.x_R = np.zeros((self.n_bins, self.horizon), dtype=float)
        self.gamma = np.ones(self.horizon, dtype=float)

    def snapshot(self) -> Dict[str, np.ndarray]:
        """Returns a copy of the current state tensors.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing copies of state tensors.
        """
        return {
            "lambdas": self.lambdas.copy(),
            "x_K": self.x_K.copy(),
            "x_R": self.x_R.copy(),
            "gamma": self.gamma.copy(),
        }


class LagrangianCoordinator:
    """
    Multiplier state manager.

    Responsibilities
    ----------------
    * Maintain lambda[i, t].
    * Receive per-period subgradients from RS workers (push model).
    * Apply the stepsize via the DualBoundTracker.
    * Clip multipliers to [lambda_min, lambda_max].
    * Anneal gamma (insertion-cost trust) on request.

    Attributes:
    -----------
    state : LagrangianState
        Shared state for the Lagrangian coordinator.
    tracker : DualBoundTracker
        The dual bound tracking and update engine.
    params : LagrangianParams
        Hyperparameters for the matheuristic.
    _lock : Lock
        Lock for thread-safe access to the state.
    _outer_iter : int
        The current outer iteration counter.
    """

    def __init__(
        self,
        state: LagrangianState,
        tracker: DualBoundTracker,
        lag_params: LagrangianParams,
    ):
        """Initializes the Lagrangian coordinator.

        Args:
            state (LagrangianState): Shared state object.
            tracker (DualBoundTracker): The dual bound tracking and update engine.
            lag_params (LagrangianParams): Hyperparameters for the matheuristic.
        """
        self.state = state
        self.tracker = tracker
        self.params = lag_params
        self._lock = Lock()
        self._outer_iter = 0

    # ------------------------------------------------------------------
    # Knapsack / routing selection write-backs
    # ------------------------------------------------------------------

    def set_knapsack_selection(self, x_K: np.ndarray) -> None:
        """Called after the selection subproblem solves.

        Args:
        -----
        x_K : np.ndarray
            The solution to the knapsack subproblem.

        Returns:
        --------
        None
        """
        assert x_K.shape == self.state.x_K.shape
        with self._lock:
            self.state.x_K = x_K.astype(float, copy=True)

    def set_routing_selection(self, period: int, x_R_column: np.ndarray) -> None:
        """Called by an RS worker after finishing period `period`.

        Args:
        -----
        period : int
            The current period.
        x_R_column : np.ndarray
            The solution to the routing subproblem for the current period.

        Returns:
        --------
        None
        """
        assert x_R_column.shape == (self.state.n_bins,)
        with self._lock:
            self.state.x_R[:, period] = x_R_column.astype(float, copy=True)

    # ------------------------------------------------------------------
    # Subgradient submission -> multiplier update (async, per-period)
    # ------------------------------------------------------------------

    def submit_period_result(
        self,
        period: int,
        lagrangian_value_contrib: float,
        tour_quality_ratio: float,
        upper_bound: float,
    ) -> Tuple[bool, float]:
        """
        Invoked by an RS worker after it writes x_R[:, period].

        Computes the local subgradient g[:, t] = x_K[:, t] - x_R[:, t],
        forwards it to the tracker, and (for the EMA tracker) performs an
        in-place async update of lambdas[:, t].  For the bundle tracker, the
        per-period call only accumulates the bundle; the multiplier vector is
        updated in `commit_outer_iteration`.

        Args:
        -----
        period : int
            The current period.
        lagrangian_value_contrib : float
            The contribution of the current period to the Lagrangian value.
        tour_quality_ratio : float
            The ratio of the tour quality.
        upper_bound : float
            The upper bound.

        Returns:
        --------
        accepted : bool
            Whether the current period was accepted.
        effective_step : float
            The effective step size for the current period.
        """
        with self._lock:
            subgrad_t = self.state.x_K[:, period] - self.state.x_R[:, period]
            lambdas_now = self.state.lambdas.copy()

        accepted = self.tracker.submit(
            period=period,
            lambdas=lambdas_now,
            subgrad=subgrad_t,
            lagrangian_value_contrib=lagrangian_value_contrib,
            tour_quality_ratio=tour_quality_ratio,
        )
        if not accepted:
            return False, 0.0

        # Bundle tracker: defer multiplier movement to commit_outer_iteration.
        if isinstance(self.tracker, ProximalBundleDualBoundTracker):
            return True, 0.0

        # EMA tracker: async per-period step using the period's subgradient
        # as the aggregate (conservative; the "proper" async version uses a
        # block-coordinate Polyak step).
        if isinstance(self.tracker, EMADualBoundTracker):
            full_sub = np.zeros_like(self.state.lambdas)
            full_sub[:, period] = subgrad_t

            new_lambdas, step = self.tracker.polyak_step(
                current_lambdas=self.state.lambdas,
                aggregate_subgrad=full_sub,
                upper_bound=upper_bound,
                mu=self._clamped_mu(),
            )
            with self._lock:
                self.state.lambdas[:, period] = np.clip(
                    new_lambdas[:, period],
                    self.params.lambda_min,
                    self.params.lambda_max,
                )
            return True, float(step)

        # Unknown tracker type -- generic suggest_step.
        full_sub = np.zeros_like(self.state.lambdas)
        full_sub[:, period] = subgrad_t
        new_lambdas, step = self.tracker.suggest_step(self.state.lambdas, full_sub)
        with self._lock:
            self.state.lambdas[:, period] = np.clip(
                new_lambdas[:, period],
                self.params.lambda_min,
                self.params.lambda_max,
            )
        return True, float(step)

    def commit_outer_iteration(self, full_lagrangian_value: float) -> Dict[str, float]:
        """
        Called by the top-level coordinator at the end of an outer iteration.

        For the bundle tracker: triggers the QP-based multiplier update and
        a serious/null-step decision.
        For the EMA tracker: a no-op (per-period async already moved lambdas),
        but we still use this moment to anneal gamma.

        Args:
        -----
        full_lagrangian_value : float
            The objective value (estimated) for the current assignment.

        Returns:
        --------
        Dict[str, float]: Statistics for the current iteration.
        """
        stats: Dict[str, float] = {}
        with self._lock:
            self._outer_iter += 1

        if isinstance(self.tracker, ProximalBundleDualBoundTracker):
            with self._lock:
                lambdas_now = self.state.lambdas.copy()
                # Use the bundle's own QP-suggested step.
                # suggest_step ignores aggregate_subgrad for bundle, it just uses
                # the internal bundle.
                new_lambdas, step = self.tracker.suggest_step(lambdas_now, np.zeros_like(lambdas_now))
                new_lambdas = np.clip(new_lambdas, self.params.lambda_min, self.params.lambda_max)
                self.state.lambdas = new_lambdas
            serious = self.tracker.commit_outer_iteration(
                lambdas=self.state.lambdas, full_lagrangian_value=full_lagrangian_value
            )
            stats["bundle_step_norm"] = float(step)
            stats["bundle_serious_step"] = float(serious)

        # Anneal gamma (insertion-cost trust) across all periods.
        with self._lock:
            self.state.gamma *= self.params.gamma_decay
        stats["gamma_mean"] = float(np.mean(self.state.gamma))
        return stats

    def reset_outer_iteration(self) -> None:
        """Optional hook at the start of a new outer iteration."""
        # Re-synchronise x_R to zero so stale per-period results don't leak
        # into the next iter's subgradients.
        with self._lock:
            self.state.x_R.fill(0.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clamped_mu(self, mu: Optional[float] = None) -> float:
        """Return a clamped (but not scaled by outer_iter) mu.

        This is the internal “canonicalized” mu used when calling the
        tracker (and also for gamma decay), used instead of passing
        self.params.polyak_mu_default directly into e.g. EMADualBoundTracker.polyak_step.

        Args:
        -----
        mu : Optional[float], optional
            If provided, use this mu; otherwise self.params.polyak_mu_default.

        Returns:
        --------
        float: Clamped mu.
        """
        mu = mu if mu is not None else self.params.polyak_mu_default
        return float(np.clip(mu, self.params.polyak_mu_floor, self.params.polyak_mu_ceil))

    def current_dual_bound(self) -> float:
        """Returns the current best dual bound from the tracker.

        Returns:
            float: Best dual bound found.
        """
        return self.tracker.current_dual_bound()

    def lambdas_snapshot(self) -> np.ndarray:
        """Returns a thread-safe copy of the current multipliers.

        Returns:
            np.ndarray: Copy of the lambda matrix.
        """
        with self._lock:
            return self.state.lambdas.copy()

    def gamma_snapshot(self) -> np.ndarray:
        """Returns a thread-safe copy of the current gamma trust weights.

        Returns:
            np.ndarray: Copy of the gamma vector.
        """
        with self._lock:
            return self.state.gamma.copy()
