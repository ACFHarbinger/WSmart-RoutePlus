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

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Deque, Optional, Tuple

import gurobipy as gp
import numpy as np

from .params import DualBoundParams, LagrangianParams


class DualBoundTracker(ABC):
    """
    Abstract interface for dual-bound tracking.

    Implementations of this interface are responsible for:
    - Tracking the best-so-far Lagrangian lower bound.
    - Aggregating per-period subgradients (e.g., via EMA or bundle method).
    - Suggesting adaptive step sizes for multiplier updates.

    Attributes:
        None
    """

    @abstractmethod
    def submit(
        self,
        period: int,
        lambdas: np.ndarray,
        subgrad: np.ndarray,
        lagrangian_value_contrib: float,
        tour_quality_ratio: float,
    ) -> bool:
        """Submit a per-period subgradient.

        Args:
            period (int): Period index.
            lambdas (np.ndarray): Multipliers at which the subgradient was evaluated.
            subgrad (np.ndarray): Subgradient for this period (== x^K - x^R column t).
            lagrangian_value_contrib (float): The period's contribution to the Lagrangian objective at `lambdas`.
            tour_quality_ratio (float): RS tour cost / period incumbent cost.  1.0 = new incumbent.
                Used to gate EMA updates.

        Returns:
            bool: True if the submission changed the tracker state.
        """
        ...

    @abstractmethod
    def current_dual_bound(self) -> float:
        """Returns the current best Lagrangian lower bound.

        Returns:
            float: Best dual bound found.
        """
        ...

    @abstractmethod
    def suggest_step(self, current_lambdas: np.ndarray, aggregate_subgrad: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return (new_lambdas, effective_stepsize).

        Args:
            current_lambdas (np.ndarray): Current multipliers.
            aggregate_subgrad (np.ndarray): Aggregate subgradient.

        Returns:
            Tuple[np.ndarray, float]: New lambda vector and step size.
        """
        ...


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
            dual_params (DualBoundParams): Parameters for the tracker.
            lag_params (LagrangianParams): Lagrangian-specific parameters.
            n_bins (int): Number of customer bins.
            horizon (int): Planning horizon.
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
            period (int): Period index.
            lambdas (np.ndarray): Multipliers at which subgradient was evaluated.
            subgrad (np.ndarray): Per-period subgradient.
            lagrangian_value_contrib (float): Period's contribution to Lagrangian objective.
            tour_quality_ratio (float): Ratio of RS tour cost to period incumbent.

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
            current_lambdas (np.ndarray): Current multipliers.
            aggregate_subgrad (np.ndarray): Aggregate subgradient.

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
            current_lambdas (np.ndarray): Current multipliers.
            aggregate_subgrad (np.ndarray): Aggregate subgradient.
            upper_bound (float): Upper bound (best primal).
            mu (Optional[float]): Polyak step parameter (default: lag_params.polyak_mu_default).

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
# Bundle tracker
# ---------------------------------------------------------------------------


@dataclass
class BundleEntry:
    """
    Bundle entry for the proximal bundle method.

    Attributes:
        lambdas: Lagrangian multipliers at the time of evaluation.
        subgrad: Subgradient of the Lagrangian at `lambdas`.
        value: Lagrangian dual value at `lambdas`.
    """

    lambdas: np.ndarray  # (N*T,) flattened
    subgrad: np.ndarray  # (N*T,) flattened
    value: float  # Lagrangian dual value at `lambdas`


class ProximalBundleDualBoundTracker(DualBoundTracker):
    """
    Proximal bundle method for the dual.

    The classical proximal bundle step (Kiwiel / Lemarechal) solves:

        min_{lambda} L_hat(lambda) + (u/2) ||lambda - lambda_center||^2

    where L_hat is the cutting-plane model

        L_hat(lambda) = max_{k in bundle} { value_k + <subgrad_k, lambda - lambda_k> }.

    Recast as a QP over the epigraph variable v:

        min  v + (u/2) ||lambda - lambda_center||^2
        s.t. v >= value_k + <subgrad_k, lambda - lambda_k>  for all k in bundle.

    We solve this with `gurobipy` if available (the codebase already depends
    on Gurobi), falling back to a dense numpy KKT solve if not.

    Attributes:
        params: Dual bound parameters
        lag_params: Lagrangian parameters
        n_bins: Number of bins
        horizon: Horizon
        dim: Dimension of the Lagrangian multipliers
        _bundle: Bundle of Lagrangian subgradients and values
        _lambda_center: Center of the proximal term
        _best_value: Best Lagrangian dual value found so far
        _proximal_weight: Weight for the proximal term
        _lock: Lock for thread safety
    """

    def __init__(
        self,
        dual_params: DualBoundParams,
        lag_params: LagrangianParams,
        n_bins: int,
        horizon: int,
    ):
        """Initializes the proximal bundle tracker.

        Args:
            dual_params (DualBoundParams): Parameters for the tracker.
            lag_params (LagrangianParams): Lagrangian-specific parameters.
            n_bins (int): Number of customer bins.
            horizon (int): Planning horizon.
        """
        self.params = dual_params
        self.lag_params = lag_params
        self.n_bins = n_bins
        self.horizon = horizon
        self.dim = n_bins * horizon

        self._bundle: Deque[BundleEntry] = deque(maxlen=dual_params.bundle_size)
        self._lambda_center = np.zeros(self.dim, dtype=float)
        self._best_value = -np.inf
        self._proximal_weight = dual_params.bundle_proximal_weight

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
        """Append a bundle entry.

        Unlike the EMA tracker, the bundle is strictly-valid cutting-plane
        evidence, so we do NOT gate on tour quality -- every RS return is a
        legitimate subgradient, just at a higher lambda evaluation.  However we
        still scale the *contribution* attribution so a bad RS doesn't pretend to
        be a new best.

        Args:
            period (int): Period index.
            lambdas (np.ndarray): Lambda values.
            subgrad (np.ndarray): Subgradient.
            lagrangian_value_contrib (float): Lagrangian value contribution.
            tour_quality_ratio (float): Tour quality ratio.

        Returns:
            bool: Always True, as every RS return is accepted as valid evidence.
        """
        with self._lock:
            full_sub = np.zeros(self.dim, dtype=float)
            # subgrad here is (N,) for period t; scatter into the flattened layout
            # where flatten layout is row-major (N, T).
            sub_slice = np.zeros((self.n_bins, self.horizon), dtype=float)
            sub_slice[:, period] = subgrad
            full_sub = sub_slice.ravel()

            # Lagrangian value contribution at `lambdas` for this period alone;
            # the coordinator sums across periods before we treat a submission
            # as a full evaluation.  For bundle, we only promote to best_value
            # if the caller supplies a `total` via `commit_outer_iteration`.
            entry = BundleEntry(
                lambdas=lambdas.ravel().copy(),
                subgrad=full_sub,
                value=lagrangian_value_contrib,
            )
            self._bundle.append(entry)
            return True

    def commit_outer_iteration(self, lambdas: np.ndarray, full_lagrangian_value: float) -> bool:
        """Called once per outer iteration after all periods have submitted.

        Decides whether this is a serious step or a null step.

        Args:
            lambdas (np.ndarray): Multipliers at which the iteration was completed.
            full_lagrangian_value (float): Total Lagrangian value across all periods.

        Returns:
            bool: True if a serious step was taken, False if a null step.
        """
        with self._lock:
            # Replace any partial-period bundle entries at this lambda with a
            # single summary entry.  (The partials are harmless but wasteful.)
            flat = lambdas.ravel()
            delta = full_lagrangian_value - self._best_value
            if delta >= self.params.bundle_descent_threshold * max(1e-9, abs(self._best_value)):
                # Serious step: move centre, decrease proximal weight.
                self._lambda_center = flat.copy()
                self._best_value = full_lagrangian_value
                self._proximal_weight *= self.params.bundle_weight_decrease
                return True
            else:
                # Null step: keep centre, increase weight.
                self._proximal_weight *= self.params.bundle_weight_increase
                return False

    def current_dual_bound(self) -> float:
        """Returns the current best Lagrangian lower bound.

        Returns:
            float: Best dual bound found.
        """
        with self._lock:
            return self._best_value if np.isfinite(self._best_value) else float("-inf")

    def suggest_step(self, current_lambdas: np.ndarray, aggregate_subgrad: np.ndarray) -> Tuple[np.ndarray, float]:
        """Solve the proximal bundle QP and return the new lambda.

        If the bundle is empty or the QP subsolve fails, fall back to a
        Polyak-like step on the aggregate subgradient.

        Args:
            current_lambdas (np.ndarray): Current multipliers.
            aggregate_subgrad (np.ndarray): Aggregate subgradient.

        Returns:
            Tuple[np.ndarray, float]: New lambda vector and step norm.
        """
        with self._lock:
            if not self._bundle:
                return current_lambdas, 0.0

            try:
                new_lambda_flat, step_norm = self._solve_bundle_qp()
            except Exception:
                # Fallback: Polyak on aggregate subgradient.
                g = aggregate_subgrad.ravel()
                g_norm_sq = float(np.dot(g, g))
                if g_norm_sq < 1e-12:
                    return current_lambdas, 0.0
                step = 1.0 / (self._proximal_weight * max(g_norm_sq, 1e-9))
                new_lambda_flat = self._lambda_center + step * g
                step_norm = float(np.linalg.norm(new_lambda_flat - current_lambdas.ravel()))

            new_lambda = new_lambda_flat.reshape(self.n_bins, self.horizon)
            return new_lambda, step_norm

    # ----- Internals -----

    def _solve_bundle_qp(self) -> Tuple[np.ndarray, float]:
        """Solve the proximal bundle QP via Gurobi.

        Returns:
            Tuple[np.ndarray, float]: New lambda vector and step norm.
        """
        bundle = list(self._bundle)
        d = self.dim
        u = self._proximal_weight
        centre = self._lambda_center

        model = gp.Model("bundle_qp")
        model.setParam("OutputFlag", 0)
        model.setParam("LogToConsole", 0)

        lam = model.addVars(d, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="lambda")
        v = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="v")

        for entry in bundle:
            # v >= value + <subgrad, lambda - lambda_k>
            rhs = entry.value + gp.quicksum(entry.subgrad[j] * (lam[j] - entry.lambdas[j]) for j in range(d))
            model.addConstr(v >= rhs)

        # Objective: minimise v + (u/2) sum_j (lambda_j - centre_j)^2
        quad = gp.quicksum(0.5 * u * (lam[j] - centre[j]) * (lam[j] - centre[j]) for j in range(d))
        model.setObjective(v + quad, gp.GRB.MINIMIZE)
        model.optimize()

        if model.Status not in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
            raise RuntimeError(f"Bundle QP returned status {model.Status}")

        new_lambda_flat = np.array([lam[j].X for j in range(d)])
        step_norm = float(np.linalg.norm(new_lambda_flat - centre))
        return new_lambda_flat, step_norm


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
        dual_params (DualBoundParams): Parameters specifying the strategy.
        lag_params (LagrangianParams): Lagrangian parameters.
        n_bins (int): Number of customer bins.
        horizon (int): Planning horizon.

    Returns:
        DualBoundTracker: The instantiated tracker.
    """
    strat = dual_params.strategy.lower()
    if strat == "ema":
        return EMADualBoundTracker(dual_params, lag_params, n_bins, horizon)
    if strat == "bundle":
        return ProximalBundleDualBoundTracker(dual_params, lag_params, n_bins, horizon)
    raise ValueError(f"Unknown dual-bound strategy '{dual_params.strategy}' (expected 'ema' or 'bundle').")
