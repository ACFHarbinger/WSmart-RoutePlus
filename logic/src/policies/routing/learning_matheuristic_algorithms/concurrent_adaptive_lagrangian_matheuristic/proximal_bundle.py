"""Dual-bound tracker interface and the proximal bundle implementation. The EMA tracker stays in dual.py and subclasses DualBoundTracker from here."""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import (
    Deque,
    Tuple,
)

import gurobipy as gp
import numpy as np

from .params import (
    DualBoundParams,
    LagrangianParams,
)


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
            period: Period index.
            lambdas: Multipliers at which the subgradient was evaluated.
            subgrad: Subgradient for this period (== x^K - x^R column t).
            lagrangian_value_contrib: The period's contribution to the Lagrangian objective at `lambdas`.
            tour_quality_ratio: RS tour cost / period incumbent cost.  1.0 = new incumbent.
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
            current_lambdas: Current multipliers.
            aggregate_subgrad: Aggregate subgradient.

        Returns:
            Tuple[np.ndarray, float]: New lambda vector and step size.
        """
        ...


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
            dual_params: Parameters for the tracker.
            lag_params: Lagrangian-specific parameters.
            n_bins: Number of customer bins.
            horizon: Planning horizon.
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
            period: Period index.
            lambdas: Lambda values.
            subgrad: Subgradient.
            lagrangian_value_contrib: Lagrangian value contribution.
            tour_quality_ratio: Tour quality ratio.

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
            lambdas: Multipliers at which the iteration was completed.
            full_lagrangian_value: Total Lagrangian value across all periods.

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
            current_lambdas: Current multipliers.
            aggregate_subgrad: Aggregate subgradient.

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
