"""LinUCB arms, sliding-window reward stats, and the context-vector and action-grid helpers used by RLController."""

from __future__ import annotations

import time
from typing import (
    Dict,
    List,
    Tuple,
)

import numpy as np

# ---------------------------------------------------------------------------
# Context vector helpers
# ---------------------------------------------------------------------------

_STATE_DEFAULTS: Dict[str, float] = {
    "n_nodes": 0.25,
    "fill_mean": 0.5,
    "fill_std": 0.2,
    "mandatory_ratio": 0.1,
    "lp_gap": 1.0,
    "pool_size": 0.0,
    "time_remaining": 1.0,
    "alpha": 0.5,
    "iter_count": 0.0,
}


def _build_context_vector(
    features: List[str],
    values: Dict[str, float],
) -> np.ndarray:
    """Build a unit-norm feature vector from the named feature list.

    Args:
        features: Ordered list of feature names.
        values:   Dict of feature name → raw value.

    Returns:
        Normalised context vector φ ∈ ℝ^d.
    """
    vec = np.array(
        [values.get(f, _STATE_DEFAULTS.get(f, 0.0)) for f in features],
        dtype=np.float64,
    )
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-9 else vec


# ---------------------------------------------------------------------------
# Action discretisation
# ---------------------------------------------------------------------------

# Budget split levels — each stage fraction from this grid
_BUDGET_GRID = [0.05, 0.10, 0.20, 0.35, 0.50]
# Operator weight multipliers
_OPERATOR_GRID = [0.5, 0.8, 1.0, 1.5, 2.0]


def _action_to_budget_fracs(action: np.ndarray) -> Dict[str, float]:
    """Convert a raw 4-dim action vector to normalised budget fractions.

    Args:
        action: Values in [0, 1], one per stage (lbbd, alns, bpc, sp).

    Returns:
        Dict of stage → fraction, summing to 1.0.
    """
    keys = ["lbbd", "alns", "bpc", "sp"]
    raw = np.clip(action[:4], 0.05, 0.6)
    fracs = raw / raw.sum()
    return dict(zip(keys, fracs.tolist(), strict=True))


def _action_to_operator_multipliers(action: np.ndarray) -> np.ndarray:
    """Convert a raw 3-dim action vector to operator weight multipliers.

    Args:
        action: Values in [0, 1], one per operator type.

    Returns:
        Array of operator multipliers.
    """
    return np.clip(action[:3], 0.5, 2.0)


# ---------------------------------------------------------------------------
# LinUCB bandit (diagonal covariance)
# ---------------------------------------------------------------------------


class _LinUCBArm:
    """One arm of a diagonal LinUCB bandit.

    Each arm corresponds to one level of one action dimension.
    The diagonal approximation makes updates O(d) instead of O(d²).

    Attributes:
        A_diag: Diagonal of the A = X^T X matrix.
        b:      b = X^T r vector.
        _lambda: Regularisation term.
    """

    def __init__(self, d: int, lam: float = 1.0) -> None:
        """Initialize LinUCB arm.

        Args:
            d: Dimension.
            lam: Regularisation term.

        """
        self.A_diag = np.ones(d) * lam
        self.b = np.zeros(d)
        self._lam = lam
        self._count = 0

    def theta(self) -> np.ndarray:
        """Compute parameter estimate θ = A⁻¹ b.

        Returns:
            Parameter estimate.
        """
        return self.b / self.A_diag

    def ucb(self, x: np.ndarray, alpha: float) -> float:
        """Compute upper confidence bound.

        Args:
            x: Feature vector.
            alpha: Confidence parameter.

        Returns:
            Upper confidence bound.
        """
        th = self.theta()
        var = x / self.A_diag  # diagonal A^{-1} x
        return float(th @ x) + alpha * float(np.sqrt(x @ var))

    def update(self, x: np.ndarray, reward: float) -> None:
        """Update the arm with new data.

        Args:
            x: Feature vector.
            reward: Reward value.

        """
        self.A_diag += x**2
        self.b += reward * x
        self._count += 1

    def to_dict(self) -> Dict:
        """Convert to dict for persistence.

        Returns:
            Dict with arm data.
        """
        return {"A_diag": self.A_diag.tolist(), "b": self.b.tolist(), "lam": self._lam, "count": self._count}

    @classmethod
    def from_dict(cls, d: int, data: Dict) -> "_LinUCBArm":
        """Convert from dict to arm.

        Args:
            d: Dimension.
            data: Arm data.

        Returns:
            _LinUCBArm.
        """
        arm = cls(d, data["lam"])
        arm.A_diag = np.array(data["A_diag"])
        arm.b = np.array(data["b"])
        arm._count = data.get("count", 0)
        return arm


class _SlidingWindowStats:
    """Tracks per-arm sliding-window mean reward for non-stationarity detection.

    Attributes:
        _win: Window length (0 = infinite / stationary).
        _buf: Buffer to store rewards.
        _n: Number of arms.
    """

    def __init__(self, n_arms: int, window: int) -> None:
        """Initialize sliding window with rewards.

        Args:
            n_arms: Number of arms.
            window: Window length.

        """
        self._n = n_arms
        self._win = window
        self._buf: List[List[Tuple[float, float]]] = [[] for _ in range(n_arms)]

    def update(self, arm: int, reward: float) -> None:
        """Update sliding window with reward.

        Args:
            arm: Arm index.
            reward: Reward to add.

        """
        self._buf[arm].append((time.perf_counter(), reward))
        if self._win > 0:
            self._buf[arm] = self._buf[arm][-self._win :]

    def mean(self, arm: int) -> float:
        """Get mean reward for an arm.

        Args:
            arm: Arm index.

        Returns:
            Mean reward.
        """
        buf = self._buf[arm]
        if not buf:
            return 0.0
        return sum(r for _, r in buf) / len(buf)
