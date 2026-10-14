"""
LinUCB contextual bandit for algorithmic-choice selection.

Action arms: the Cartesian product of `engines x cut_strategies`.  Each arm
maintains its own ridge-regression model over the context vector.  At each
outer iteration of the matheuristic, the coordinator:

    1. Builds a context vector phi(state) in R^d.
    2. Queries `select_arm(phi)` -> (engine, cut_strategy).
    3. Runs the selected configuration for one outer iter.
    4. Observes reward = (primal improvement) / (wallclock spent), clipped.
    5. Calls `update(phi, arm_id, reward)`.

The LinUCB formulation (Li et al. 2010):

    For arm a with data matrix D_a (n_a x d) and rewards c_a (n_a,):
        A_a  = D_a^T D_a + lambda I          (d x d)
        b_a  = D_a^T c_a                     (d,)
        theta_a = A_a^-1 b_a
        p_a(phi) = theta_a^T phi + alpha * sqrt(phi^T A_a^-1 phi)

    select arm = argmax_a p_a(phi).

We maintain A_a incrementally via Sherman-Morrison on updates.

Design notes
------------
* Features are normalised to [0, 1] where possible (see `build_context`).
* `alpha` (exploration width) is fixed; if needed we can anneal it over
  outer iterations.
* On the first call, every arm has A_a = lambda I and theta_a = 0; all
  arms tie at sqrt(phi^T phi / lambda) * alpha, so the tiebreaker (first
  arm) is used.  We shuffle the arm order once at init to avoid systematic
  bias.

Attributes:
    BanditArm: Per-arm LinUCB state.
    LinUCBBandit: Contextual bandit with LinUCB arm selection.
    build_context: Build the context vector.
    select_arm: Select the best arm based on the UCB score.
    update: Update the bandit with the reward for the given arm and context.

Examples:
    >>> from .params import BanditParams
    >>> bandit = LinUCBBandit(BanditParams())
    >>> arm_idx, engine, cut_strategy, score = bandit.select_arm(np.zeros(bandit.params.feature_dim))
    >>> bandit.update(arm_idx, np.zeros(bandit.params.feature_dim), 1.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .params import BanditParams


@dataclass
class BanditArm:
    """Per-arm LinUCB state.

    Attributes:
        engine (str): Engine name.
        cut_strategy (str): Cut strategy name.
        A (np.ndarray): (d, d), inverse maintained separately
        A_inv (np.ndarray): (d, d), inverse maintained separately
        b (np.ndarray): (d,)
        n_pulls (int): Number of pulls.
        cumulative_reward (float): Cumulative reward.
    """

    engine: str
    cut_strategy: str
    A: np.ndarray  # (d, d), inverse maintained separately
    A_inv: np.ndarray  # (d, d)
    b: np.ndarray  # (d,)
    n_pulls: int = 0
    cumulative_reward: float = 0.0

    @property
    def theta(self) -> np.ndarray:
        """Returns the regression coefficients (weights) for this arm.

        Returns:
            np.ndarray: Computed theta vector (A_inv @ b).
        """
        return self.A_inv @ self.b


class LinUCBBandit:
    """Contextual bandit with LinUCB arm selection.

    Attributes:
        _arms (List[BanditArm]): List of bandit arms.
        _arm_index (Dict[Tuple[str, str], int]): Mapping of (engine, cut_strategy) to arm index.
        params (BanditParams): Configuration for features, lambda, and exploration.
        rng (Optional[np.random.Generator]): Random generator for arm shuffling.
    """

    def __init__(self, params: BanditParams, rng: Optional[np.random.Generator] = None):
        """Initialize the LinUCB bandit coordinator.

        Args:
            params (BanditParams): Configuration for features, lambda, and exploration.
            rng (Optional[np.random.Generator]): Random generator for arm shuffling.
        """
        self.params = params
        self.rng = rng or np.random.default_rng()
        self._arms: List[BanditArm] = []
        self._build_arms()
        # Shuffle arm order once to avoid systematic tiebreak bias.
        self.rng.shuffle(self._arms)
        self._arm_index: Dict[Tuple[str, str], int] = {
            (a.engine, a.cut_strategy): idx for idx, a in enumerate(self._arms)
        }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_arms(self) -> None:
        """Build the bandit arms.

        Iterates over all combinations of engines and cut strategies, creating a BanditArm for each.

        Returns:
            None
        """
        d = self.params.feature_dim
        lam = self.params.ridge_lambda
        for engine in self.params.engines:
            for cut in self.params.cut_strategies:
                A = lam * np.eye(d)
                A_inv = (1.0 / lam) * np.eye(d)
                b = np.zeros(d)
                self._arms.append(BanditArm(engine=engine, cut_strategy=cut, A=A, A_inv=A_inv, b=b))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def arms(self) -> List[BanditArm]:
        """Returns a list of all bandit arms (read-only copy).

        Returns:
            List[BanditArm]: All arms managed by the bandit.
        """
        return list(self._arms)

    def select_arm(self, context: np.ndarray) -> Tuple[int, str, str, float]:
        """
        Select the best arm based on the UCB score.

        Args:
            context (np.ndarray): Context vector.

        Returns:
            Tuple[int, str, str, float]: Tuple of (arm_index, engine, cut_strategy, ucb_score).
        """
        assert context.shape == (self.params.feature_dim,), (
            f"Context dimension {context.shape} does not match feature_dim={self.params.feature_dim}"
        )
        best_idx = -1
        best_score = -np.inf
        for idx, arm in enumerate(self._arms):
            mean = float(arm.theta @ context)
            # sqrt(x^T A^-1 x) = sqrt(x @ A_inv @ x)
            quad = float(context @ arm.A_inv @ context)
            # Numerical guard
            quad = max(quad, 0.0)
            ucb = mean + self.params.alpha * np.sqrt(quad)
            if ucb > best_score:
                best_score = ucb
                best_idx = idx
        arm = self._arms[best_idx]
        return best_idx, arm.engine, arm.cut_strategy, best_score

    def update(self, arm_index: int, context: np.ndarray, reward: float) -> None:
        """
        Update the bandit with the reward for the given arm and context.

        Args:
            arm_index (int): Index of the arm to update.
            context (np.ndarray): Context vector.
            reward (float): Reward for the given arm and context.
        """
        arm = self._arms[arm_index]
        x = context
        scale = self.params.reward_scale
        clip = self.params.reward_clip
        scaled_r = float(np.clip(reward * scale, -clip, clip))

        arm.A = arm.A + np.outer(x, x)
        # Sherman-Morrison: (A + x x^T)^-1 = A^-1 - (A^-1 x x^T A^-1) / (1 + x^T A^-1 x)
        Ainv_x = arm.A_inv @ x
        denom = 1.0 + float(x @ Ainv_x)
        if denom > 1e-12:
            arm.A_inv = arm.A_inv - np.outer(Ainv_x, Ainv_x) / denom
        else:
            # Numerical fallback: recompute from A.
            arm.A_inv = np.linalg.pinv(arm.A)
        arm.b = arm.b + scaled_r * x
        arm.n_pulls += 1
        arm.cumulative_reward += scaled_r

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> List[Dict[str, Union[float, str]]]:
        """Generates a diagnostic summary of arm performance.

        Returns:
            List[Dict[str, Union[float, str]]]: List of statistics per arm.
        """
        return [
            {
                "engine": a.engine,
                "cut_strategy": a.cut_strategy,
                "n_pulls": a.n_pulls,
                "mean_reward": (a.cumulative_reward / a.n_pulls) if a.n_pulls else 0.0,
            }
            for a in self._arms
        ]


# ---------------------------------------------------------------------------
# Context vector construction
# ---------------------------------------------------------------------------


def build_context(
    *,
    outer_iter: int,
    max_outer: int,
    primal_gap_frac: float,
    dual_progress_frac: float,
    iters_since_improvement: int,
    stagnation_patience: int,
    fraction_bins_selected: float,
    fraction_periods_saturated: float,
    lambda_norm: float,
    lambda_norm_scale: float,
    feature_dim: int,
) -> np.ndarray:
    """
    Build a fixed-length context vector from the current coordinator state.

    Rough feature semantics (all normalised to [0, 1] where possible):
        0: outer_iter / max_outer              -- "how far along are we"
        1: primal_gap_frac                     -- |best - dual| / |best|
        2: dual_progress_frac                  -- recent dual bound improvement
        3: iters_since_improvement / patience  -- stagnation signal
        4: fraction_bins_selected              -- sparsity of the selection
        5: fraction_periods_saturated          -- overflow pressure in the instance
        6: lambda_norm / scale                 -- multiplier magnitude (clipped 0..1)
        7: bias (always 1.0)

    If `feature_dim` is larger than 8 we pad with zeros; smaller, we truncate.

    Args:
        outer_iter (int): Current outer iteration number.
        max_outer (int): Maximum number of outer iterations.
        primal_gap_frac (float): Fractional primal gap.
        dual_progress_frac (float): Fractional dual progress.
        iters_since_improvement (int): Number of iterations since improvement.
        stagnation_patience (int): Patience for stagnation detection.
        fraction_bins_selected (float): Fraction of bins selected.
        fraction_periods_saturated (float): Fraction of periods saturated.
        lambda_norm (float): L2 norm of Lagrangian multipliers.
        lambda_norm_scale (float): Scale factor for L2 norm of Lagrangian multipliers.
        feature_dim (int): Dimension of the feature vector.

    Returns:
        np.ndarray: Feature vector.
    """
    raw = np.array(
        [
            np.clip(outer_iter / max(1, max_outer), 0.0, 1.0),
            np.clip(primal_gap_frac, 0.0, 1.0),
            np.clip(dual_progress_frac, 0.0, 1.0),
            np.clip(iters_since_improvement / max(1, stagnation_patience), 0.0, 1.0),
            np.clip(fraction_bins_selected, 0.0, 1.0),
            np.clip(fraction_periods_saturated, 0.0, 1.0),
            np.clip(lambda_norm / max(1e-9, lambda_norm_scale), 0.0, 1.0),
            1.0,
        ],
        dtype=float,
    )
    if feature_dim == len(raw):
        return raw
    out = np.zeros(feature_dim, dtype=float)
    k = min(feature_dim, len(raw))
    out[:k] = raw[:k]
    return out
