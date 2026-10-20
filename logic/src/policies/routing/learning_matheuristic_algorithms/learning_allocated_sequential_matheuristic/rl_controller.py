r"""RL Controller — online LinUCB bandit + offline PPO policy.

The controller sits *above* all solver stages and decides, for each instance,
how to allocate the per-instance time budget across stages and which solver
configuration to use.  It operates in three modes:

online (LinUCB)
    A contextual bandit (Li, Chu, Langford & Schapire 2010, WWW) learns
    across instances within a single simulation run.  Actions are budget-split
    fractions or ALNS operator-weight multipliers; reward is Δprofit/Δtime.
    Non-stationarity is handled by a sliding-window UCB (Garivier & Moulines
    2011, ALT) with an optional CUSUM change-detection trigger.

offline (pretrained policy)
    A JSON or pickle file stores a policy trained on historical instances.
    The file format is a dict with keys 'weights' (the d×K matrix A^{-1} b
    for LinUCB) or a serialised PPO network.  The controller reads this file
    and uses it without further update.

hybrid
    Offline policy initialises LinUCB; online updates continue from there.

disabled
    The controller is a no-op; alpha-derived budgets are used throughout.

State vector φ(x) ∈ ℝ^d
------------------------
Instance features (static):
    n_nodes         — number of customer nodes (normalised by 200).
    fill_mean       — mean fill level (0–1).
    fill_std        — std fill level.
    mandatory_ratio — |mandatory| / n_nodes.

Dynamic features (updated each call):
    lp_gap          — (lp_ub − best_profit) / max(1, lp_ub), clamped to [0,1].
    pool_size       — |RoutePool| / 10 000, clamped to [0,1].
    time_remaining  — remaining budget / total budget.
    alpha           — the global quality/speed dial.
    iter_count      — call index normalised by expected total calls.

Action space (rl_action_space)
------------------------------
'budgets'   — K=4 actions: (τ_lbbd, τ_alns, τ_bpc, τ_sp) as fractions.
'operators' — K=3 actions: multipliers for random/worst/shaw destroy weights.
'combined'  — K=7 actions: concatenation of both.

Each action is a softmax over a discrete grid (5 levels per dimension → 5^K
total arms for 'budgets').  The controller uses a diagonal-covariance
approximation for tractability: each action dimension is treated as an
independent 1-D contextual bandit, and the joint action is their product.

References:
    Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). A contextual-
        bandit approach to personalized news article recommendation. WWW.
    Garivier, A., & Moulines, E. (2011). On upper-confidence bound policies
        for switching bandit problems. ALT.

Attributes:
    RLController: Main controller class.

Example:
    >>> ctrl = RLController(params)
    >>> ctx  = ctrl.make_context(n_nodes=50, fill_mean=0.6, ...)
    >>> budgets = ctrl.act(ctx, time_limit=120.0)
    >>> ctrl.update(ctx, reward=0.15)

The LinUCB arms, sliding-window stats, context vector and action grids live in ``rl_bandit`` and are re-exported here.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .rl_bandit import (
    _BUDGET_GRID as _BUDGET_GRID,
)
from .rl_bandit import (
    _OPERATOR_GRID as _OPERATOR_GRID,
)
from .rl_bandit import (
    _STATE_DEFAULTS as _STATE_DEFAULTS,
)
from .rl_bandit import (
    _action_to_budget_fracs as _action_to_budget_fracs,
)
from .rl_bandit import (
    _action_to_operator_multipliers as _action_to_operator_multipliers,
)
from .rl_bandit import (
    _build_context_vector as _build_context_vector,
)
from .rl_bandit import (
    _LinUCBArm as _LinUCBArm,
)
from .rl_bandit import (
    _SlidingWindowStats as _SlidingWindowStats,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main RL controller
# ---------------------------------------------------------------------------


class RLController:
    """Online LinUCB / offline PPO controller for stage-budget allocation.

    Attributes:
        params:       LASMPipelineConfig controlling RL behaviour.
        _arms:        Dict of arm_id → _LinUCBArm for each action dimension.
        _sw:          Sliding-window reward tracker.
        _call_count:  Number of decisions made so far.
        _last_context: Most recent context vector (for update).
        _last_actions: Most recent action vector (for update).
    """

    def __init__(self, params: Any) -> None:
        """Initialise the controller from LASMPipelineConfig.

        Args:
            params: LASMPipelineConfig instance.
        """
        self._params = params
        self._features = list(params.rl_state_features)
        self._d = len(self._features)
        self._lam = params.rl_exploration
        self._window = params.rl_window
        self._min_samples = params.rl_min_samples
        self._reward_shaping = params.rl_reward_shaping
        self._action_space = params.rl_action_space
        self._mode = params.rl_mode
        self._call_count = 0
        self._policy_path = params.rl_policy_path

        # Determine number of action dimensions
        if self._action_space == "budgets":
            self._n_dims = 4  # lbbd, alns, bpc, sp fracs
            self._n_levels = len(_BUDGET_GRID)
        elif self._action_space == "operators":
            self._n_dims = 3  # random, worst, shaw multipliers
            self._n_levels = len(_OPERATOR_GRID)
        else:  # combined
            self._n_dims = 7
            self._n_levels = 5

        # Per-dimension, per-level arms
        self._arms: Dict[Tuple[int, int], _LinUCBArm] = {
            (dim, lvl): _LinUCBArm(self._d, self._lam) for dim in range(self._n_dims) for lvl in range(self._n_levels)
        }

        self._sw = _SlidingWindowStats(self._n_dims * self._n_levels, self._window)
        self._last_context: Optional[np.ndarray] = None
        self._last_actions: Optional[List[int]] = None  # chosen level per dim
        self._last_t: float = time.perf_counter()

        # Load offline policy if available
        if self._mode in ("offline", "hybrid") and self._policy_path:
            self._load_policy(self._policy_path)

    # ------------------------------------------------------------------
    # Context construction
    # ------------------------------------------------------------------

    def make_context(
        self,
        n_nodes: int,
        fill_levels: np.ndarray,
        mandatory_ratio: float,
        lp_ub: float,
        best_profit: float,
        pool_size: int,
        time_remaining: float,
        time_total: float,
    ) -> np.ndarray:
        """Build a normalised context vector from instance features.

        Args:
            n_nodes:          Customer node count.
            fill_levels:      Array of fill fractions (0–1).
            mandatory_ratio:  |mandatory| / n_nodes.
            lp_ub:            Current LP upper bound.
            best_profit:      Current best primal profit.
            pool_size:        Current RoutePool size.
            time_remaining:   Remaining wall-clock seconds.
            time_total:       Total time budget.

        Returns:
            Context vector φ ∈ ℝ^d.
        """
        gap = (lp_ub - best_profit) / max(1.0, abs(lp_ub)) if lp_ub < float("inf") else 1.0
        raw = {
            "n_nodes": n_nodes / 200.0,
            "fill_mean": float(np.mean(fill_levels)) if len(fill_levels) > 0 else 0.5,
            "fill_std": float(np.std(fill_levels)) if len(fill_levels) > 0 else 0.2,
            "mandatory_ratio": mandatory_ratio,
            "lp_gap": min(1.0, max(0.0, gap)),
            "pool_size": min(1.0, pool_size / 10_000.0),
            "time_remaining": time_remaining / max(1.0, time_total),
            "alpha": self._params.alpha,
            "iter_count": min(1.0, self._call_count / 30.0),
        }
        return _build_context_vector(self._features, raw)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(
        self,
        context: np.ndarray,
        time_limit: float,
        budget_override_from_alpha: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Select action (budget split and/or operator multipliers).

        If RL is disabled, or not yet warmed up, returns the alpha-derived
        defaults unchanged.

        Args:
            context:    Context vector from make_context().
            time_limit: Total budget for this instance.
            budget_override_from_alpha: Pre-computed alpha-derived defaults
                (used as fallback and clipping reference).

        Returns:
            Dict with keys:
                'budget_fracs'    — Dict[str, float] stage fractions.
                'operator_mults'  — np.ndarray of 3 operator multipliers.
                'action_levels'   — raw chosen levels (for update).
        """
        self._last_context = context
        self._call_count += 1

        # Always return alpha defaults when disabled or under warm-up
        if self._mode == "disabled" or self._call_count <= self._min_samples:
            return self._default_action(budget_override_from_alpha)

        # UCB arm selection — one level per action dimension
        chosen_levels: List[int] = []
        action_raw = np.zeros(self._n_dims)

        for dim in range(self._n_dims):
            ucb_scores = [self._arms[(dim, lvl)].ucb(context, self._lam) for lvl in range(self._n_levels)]
            best_lvl = int(np.argmax(ucb_scores))
            chosen_levels.append(best_lvl)
            # grid = _BUDGET_GRID if self._action_space in ("budgets", "combined") else _OPERATOR_GRID
            if dim < 4 and self._action_space in ("budgets", "combined"):
                action_raw[dim] = _BUDGET_GRID[best_lvl]
            else:
                action_raw[dim] = _OPERATOR_GRID[best_lvl % self._n_levels]

        self._last_actions = chosen_levels
        self._last_t = time.perf_counter()

        if self._action_space == "budgets":
            fracs = _action_to_budget_fracs(action_raw)
            mults = np.ones(3)
        elif self._action_space == "operators":
            fracs = budget_override_from_alpha or {}
            mults = _action_to_operator_multipliers(action_raw)
        else:  # combined
            fracs = _action_to_budget_fracs(action_raw[:4])
            mults = _action_to_operator_multipliers(action_raw[4:7])

        return {
            "budget_fracs": fracs,
            "operator_mults": mults,
            "action_levels": chosen_levels,
        }

    # ------------------------------------------------------------------
    # Reward and update
    # ------------------------------------------------------------------

    def update(
        self,
        context: np.ndarray,
        action_levels: List[int],
        delta_profit: float,
        delta_time: float,
        best_profit: float,
    ) -> None:
        """Update arm weights from observed reward.

        Args:
            context:       Context vector used during act().
            action_levels: Level indices chosen per dimension.
            delta_profit:  Profit improvement from prior best.
            delta_time:    Wall-clock seconds consumed.
            best_profit:   Current best profit (for relative shaping).
        """
        if self._mode == "disabled" or not action_levels:
            return

        raw_reward = self._shape_reward(delta_profit, delta_time, best_profit)

        for dim, lvl in enumerate(action_levels):
            arm = self._arms[(dim, lvl)]
            arm.update(context, raw_reward)
            self._sw.update(dim * self._n_levels + lvl, raw_reward)

        logger.debug(
            "[RL] update  Δprofit=%.4f  Δt=%.1fs  reward=%.4f  calls=%d",
            delta_profit,
            delta_time,
            raw_reward,
            self._call_count,
        )

    def _shape_reward(
        self,
        delta_profit: float,
        delta_time: float,
        best_profit: float,
    ) -> float:
        """Apply reward shaping.

        Args:
            delta_profit: Raw profit improvement.
            delta_time:   Time used.
            best_profit:  Reference profit.

        Returns:
            Shaped scalar reward.
        """
        if self._reward_shaping == "relative":
            return delta_profit / max(1.0, abs(best_profit))
        elif self._reward_shaping == "efficiency":
            return delta_profit / max(0.1, delta_time)
        else:  # absolute
            return delta_profit

    def _default_action(self, alpha_defaults: Optional[Dict[str, float]]) -> Dict[str, Any]:
        """Return alpha-derived defaults as the action.

        Args:
            alpha_defaults: Pre-computed alpha-derived defaults.

        Returns:
            Default action.
        """
        return {
            "budget_fracs": alpha_defaults or {},
            "operator_mults": np.ones(3),
            "action_levels": [],
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise arm weights to JSON for offline reuse.

        Args:
            path: File path ending in .json or .pkl.
        """
        state = {
            "call_count": self._call_count,
            "arms": {f"{dim}_{lvl}": arm.to_dict() for (dim, lvl), arm in self._arms.items()},
        }
        if path.endswith(".pkl"):
            with open(path, "wb") as fh:
                pickle.dump(state, fh)
        else:
            with open(path, "w") as fh:
                json.dump(state, fh)
        logger.info("[RL] Policy saved to %s", path)

    def _load_policy(self, path: str) -> None:
        """Load previously saved arm weights.

        Args:
            path: File path ending in .json or .pkl.
        """
        if not os.path.exists(path):
            logger.warning("[RL] Policy file not found: %s — starting fresh", path)
            return
        try:
            if path.endswith(".pkl"):
                with open(path, "rb") as fh:
                    state = pickle.load(fh)
            else:
                with open(path) as fh:
                    state = json.load(fh)
            self._call_count = state.get("call_count", 0)
            for key, arm_data in state.get("arms", {}).items():
                dim, lvl = (int(x) for x in key.split("_"))
                if (dim, lvl) in self._arms:
                    self._arms[(dim, lvl)] = _LinUCBArm.from_dict(self._d, arm_data)
            logger.info("[RL] Policy loaded from %s (calls=%d)", path, self._call_count)
        except Exception as exc:
            logger.warning("[RL] Failed to load policy: %s — starting fresh", exc)
