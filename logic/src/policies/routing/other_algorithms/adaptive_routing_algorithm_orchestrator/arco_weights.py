"""Adaptive selection and weight-update layer for the ARCO orchestrator.

Moved unchanged from :mod:`policy_arco`: the mixin holding the adaptive
sequence selection over the w_first / W weight matrices (greedy,
epsilon-greedy and softmax strategies), the EMA weight update from
per-step marginal rewards, the multiplicative weight decay, and the
sigmoid reward helper.
"""

from typing import List, Optional

import numpy as np


class ARCOWeightsMixin:
    """Adaptive sequence selection and EMA weight updates for ARCO."""

    # ------------------------------------------------------------------
    # Adaptive sequence selection
    # ------------------------------------------------------------------

    def _select_sequence(self) -> List[int]:
        """
        Build a full permutation of routing algorithm indices using adaptive weights.

        At step k:
          - k == 0: score vector = w_first.
          - k  > 0: score vector = W[prev, :].
          Mask out already-chosen routing algorithms, then apply the configured
          selection strategy.

        Returns:
            List[int]: Routing algorithm indices in the selected execution order.
        """
        assert self._w_first is not None
        assert self._W is not None

        chosen: List[int] = []
        available: List[int] = list(range(self._n))

        prev: Optional[int] = None

        while available:
            raw_scores = self._w_first.copy() if prev is None else self._W[prev, :].copy()

            # Mask already-chosen routing algorithms
            masked_scores = np.full(self._n, -np.inf)
            for a in available:
                masked_scores[a] = max(raw_scores[a], self.params.weight_floor)

            next_idx = self._apply_strategy(masked_scores, available)
            chosen.append(next_idx)
            available.remove(next_idx)
            prev = next_idx

        return chosen

    def _apply_strategy(self, scores: np.ndarray, available: List[int]) -> int:
        """
        Apply the configured selection strategy over the available candidates.

        Args:
            scores: Full score vector (−inf for already-chosen).
            available: Indices of routing algorithms still available.

        Returns:
            int: Selected routing algorithm index.
        """
        strategy = self.params.selection_strategy

        if strategy == "greedy":
            return int(np.argmax(scores))

        if strategy == "epsilon_greedy":
            if self._rng.random() < self.params.epsilon:
                return self._rng.choice(available)
            return int(np.argmax(scores))

        if strategy == "softmax":
            avail_scores = np.array([scores[i] for i in available], dtype=float)
            avail_scores -= avail_scores.max()  # Numerical stability
            weights = np.exp(avail_scores / max(self.params.temperature, 1e-12))
            total = weights.sum()
            if total < 1e-12:
                return self._rng.choice(available)
            probs = weights / total
            dart = self._rng.random()
            cumulative = 0.0
            for j, p in enumerate(probs):
                cumulative += float(p)
                if dart <= cumulative:
                    return available[j]
            return available[int(np.argmax(probs))]

        raise ValueError(f"ARCO: unknown selection strategy {strategy!r}")

    # ------------------------------------------------------------------
    # Online weight update (EMA)
    # ------------------------------------------------------------------

    def _update_weights(
        self,
        sequence: List[int],
        step_profits: List[float],
        step_costs: List[float],
    ) -> None:
        """
        Update adaptive weights based on observed per-step marginal rewards.

        The reward signal combines profit gain and cost reduction, then is
        normalised to [0, 1] via a sigmoid-like clamp before the EMA step.
        Negative rewards (worsening steps) are clamped to 0 — the ARCO
        does not punish routing algorithms that happen to worsen the interim
        objective; it only rewards those that improve it.

        Args:
            sequence: Routing algorithm indices in the order they were executed.
            step_profits: Marginal profit change Δ for each step.
            step_costs: Marginal cost improvement Δ for each step (positive = better).
        """
        assert self._w_first is not None
        assert self._W is not None
        assert self._call_counts is not None

        alpha = self.params.alpha_ema
        floor = self.params.weight_floor

        for step, c_idx in enumerate(sequence):
            self._call_counts[c_idx] += 1

            # Combined reward: use profit if available, else cost
            raw_reward = step_profits[step] if step < len(step_profits) else 0.0
            if raw_reward <= 0.0 and step < len(step_costs):
                raw_reward = step_costs[step]

            # Sigmoid normalisation: maps any positive value to (0, 1)
            reward = _sigmoid_reward(raw_reward)

            if step == 0:
                # Update first-position weight
                self._w_first[c_idx] = max(
                    floor,
                    (1.0 - alpha) * self._w_first[c_idx] + alpha * reward,
                )
            else:
                prev_idx = sequence[step - 1]
                self._W[prev_idx, c_idx] = max(
                    floor,
                    (1.0 - alpha) * self._W[prev_idx, c_idx] + alpha * reward,
                )

    def _apply_decay(self) -> None:
        """Apply multiplicative decay to all weights, floored at weight_floor."""
        assert self._w_first is not None
        assert self._W is not None

        floor = self.params.weight_floor
        d = self.params.decay

        self._w_first = np.maximum(floor, self._w_first * d)
        self._W = np.maximum(floor, self._W * d)
        np.fill_diagonal(self._W, 0.0)  # Diagonal must stay 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sigmoid_reward(raw: float, scale: float = 1.0) -> float:
    """
    Map a raw improvement value to a reward in [0, 1) via a sigmoid-like clamp.

    Non-positive values map to 0.  Large positive values approach 1.
    This keeps rewards bounded and comparable regardless of problem scale.

    f(x) = x / (x + scale)    for x > 0
    f(x) = 0                  for x ≤ 0
    Args:
        raw: Description of raw.
        scale: Description of scale.

    Returns:
        Description of return value.
    """
    if raw <= 0.0:
        return 0.0
    return raw / (raw + scale)
