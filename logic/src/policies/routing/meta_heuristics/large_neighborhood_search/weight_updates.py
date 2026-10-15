"""Operator weight-update strategies for the shared LNS engine (WP-11 P-31).

Each strategy owns the per-iteration score accounting and applies it to the
operator weight lists in place (the lists are shared by reference with the
solver so subclass post-construction appends stay visible).

Attributes:
    RopkePisingerSegments: Segment-based update used by the classic ALNS.
    EmaWeightUpdate: Per-iteration exponential moving average used by PG-CLNS.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.large_neighborhood_search.weight_updates import (
    ...     RopkePisingerSegments,
    ... )
    >>> weights = [1.0, 1.0]
    >>> strategy = RopkePisingerSegments(weights, [1.0], reaction_factor=0.1)
"""

from typing import List


class RopkePisingerSegments:
    """Segment-based operator weight update (Ropke & Pisinger 2005, Eq. 4.1).

    Scores and usage counts accumulate during a segment; at segment end the
    weights are decayed toward the segment average score:

        w_{i,j+1} = w_{i,j} * (1 - r) + r * (π_i / θ_i)

    where r is the reaction factor, π_i the accumulated score and θ_i the
    number of uses of operator i during the segment.

    Attributes:
        destroy_weights: Live destroy-operator weight list (updated in place).
        repair_weights: Live repair-operator weight list (updated in place).
        destroy_scores: Accumulated destroy scores for the current segment.
        repair_scores: Accumulated repair scores for the current segment.
        destroy_counts: Destroy usage counts for the current segment.
        repair_counts: Repair usage counts for the current segment.
        weight_decay: Weight decay factor (1 - reaction_factor).
        weight_learning_rate: Weight learning rate (reaction_factor).
    """

    def __init__(
        self,
        destroy_weights: List[float],
        repair_weights: List[float],
        reaction_factor: float,
    ):
        """Initialize the segment strategy over live weight lists.

        Args:
            destroy_weights: Live destroy-operator weight list.
            repair_weights: Live repair-operator weight list.
            reaction_factor: Reaction factor r controlling adaptation speed.
        """
        self.destroy_weights = destroy_weights
        self.repair_weights = repair_weights
        self.destroy_scores: List[float] = [0.0] * len(destroy_weights)
        self.repair_scores: List[float] = [0.0] * len(repair_weights)
        self.destroy_counts: List[int] = [0] * len(destroy_weights)
        self.repair_counts: List[int] = [0] * len(repair_weights)
        self.weight_decay = 1.0 - reaction_factor
        self.weight_learning_rate = reaction_factor

    def update(self, d_idx: int, r_idx: int, score: float) -> None:
        """Accumulate the score and usage counts for the used operators.

        Args:
            d_idx: Index of the destroy operator used.
            r_idx: Index of the repair operator used.
            score: Score earned by this iteration's operators.
        """
        self.destroy_scores[d_idx] += score
        self.repair_scores[r_idx] += score
        self.destroy_counts[d_idx] += 1
        self.repair_counts[r_idx] += 1

    def end_segment(self) -> None:
        """Apply the segment weight update and reset the segment accumulators."""
        for i in range(len(self.destroy_weights)):
            if self.destroy_counts[i] > 0:
                avg_score = self.destroy_scores[i] / self.destroy_counts[i]
                self.destroy_weights[i] = (
                    self.weight_decay * self.destroy_weights[i] + self.weight_learning_rate * avg_score
                )
            self.destroy_scores[i] = 0.0
            self.destroy_counts[i] = 0

        for i in range(len(self.repair_weights)):
            if self.repair_counts[i] > 0:
                avg_score = self.repair_scores[i] / self.repair_counts[i]
                self.repair_weights[i] = (
                    self.weight_decay * self.repair_weights[i] + self.weight_learning_rate * avg_score
                )
            self.repair_scores[i] = 0.0
            self.repair_counts[i] = 0


class EmaWeightUpdate:
    """Per-iteration exponential moving-average weight update (PG-CLNS LNS).

    Weights are updated after every iteration (no segments):

        w_i <- λ * w_i + (1 - λ) * max(floor, score)

    with decay λ = 0.8 and floor 0.1, matching the pre-P-31 PG-CLNS LNS copy.

    Attributes:
        destroy_weights: Live destroy-operator weight list (updated in place).
        repair_weights: Live repair-operator weight list (updated in place).
        decay: Exponential decay factor λ.
        floor: Lower bound applied to the score before blending.
    """

    def __init__(
        self,
        destroy_weights: List[float],
        repair_weights: List[float],
        decay: float = 0.8,
        floor: float = 0.1,
    ):
        """Initialize the EMA strategy over live weight lists.

        Args:
            destroy_weights: Live destroy-operator weight list.
            repair_weights: Live repair-operator weight list.
            decay: Exponential decay factor (default 0.8).
            floor: Score floor (default 0.1).
        """
        self.destroy_weights = destroy_weights
        self.repair_weights = repair_weights
        self.decay = decay
        self.floor = floor

    def update(self, d_idx: int, r_idx: int, score: float) -> None:
        """Blend the used operators' weights toward the floored score.

        Args:
            d_idx: Index of the destroy operator used.
            r_idx: Index of the repair operator used.
            score: Score earned by this iteration's operators.
        """
        self.destroy_weights[d_idx] = (
            self.decay * self.destroy_weights[d_idx] + (1 - self.decay) * max(self.floor, score)
        )
        self.repair_weights[r_idx] = (
            self.decay * self.repair_weights[r_idx] + (1 - self.decay) * max(self.floor, score)
        )

    def end_segment(self) -> None:
        """No-op: the EMA update is applied per iteration, not per segment."""
        return None
