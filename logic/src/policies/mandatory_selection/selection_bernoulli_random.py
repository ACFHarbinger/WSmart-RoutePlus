"""
Bernoulli Trial Random Selection Module.

Each bin is independently included in the mandatory set via a Bernoulli
trial with fixed probability p:

    X_i ~ Bernoulli(p),  i = 1, ..., n

A bin is mandated if X_i = 1. Unlike fixed-K random selection, the realised
set cardinality is random:

    |S| ~ Binomial(n, p)

with E[|S|] = n·p, providing a stochastic baseline whose variability in set
size mirrors a memoryless per-bin decision process.

Attributes:
    None

Example:
    >>> from logic.src.policies.mandatory.selection_bernoulli import BernoulliRandomSelection
    >>> strategy = BernoulliRandomSelection()
    >>> bins, ctx = strategy.select_bins(context)
"""

import random
from typing import List, Optional, Tuple

import numpy as np

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context import SearchContext, SelectionContext
from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy
from logic.src.policies.mandatory_selection.base import MandatorySelectionRegistry


@GlobalRegistry.register(
    PolicyTag.SELECTION,
    PolicyTag.HEURISTIC,
    PolicyTag.STOCHASTIC,
)
@MandatorySelectionRegistry.register("bernoulli_random")
class BernoulliRandomSelection(IMandatorySelectionStrategy):
    """Independent Bernoulli trial selection strategy.

    Each eligible bin i is independently mandated with probability p:
    X_i ~ Bernoulli(p). The ``threshold`` parameter is interpreted as the
    selection probability p and is clipped to [0, 1].

    Attributes:
        min_fill (float): Minimum fill ratio for a bin to be eligible for the
                          Bernoulli trial. Defaults to 0.0.
    """

    def __init__(self, p: Optional[float] = None, min_fill: float = 0.0) -> None:
        """
        Args:
            p (Optional[float]): Selection probability in [0, 1]; ``None`` uses ``context.threshold``.
            min_fill (float): Minimum fill ratio for a bin to be eligible for the trial.
        """
        self.p = None if p is None else float(min(max(p, 0.0), 1.0))
        self.min_fill = float(min_fill)

    def select_bins(self, context: SelectionContext) -> Tuple[List[int], SearchContext]:
        """
        Sample each eligible bin independently via Bernoulli(p).

        Bins below ``min_fill`` are excluded before sampling (they are not
        eligible for the trial, not just rejected by it), so the effective
        population for the Binomial distribution is the eligible subset.

        This mechanism ensures that bins with insufficient capacity or
        fill levels are strictly excluded from the stochastic process.

        Args:
            context (SelectionContext): The selection context.

        Returns:
            Tuple[List[int], SearchContext]: Selected bin IDs and search context.
        """
        p: float = self.p if self.p is not None else float(getattr(context, "threshold", 0.5))
        p = max(0.0, min(1.0, p))  # Clip to valid Bernoulli parameter range.

        min_fill: float = float(getattr(self, "min_fill", 0.0))

        # Seed resolution: prefer an explicit seed; fall back to current_day
        # so that each simulation day produces a different but reproducible draw.
        explicit_seed: Optional[int] = getattr(context, "seed", None)
        current_day: int = getattr(context, "current_day", 1)
        seed: Optional[int] = explicit_seed if explicit_seed is not None else current_day

        current_fill: Optional[np.ndarray] = getattr(context, "current_fill", None)
        bin_ids: Optional[np.ndarray] = getattr(context, "bin_ids", None)

        rng = random.Random(seed)

        eligible_bins: List[int] = []
        n_trials: int = 0  # Number of bins that entered the Bernoulli trial.

        if current_fill is not None and len(current_fill) > 0:
            for i, fill in enumerate(current_fill):
                if fill < min_fill:
                    continue  # Below eligibility threshold — not trialled.
                n_trials += 1
                if rng.random() < p:
                    eligible_bins.append(i + 1)  # 1-based indexing for routing
        elif bin_ids is not None and len(bin_ids) > 0:
            for raw_id in bin_ids:
                n_trials += 1
                if rng.random() < p:
                    eligible_bins.append(int(raw_id) + 1)

        metrics: dict = {
            "strategy": "BernoulliRandomSelection",
            "p": p,
            "n_trials": n_trials,
            "n_selected": len(eligible_bins),
            # Realised rate for diagnosing RNG quality in simulation logs.
            "realised_rate": len(eligible_bins) / n_trials if n_trials > 0 else 0.0,
        }
        return eligible_bins, SearchContext.initialize(selection_metrics=metrics)
