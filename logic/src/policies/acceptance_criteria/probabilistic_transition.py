"""Probabilistic Transition Acceptance Criterion.

Stochastic criterion inspired by ACO transition rules, preferring candidates
with higher relative fitness.

Attributes:
    ProbabilisticTransitionAcceptance: The Probabilistic Transition criterion.

Example:
    >>> from logic.src.policies.acceptance_criteria.probabilistic_transition import ProbabilisticTransitionAcceptance
    >>> criterion = ProbabilisticTransitionAcceptance(alpha=1.0)
    >>> accepted, metrics = criterion.accept(current_obj=100.0, candidate_obj=98.0)
    False, {'accepted': False, 'delta': -2.0, 'alpha': 1.0}
"""

import math
import random
from typing import Any, Dict, Optional, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("pt")
class ProbabilisticTransitionAcceptance(IAcceptanceCriterion):
    """Probabilistic Transition Acceptance Criterion.

    Inspired by Ant Colony Optimization (ACO) transition rules. It evaluates the
    attractiveness of the candidate relative to the current state using an
    exponentiated formulation.

    P(accept) = (cand_obj^alpha) / (cur_obj^alpha + cand_obj^alpha)

    Attributes:
        alpha (float): Scaling factor for greediness.
        rng (random.Random): Random number generator.
    """

    def __init__(self, alpha: float = 1.0, seed: Optional[int] = None) -> None:
        """Initialize the Probabilistic Transition criterion.

        Args:
            alpha: Scaling factor governing how aggressively higher
                objectives are preferred. Defaults to 1.0.
            seed: Random seed. Defaults to None.
        """
        self.alpha = alpha
        self.rng = random.Random(seed)

    def setup(self, initial_objective: ObjectiveValue) -> None:
        """Initialize the criterion state.

        Args:
            initial_objective: The initial solution's objective.
        """
        initial_objective = cast(float, initial_objective)
        pass

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        """Determine acceptance based on relative weighted fitness.

        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            kwargs: Additional context.

        Returns:
            Tuple[bool, AcceptanceMetrics]: A tuple containing:
                - accepted (bool): True if candidate is selected via transition rule.
                - metrics (AcceptanceMetrics): Performance metadata.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        cur_fit = max(0.0, current_obj)
        cand_fit = max(0.0, candidate_obj)

        if cur_fit == 0.0 and cand_fit == 0.0:
            _accepted = bool(self.rng.random() < 0.5)
            return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj, "alpha": self.alpha}

        cur_weight = math.pow(cur_fit, self.alpha) if cur_fit > 0 else 0.0
        cand_weight = math.pow(cand_fit, self.alpha) if cand_fit > 0 else 0.0

        prob_accept = cand_weight / (cur_weight + cand_weight)
        _accepted = bool(self.rng.random() < prob_accept)
        return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj, "alpha": self.alpha}

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        """No-op update step.

        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            accepted: Whether the move was accepted.
            kwargs: Additional context.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        pass

    def get_state(self) -> Dict[str, Any]:
        """Return the current alpha scaling factor.

        Returns:
            Dict[str, Any]: State dictionary.
        """
        return {"alpha": self.alpha}
