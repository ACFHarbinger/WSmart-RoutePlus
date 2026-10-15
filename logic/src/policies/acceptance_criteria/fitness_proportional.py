"""Fitness Proportional Selection (FPS) Acceptance Criterion.

Stochastic acceptance logic where the probability of acceptance is proportional
to the candidate's fitness.

Attributes:
    FitnessProportionalAcceptance: The FPS criterion.

Example:
    >>> from logic.src.policies.acceptance_criteria.fitness_proportional import FitnessProportionalAcceptance
    >>> criterion = FitnessProportionalAcceptance()
    >>> criterion.setup(initial_objective=100.0)
    >>> accepted, metrics = criterion.accept(current_obj=100.0, candidate_obj=120.0)
    True, {'accepted': True, 'delta': 20.0}
"""

import random
from typing import Any, Dict, Optional, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("fp")
class FitnessProportionalAcceptance(IAcceptanceCriterion):
    """Fitness Proportional Selection (FPS) Acceptance Criterion.

    Accepts candidates probabilistically in proportion to their fitness relative
    to the current solution (similar to Roulette-Wheel selection adapted for
    binary choice).

    Probability of accepting the candidate:
    P(accept) = max(0, cand_obj) / (max(0, cur_obj) + max(0, cand_obj))

    Attributes:
        rng (random.Random): Random number generator.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize the Fitness Proportional criterion.

        Args:
            seed: Random seed. Defaults to None.
        """
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
        """Determine acceptance based on relative fitness.

        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            kwargs: Additional context.

        Returns:
            Tuple[bool, AcceptanceMetrics]: A tuple containing:
                - accepted (bool): True if candidate is selected via roulette wheel.
                - metrics (AcceptanceMetrics): Performance metadata.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        # Prevent negative fitness values
        cur_fit = max(0.0, current_obj)
        cand_fit = max(0.0, candidate_obj)

        total_fit = cur_fit + cand_fit
        if total_fit == 0.0:
            _accepted = bool(self.rng.random() < 0.5)
            return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj}

        prob_accept = cand_fit / total_fit
        _accepted = bool(self.rng.random() < prob_accept)
        return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj}

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
        """Return an empty state dictionary.

        Returns:
            Dict[str, Any]: Empty dictionary.
        """
        return {}
