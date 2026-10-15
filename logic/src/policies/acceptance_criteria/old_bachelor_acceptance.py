"""Old Bachelor Acceptance (OBA) Criterion.

Dynamic threshold strategy that adjusts standards based on acceptance success.

Attributes:
    OldBachelorAcceptance: The OBA criterion.

Example:
    >>> from logic.src.policies.acceptance_criteria.old_bachelor_acceptance import OldBachelorAcceptance
    >>> criterion = OldBachelorAcceptance(contraction=0.1, dilation=0.1)
    >>> criterion.setup(initial_objective=100.0)
    >>> accepted, metrics = criterion.accept(current_obj=100.0, candidate_obj=98.0)
    True, {'accepted': True, 'delta': -2.0, 'threshold': 0.0, 'age': 0}
"""

from typing import Any, Dict, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("oba")
class OldBachelorAcceptance(IAcceptanceCriterion):
    """Dynamic, non-monotone threshold strategy.

    Tightens standards upon acceptance (contraction) and relaxes standards
    upon rejection streaks (dilation) to autonomously balance exploitation
    and exploration without a rigid decay schedule.

    Attributes:
        contraction (float): Threshold reduction magnitude upon acceptance.
        dilation (float): Base threshold relaxation magnitude upon rejection.
        threshold (float): Current absolute tolerance for deterioration.
        age (int): Number of consecutive rejections.
    """

    def __init__(self, contraction: float, dilation: float):
        """Initialize the Old Bachelor criterion.

        Args:
            contraction: Amount to reduce the threshold upon an accepted move.
            dilation: Base amount to increase the threshold upon a
                rejected move.
        """
        self.contraction = contraction
        self.dilation = dilation
        self.threshold = 0.0
        self.age = 0

    def setup(self, initial_objective: ObjectiveValue) -> None:
        """Initialize the threshold and age.

        Args:
            initial_objective: The initial solution's objective.
        """
        initial_objective = cast(float, initial_objective)
        self.threshold = 0.0  # Start with strict greedy behavior
        self.age = 0

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        """Determine acceptance based on the dynamic threshold.

        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            kwargs: Additional context.

        Returns:
            Tuple[bool, AcceptanceMetrics]: A tuple containing:
                - accepted (bool): True if delta >= -threshold.
                - metrics (AcceptanceMetrics): Performance metadata.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        # delta is negative for worsening moves.
        # e.g., candidate=90, current=100 -> delta=-10. If threshold is 15, -10 >= -15 (Accept).
        delta = candidate_obj - current_obj
        _accepted = bool(delta >= -self.threshold)
        return _accepted, {
            "accepted": _accepted,
            "delta": candidate_obj - current_obj,
            "threshold": self.threshold,
            "age": self.age,
        }

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        """Update the threshold and age based on acceptance result.

        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            accepted: Whether the move was accepted.
            kwargs: Additional context.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        if accepted:
            # Move accepted: Reset rejection streak and tighten standards
            self.age = 0
            self.threshold = max(0.0, self.threshold - self.contraction)
        else:
            # Move rejected: Increase age and relax standards exponentially
            self.age += 1
            self.threshold += self.dilation * self.age

    def get_state(self) -> Dict[str, Any]:
        """Return the current threshold and age.

        Returns:
            Dict[str, Any]: State dictionary.
        """
        return {"threshold": self.threshold, "age": self.age}
