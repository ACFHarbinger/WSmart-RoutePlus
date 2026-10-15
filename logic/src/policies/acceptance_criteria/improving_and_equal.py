"""Improving and Equal (IE) Criterion.

Weakly elitist strategy that accepts improvements and neutral moves.

Attributes:
    ImprovingAndEqual: The IE criterion.

Example:
    >>> from logic.src.policies.acceptance_criteria.improving_and_equal import ImprovingAndEqual
    >>> criterion = ImprovingAndEqual()
    >>> accepted, metrics = criterion.accept(current_obj=100.0, candidate_obj=98.0)
    False, {'accepted': False, 'delta': -2.0}
"""

from typing import Any, Dict, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("ie")
class ImprovingAndEqual(IAcceptanceCriterion):
    """Weakly elitist strategy.

    Accepts improving moves and identical-cost moves, allowing the solver to
    transverse neutral objective plateaus.

    Attributes:
        None
    """

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
        """Accept if the candidate objective is greater than or equal to current.

        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            kwargs: Additional context.

        Returns:
            Tuple[bool, AcceptanceMetrics]: A tuple containing:
                - accepted (bool): True if candidate_obj >= current_obj.
                - metrics (AcceptanceMetrics): Performance metadata.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        _accepted = bool(candidate_obj >= current_obj)
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
