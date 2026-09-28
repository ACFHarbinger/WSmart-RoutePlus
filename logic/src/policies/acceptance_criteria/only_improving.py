"""Only Improving (OI) Criterion.

Strictly elitist acceptance strategy that only allows improvements.

Attributes:
    OnlyImproving: The OI criterion.

Example:
    >>> from logic.src.policies.acceptance_criteria.only_improving import OnlyImproving
    >>> criterion = OnlyImproving()
    >>> accepted, metrics = criterion.accept(current_obj=10.0, candidate_obj=12.0)
    >>> accepted
    True
    >>> criterion.accept(current_obj=10.0, candidate_obj=10.0)[0]  # ties are rejected
    False
"""

from typing import Any, Dict, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("oi")
class OnlyImproving(IAcceptanceCriterion):
    """Strictest form of greedy, elitist move acceptance.

    Accepts a candidate only if it yields a strict objective enhancement.

    Attributes:
        None
    """

    def setup(self, initial_objective: ObjectiveValue) -> None:
        """Initialize the criterion state.

        Args:
            initial_objective (ObjectiveValue): The initial solution's objective.
        """
        initial_objective = cast(float, initial_objective)
        pass

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        """Accept only if the candidate objective is strictly greater.

        Args:
            current_obj (ObjectiveValue): Objective of the current solution.
            candidate_obj (ObjectiveValue): Objective of the candidate solution.
            kwargs (Any): Additional context.

        Returns:
            Tuple[bool, AcceptanceMetrics]: A tuple containing:
                - accepted (bool): True if candidate_obj > current_obj.
                - metrics (AcceptanceMetrics): Performance metadata.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        _accepted = bool(candidate_obj > current_obj)
        return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj}

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        """No-op update step.

        Args:
            current_obj (ObjectiveValue): Objective of the current solution.
            candidate_obj (ObjectiveValue): Objective of the candidate solution.
            accepted (bool): Whether the move was accepted.
            kwargs (Any): Additional context.
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
