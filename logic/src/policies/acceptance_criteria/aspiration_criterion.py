"""Aspiration Criterion (AC).

Accepts candidates that surpass the best objective found so far, regardless
of other criteria.

Attributes:
    AspirationCriterion: Memory-overriding acceptance criterion that accepts
        only global-best-improving candidates.

Example:
    >>> from logic.src.policies.acceptance_criteria.aspiration_criterion import AspirationCriterion
    >>> criterion = AspirationCriterion()
    >>> criterion.setup(initial_objective=100.0)
    >>> accepted, metrics = criterion.accept(current_obj=95.0, candidate_obj=105.0)
    >>> assert accepted is True
"""

from typing import Any, Dict, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("ac")
class AspirationCriterion(IAcceptanceCriterion):
    """Memory-overriding acceptance criterion.

    Accepts a candidate only if its objective surpasses the global best objective
    found since the criterion was initialized.

    Attributes:
        global_best (float): The best objective value observed during the search.
    """

    def __init__(self) -> None:
        """Initialize the Aspiration criterion."""
        self.global_best = -float("inf")

    def setup(self, initial_objective: ObjectiveValue) -> None:
        """Initialize the global best with the starting objective.

        Args:
            initial_objective: The initial solution's objective.
        """
        initial_objective = cast(float, initial_objective)
        self.global_best = initial_objective

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        """Accept if the candidate surpasses the global best objective.

        Args:
            current_obj: Objective value of the current incumbent solution.
            candidate_obj: Objective value of the proposed candidate solution.
            kwargs: Additional context passed through from the search loop.

        Returns:
            Tuple[bool, AcceptanceMetrics]: A tuple containing:
                - accepted (bool): True if candidate_obj > global_best.
                - metrics (AcceptanceMetrics): Performance metadata.

        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        _accepted = bool(candidate_obj > self.global_best)
        return _accepted, {
            "accepted": _accepted,
            "delta": candidate_obj - current_obj,
            "global_best_objective": self.global_best,
        }

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        """Update the global best objective if a better solution was accepted.



        Args:
            current_obj: Description of current_obj.
            candidate_obj: Description of candidate_obj.
            accepted: Description of accepted.
            kwargs: Description of kwargs.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        if accepted and candidate_obj > self.global_best:
            self.global_best = candidate_obj

    def get_state(self) -> Dict[str, Any]:
        """Return the current global best objective.

        Returns:
            Dict[str, Any]: State dictionary.
        """
        return {"global_best_objective": self.global_best}
