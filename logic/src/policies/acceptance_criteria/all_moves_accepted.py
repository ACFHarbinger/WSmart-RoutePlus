"""All Moves Accepted (AMA) Criterion.

Trivial criterion used for random walks or testing.

Attributes:
    AllMovesAccepted: Trivial acceptance criterion that unconditionally accepts
        every candidate solution.

Example:
    >>> from logic.src.policies.acceptance_criteria.all_moves_accepted import AllMovesAccepted
    >>> criterion = AllMovesAccepted()
    >>> criterion.setup(initial_objective=50.0)
    >>> accepted, metrics = criterion.accept(current_obj=50.0, candidate_obj=30.0)
    >>> assert accepted is True
"""

from typing import Any, Dict, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("ama")
class AllMovesAccepted(IAcceptanceCriterion):
    """Trivial acceptance criterion that accepts every generated candidate.

    This effectively transforms the metaheuristic into an unconstrained
    Random Walk across the search space.

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
        """Unconditionally accept the move.

        Args:
            current_obj: Objective value of the current incumbent solution.
            candidate_obj: Objective value of the proposed candidate solution.
            kwargs: Additional context passed through from the search loop.

        Returns:
            Tuple[bool, AcceptanceMetrics]: A tuple containing:
                - accepted (bool): Always True.
                - metrics (AcceptanceMetrics): Performance metadata.

        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        _accepted = bool(True)
        return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj}

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        """No-op update step.

        Args:
            current_obj: Objective value of the current incumbent solution.
            candidate_obj: Objective value of the proposed candidate solution.
            accepted: Whether the candidate was accepted in this iteration.
            kwargs: Additional context passed through from the search loop.
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
