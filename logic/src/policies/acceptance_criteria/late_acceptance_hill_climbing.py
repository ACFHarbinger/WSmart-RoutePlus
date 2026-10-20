"""Late Acceptance Hill Climbing (LAHC) Criterion.

Memory-based thresholding utilizing a finite circular array to escape local
optima.

Attributes:
    LateAcceptance: The LAHC criterion.

Example:
    >>> from logic.src.policies.acceptance_criteria.late_acceptance_hill_climbing import LateAcceptance
    >>> criterion = LateAcceptance(queue_size=10)
    >>> criterion.setup(initial_objective=100.0)
    >>> accepted, metrics = criterion.accept(current_obj=100.0, candidate_obj=98.0)
    False, {'accepted': False, 'delta': -2.0, 'pointer': 0, 'current_history_val': 100.0}
"""

from typing import Any, Dict, List, Optional, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("lahc")
class LateAcceptance(IAcceptanceCriterion):
    """Memory-based thresholding utilizing a finite circular array.

    A candidate is accepted if its objective is strictly improving against the
    current solution OR if it is better than the cost encountered exactly L
    steps ago. Reference: Burke & Bykov (2017).

    Attributes:
        L (int): The length of the historical memory queue.
        history (List[float]): Circular array storing historical objective values.
        pointer (int): Current index in the circular array.
    """

    def __init__(
        self,
        queue_size: Optional[int] = None,
        history_length: Optional[int] = None,
        maximization: bool = True,
    ):
        """Initialize the LAHC criterion.

        Args:
            queue_size: The length of the historical memory queue (L).
            history_length: Name used by ``ac_lahc.yaml`` for ``queue_size``.
            maximization: Stored for the packaged file.
        """
        if queue_size is None:
            queue_size = history_length
        if queue_size is None:
            raise TypeError("LateAcceptance.__init__() missing 1 required positional argument: 'queue_size'")
        self.L = max(1, queue_size)
        self.maximization = maximization
        self.history: List[float] = []
        self.pointer = 0

    def setup(self, initial_objective: ObjectiveValue) -> None:
        """Initialize the historical memory with the starting objective.

        Args:
            initial_objective: The initial solution's objective.
        """
        initial_objective = cast(float, initial_objective)
        # Populate the entire circular array with the starting objective
        self.history = [initial_objective] * self.L
        self.pointer = 0

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        """Determine whether to accept based on current and historical costs.

        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            kwargs: Additional context.

        Returns:
            Tuple[bool, AcceptanceMetrics]: A tuple containing:
                - accepted (bool): True if candidate improves current OR historical.
                - metrics (AcceptanceMetrics): State metadata including historical bound.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        # Accept if improving vs current OR improving vs L steps ago
        _accepted = bool(candidate_obj >= current_obj or candidate_obj >= self.history[self.pointer])
        return _accepted, {
            "accepted": _accepted,
            "delta": candidate_obj - current_obj,
            "pointer": self.pointer,
            "current_history_val": self.history[self.pointer],
        }

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        """Update the history queue with the cost of the accepted solution.

        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            accepted: Whether the move was accepted.
            kwargs: Additional context.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        # Always insert the *current accepted state's* cost into the array
        self.history[self.pointer] = current_obj
        self.pointer = (self.pointer + 1) % self.L

    def get_state(self) -> Dict[str, Any]:
        """Return the current pointer and historical bound.

        Returns:
            Dict[str, Any]: State dictionary.
        """
        return {"pointer": self.pointer, "current_history_val": self.history[self.pointer]}
