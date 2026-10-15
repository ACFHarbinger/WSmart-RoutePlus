"""Record-to-Record Travel (RRT) Criterion.

Deterministic criterion that accepts solutions within a tolerance deviation
of the global best known solution.

Attributes:
    RecordToRecordTravel: The Record-to-Record Travel acceptance criterion.

Example:
    >>> from logic.src.policies.acceptance_criteria.record_to_record_travel import RecordToRecordTravel
    >>> criterion = RecordToRecordTravel(tolerance=0.05)
    >>> criterion.setup(initial_objective=100.0)
    >>> accepted, metrics = criterion.accept(current_obj=100.0, candidate_obj=98.0)
    True, {'accepted': True, 'delta': -2.0, 'global_best': 100.0, 'tolerance_floor': 95.0}
"""

from typing import Any, Dict, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("rrt")
class RecordToRecordTravel(IAcceptanceCriterion):
    """Deterministic Record-to-Record Travel.

    Accepts a candidate solution if its objective is within a defined percentage
    tolerance (deviation) of the GLOBAL best known solution, rather than the
    immediate predecessor.

    Attributes:
        tolerance (float): Allowed percentage deviation from global best.
        global_best (float): The best objective value found so far.
    """

    def __init__(self, tolerance: float):
        """Initialize the Record-to-Record Travel criterion.

        Args:
            tolerance: The allowed percentage deviation from the global best
                (e.g., 0.05 means solutions within 5% of the global best are accepted).
        """
        self.tolerance = tolerance
        self.global_best = -float("inf")

    def setup(self, initial_objective: ObjectiveValue) -> None:
        """Initialize the global best record.

        Args:
            initial_objective: The initial solution's objective.
        """
        initial_objective = cast(float, initial_objective)
        self.global_best = initial_objective

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        """Determine acceptance relative to the global best record.

        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            kwargs: Additional context.

        Returns:
            Tuple[bool, AcceptanceMetrics]: A tuple containing:
                - accepted (bool): True if candidate is within tolerance of global best.
                - metrics (AcceptanceMetrics): Performance metadata.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        # Assuming strictly positive profit objectives.
        # Accept if candidate is greater than or equal to 95% of the global best.
        allowable_floor = self.global_best * (1.0 - self.tolerance)
        _accepted = bool(candidate_obj >= allowable_floor)
        return _accepted, {
            "accepted": _accepted,
            "delta": candidate_obj - current_obj,
            "global_best": self.global_best,
            "tolerance_floor": self.global_best * (1.0 - self.tolerance),
        }

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        """Update the global best record if a superior solution was accepted.

        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            accepted: Whether the move was accepted.
            kwargs: Additional context.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        # Update the global best record if the new solution dominates it
        if accepted and current_obj > self.global_best:
            self.global_best = current_obj

    def get_state(self) -> Dict[str, Any]:
        """Return the current global best and tolerance floor.

        Returns:
            Dict[str, Any]: State dictionary.
        """
        return {"global_best": self.global_best, "tolerance_floor": self.global_best * (1.0 - self.tolerance)}
