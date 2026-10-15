"""Pareto Dominance Acceptance Criterion.

Standard multi-objective acceptance logic based on strict Pareto superiority.

Attributes:
    ParetoDominanceAcceptance: The Pareto Dominance acceptance criterion.

Example:
    >>> from logic.src.policies.acceptance_criteria.pareto_dominance import ParetoDominanceAcceptance
    >>> criterion = ParetoDominanceAcceptance()
    >>> accepted, metrics = criterion.accept(current_obj=[100.0, 50.0], candidate_obj=[98.0, 48.0])
    False, {'accepted': False, 'delta': 0.0}
    >>> accepted, metrics = criterion.accept(current_obj=[100.0, 50.0], candidate_obj=[102.0, 51.0])
    True, {'accepted': True, 'delta': 0.0}
"""

from typing import Any, Dict, Sequence, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("pd")
class ParetoDominanceAcceptance(IAcceptanceCriterion):
    """Pareto-Dominance Acceptance Criterion.

    Accepts a candidate only if it strictly dominates the current solution across
    all objectives (Pareto superiority).

    This expects the `current_obj` and `candidate_obj` parameters to actually be
    iterables of floats (e.g., tuple or list of objective values) rather than
    scalar floats.

    Attributes:
        None
    """

    def setup(self, initial_objective: ObjectiveValue) -> None:
        """Initialize the criterion state.

        Args:
            initial_objective: The initial solution's objective.
        """
        pass

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        """Determine whether the candidate strictly dominates the current solution.

        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            kwargs: Additional context.

        Returns:
            Tuple[bool, AcceptanceMetrics]: A tuple containing:
                - accepted (bool): True if candidate strictly dominates current.
                - metrics (AcceptanceMetrics): Performance metadata.

        Raises:
            ValueError: If objective dimensions do not match.
        """
        # Multi-objective criteria expect Sequence[float]
        cur_objs = cast(Sequence[float], current_obj)
        cand_objs = cast(Sequence[float], candidate_obj)

        if len(cur_objs) != len(cand_objs):
            raise ValueError("Objective dimensions must match for Pareto dominance.")

        # Candidate dominates if it is >= in all objectives, and > in at least one
        # Assuming maximization for all objectives (as per VRPP structure)
        better_in_one = False
        for cur_val, cand_val in zip(cur_objs, cand_objs, strict=True):
            if cand_val < cur_val:
                _accepted = bool(False)
                return _accepted, {"accepted": _accepted, "delta": 0.0}
            if cand_val > cur_val:
                better_in_one = True

        _accepted = bool(better_in_one)
        return _accepted, {"accepted": _accepted, "delta": 0.0}

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        """No-op update step.

        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            accepted: Whether the move was accepted.
            kwargs: Additional context.
        """
        pass

    def get_state(self) -> Dict[str, Any]:
        """Return an empty state dictionary.

        Returns:
            Dict[str, Any]: Empty dictionary.
        """
        return {}
