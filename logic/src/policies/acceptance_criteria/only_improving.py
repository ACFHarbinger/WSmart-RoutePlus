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

from typing import Any, Dict, Optional, Sequence, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics
from logic.src.tracking.execution_trace import solution_record

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
            initial_objective: The initial solution's objective.
        """
        initial_objective = cast(float, initial_objective)
        pass

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        """Accept only if the candidate objective is strictly greater.

        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            kwargs: Additional context.

        Returns:
            Tuple[bool, AcceptanceMetrics]: A tuple containing:
                - accepted (bool): True if candidate_obj > current_obj.
                - metrics (AcceptanceMetrics): Performance metadata.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        _accepted = bool(candidate_obj > current_obj)
        trace = kwargs.get("trace")
        if trace is not None:
            parent_id = kwargs.get("parent_id")
            if parent_id is None and trace.events:
                parent_id = trace.events[-1]["id"]
            current_routes = cast(Optional[Sequence[Sequence[int]]], kwargs.get("current_routes"))
            candidate_routes = cast(Optional[Sequence[Sequence[int]]], kwargs.get("candidate_routes"))
            before = (
                solution_record(current_routes, objective=current_obj)
                if current_routes is not None
                else solution_record([], objective=current_obj)
            )
            after_routes: Sequence[Sequence[int]] = (
                candidate_routes if _accepted else current_routes
            ) or []
            after = solution_record(after_routes, objective=candidate_obj if _accepted else current_obj)
            arguments: Dict[str, Any] = {
                "accepted": _accepted,
                "current_obj": float(current_obj),
                "candidate_obj": float(candidate_obj),
                "delta": float(candidate_obj - current_obj),
            }
            if "iteration" in kwargs:
                arguments["iteration"] = kwargs["iteration"]
            trace.add(
                name="oi",
                kind="operator",
                parent_id=parent_id,
                changed_solution=_accepted,
                arguments=arguments,
                before=before,
                after=after,
            )
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
