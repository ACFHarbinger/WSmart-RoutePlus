"""Threshold Accepting (TA) Criterion.

Deterministic counterpart to Simulated Annealing that explicitly bounds
allowable deterioration via a decaying threshold.

Attributes:
    ThresholdAccepting: The Threshold Accepting (TA) acceptance criterion.

Example:
    >>> from logic.src.policies.acceptance_criteria.threshold_accepting import ThresholdAccepting
    >>> criterion = ThresholdAccepting(initial_threshold=10.0, max_iterations=1000)
    >>> accepted, metrics = criterion.accept(current_obj=100.0, candidate_obj=95.0)
    True, {'accepted': True, 'delta': -5.0, 'threshold': 10.0}
"""

from typing import Any, Dict, Optional, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("ta")
class ThresholdAccepting(IAcceptanceCriterion):
    """Deterministic Threshold Accepting.

    A deterministic counterpart to Simulated Annealing. It explicitly bounds the
    allowable deterioration using a threshold parameter that decays linearly
    towards zero over the specified iteration budget.

    Attributes:
        threshold (float): Current absolute tolerance for deterioration.
        decay_rate (float): Amount by which the threshold is reduced at each step.
    """

    def __init__(
        self,
        initial_threshold: float,
        max_iterations: Optional[int] = None,
        decay_rate: Optional[float] = None,
        maximization: bool = True,
    ):
        """Initialize the Threshold Accepting criterion.

        Args:
            initial_threshold: The starting absolute tolerance for
                deterioration.
            max_iterations: The budget used to calculate linear decay.
            decay_rate: Per-step decay from ``ac_ta.yaml``, used when
                ``max_iterations`` is omitted.
            maximization: Stored for the packaged file. The threshold test
                is a bound on deterioration in either direction.
        """
        self.threshold = initial_threshold
        self.maximization = maximization
        if decay_rate is not None:
            self.decay_rate = float(decay_rate)
        elif max_iterations is not None:
            self.decay_rate = initial_threshold / max_iterations if max_iterations > 0 else 0.0
        else:
            raise TypeError("ThresholdAccepting.__init__() missing 1 required positional argument: 'max_iterations'")

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
        """Determine acceptance based on the current decaying threshold.

        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            kwargs: Additional context.

        Returns:
            Tuple[bool, AcceptanceMetrics]: A tuple containing:
                - accepted (bool): True if candidate is within threshold tolerance.
                - metrics (AcceptanceMetrics): Performance metadata.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        # delta = candidate - current. If worsening by 10, delta is -10.
        # If threshold is 15, -10 >= -15 is True.
        delta = candidate_obj - current_obj
        _accepted = bool(delta >= -self.threshold)
        return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj, "threshold": self.threshold}

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        """Apply linear decay to the threshold.

        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            accepted: Whether the move was accepted.
            kwargs: Additional context.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        self.threshold = max(0.0, self.threshold - self.decay_rate)

    def get_state(self) -> Dict[str, Any]:
        """Return the current threshold.

        Returns:
            Dict[str, Any]: State dictionary.
        """
        return {"threshold": self.threshold}
