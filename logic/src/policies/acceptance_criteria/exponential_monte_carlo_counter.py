"""Exponential Monte Carlo with Counter (EMC-Q) Acceptance Criterion.

Stochastic acceptance logic that boosts the acceptance probability after a
threshold number of consecutive rejections to prevent stagnation.

Attributes:
    EMCQAcceptance: The EMCQAcceptance criterion.

Example:
    >>> from logic.src.policies.acceptance_criteria.exponential_monte_carlo_counter import EMCQAcceptance
    >>> # Initialize for minimization with baseline p=0.1 and boosted p=0.5
    >>> criterion = EMCQAcceptance(p=0.1, p_boost=0.5, q_threshold=100)
    >>> criterion.setup(initial_objective=100.0)
    >>> accepted, metrics = criterion.accept(current_obj=100.0, candidate_obj=102.0)
    False, {'accepted': False, 'delta': 2.0, 'rejection_counter': 0, 'q_threshold': 100, 'p_active': 0.1, 'maximization': False}
"""

import random
from typing import Any, Callable, Dict, Optional, Tuple, Union, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("emc_q")
class EMCQAcceptance(IAcceptanceCriterion):
    """Exponential Monte Carlo with Counter (EMC-Q) Acceptance Criterion.

    Tracks consecutive rejections. After Q failures, the fixed probability p
    is replaced by a temporary boost, preventing indefinite stagnation on flat
    regions. Reference: Franzin & Stützle (2019).

    Mathematical Formulation:
    P(A | worsening) = p        if q < Q
    P(A | worsening) = p_boost  if q >= Q
    where q is the consecutive rejection counter and Q is the threshold.

    Attributes:
        p (float): Baseline acceptance probability for worsening moves.
        p_boost (float): Boosted acceptance probability.
        q_threshold (Union[int, Callable[[], int]]): Threshold Q for boosting.
        maximization (bool): Whether the problem is maximization.
        rng (random.Random): Random number generator.
        rejection_counter (int): Current number of consecutive rejections.
    """

    def __init__(
        self,
        p: float = 0.05,
        p_boost: float = 0.5,
        q_threshold: Union[int, Callable[[], int]] = 100,
        seed: Optional[int] = None,
        maximization: bool = False,
    ):
        """Initialize the EMC-Q criterion.

        Args:
            p: Baseline acceptance probability. Defaults to 0.05.
            p_boost: Boosted probability. Defaults to 0.5.
            q_threshold: The rejection counter threshold Q.
                Defaults to 100.
            seed: Random seed. Defaults to None.
            maximization: Whether the problem is maximization. Defaults to False.
        """
        self.p = p
        self.p_boost = p_boost
        self.q_threshold = q_threshold
        self.maximization = maximization

        self.rng = random.Random(seed)
        self.rejection_counter = 0

    def setup(self, initial_objective: ObjectiveValue) -> None:
        """Initialize the rejection counter.

        Args:
            initial_objective: The initial solution's objective.
        """
        initial_objective = cast(float, initial_objective)
        self.rejection_counter = 0

    def _get_q_threshold(self) -> int:
        """Evaluate the threshold Q if it is a callable.

        Returns:
            int: The threshold value.
        """
        if callable(self.q_threshold):
            return int(self.q_threshold())
        return int(self.q_threshold)

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        """Determine whether to accept based on the counter-boosted probability.

        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            kwargs: Additional context.

        Returns:
            Tuple[bool, AcceptanceMetrics]: A tuple containing:
                - accepted (bool): True if candidate is accepted.
                - metrics (AcceptanceMetrics): Performance metadata.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        delta = candidate_obj - current_obj
        if self.maximization:
            delta = -delta

        # Always accept improving or equal moves
        if delta <= 0:
            _accepted = bool(True)
            return _accepted, {
                "accepted": _accepted,
                "delta": candidate_obj - current_obj,
                "rejection_counter": self.rejection_counter,
                "q_threshold": self._get_q_threshold(),
                "p_active": self.p if self.rejection_counter < self._get_q_threshold() else self.p_boost,
                "maximization": self.maximization,
            }

        # Worsening move: check counter q against threshold Q
        q_thresh = self._get_q_threshold()
        prob = self.p if self.rejection_counter < q_thresh else self.p_boost

        _accepted = bool(self.rng.random() < prob)
        return _accepted, {
            "accepted": _accepted,
            "delta": candidate_obj - current_obj,
            "rejection_counter": self.rejection_counter,
            "q_threshold": self._get_q_threshold(),
            "p_active": self.p if self.rejection_counter < self._get_q_threshold() else self.p_boost,
            "maximization": self.maximization,
        }

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        """Update the rejection counter.

        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            accepted: Whether the move was accepted.
            kwargs: Additional context.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        if accepted:
            # Reset counter on any acceptance (improving or worsening)
            self.rejection_counter = 0
        else:
            # Increment on rejection
            self.rejection_counter += 1

    def get_state(self) -> Dict[str, Any]:
        """Return the current rejection counter and parameters.

        Returns:
            Dict[str, Any]: State dictionary.
        """
        return {
            "rejection_counter": self.rejection_counter,
            "q_threshold": self._get_q_threshold(),
            "p_active": self.p if self.rejection_counter < self._get_q_threshold() else self.p_boost,
            "maximization": self.maximization,
        }
