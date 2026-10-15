"""Demon Algorithm Acceptance Criterion.

Deterministic criterion that uses a 'demon' credit budget to autorize worsening
moves.

Attributes:
    DemonAlgorithm: The Demon Algorithm acceptance criterion.

Example:
    >>> from logic.src.policies.acceptance_criteria.demon_algorithm import DemonAlgorithm
    >>> criterion = DemonAlgorithm(warm_up_steps=5, maximization=False)
    >>> criterion.setup(initial_objective=100.0)
    >>> accepted, metrics = criterion.accept(current_obj=100.0, candidate_obj=98.0)
    True, {'accepted': True, 'delta': -2.0, 'demon_credit': 0.0, 'warmed_up': False, 'maximization': False}
"""

from typing import Any, Dict, List, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("da")
class DemonAlgorithm(IAcceptanceCriterion):
    """Demon Algorithm Acceptance Criterion.

    A deterministic alternative to Simulated Annealing that uses a "demon" credit budget.
    Improving moves populate the credit, and worsening moves are accepted only if
    authorized by the current credit balance. Reference: Wood (2000).

    Mathematical Formulation (Minimization):
    - Improving Move (delta_f < 0): Always accepted; D = abs(delta_f).
    - Worsening Move (delta_f > 0): Accepted if delta_f <= D. If accepted, D = D - delta_f.

    Attributes:
        warm_up_steps (int): Number of initial moves for estimation.
        maximization (bool): Whether the problem is maximization.
        demon_credit (float): Current authorized credit for worsening moves.
        history (List[float]): Magnitudes of deltas observed during warm-up.
        _warmed_up (bool): Flag indicating if warm-up is complete.
    """

    def __init__(self, warm_up_steps: int = 5, maximization: bool = False):
        """Initialize the Demon Algorithm.

        Args:
            warm_up_steps: Number of initial moves used to estimate initial
                demon credit. Defaults to 5.
            maximization: Whether the problem is maximization. Defaults to False.
        """
        self.warm_up_steps = warm_up_steps
        self.maximization = maximization

        self.demon_credit = 0.0
        self.history: List[float] = []
        self._warmed_up = False

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
        """Determine whether to accept based on current demon credit.

        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            kwargs: Additional context.

        Returns:
            Tuple[bool, AcceptanceMetrics]: A tuple containing:
                - accepted (bool): True if candidate is autorized by credit.
                - metrics (AcceptanceMetrics): Performance metadata.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        delta = candidate_obj - current_obj
        if self.maximization:
            delta = -delta

        # Always accept improving or equal moves (delta <= 0 in minimization)
        if delta <= 0:
            _accepted = bool(True)
            return _accepted, {
                "accepted": _accepted,
                "delta": candidate_obj - current_obj,
                "demon_credit": self.demon_credit,
                "warmed_up": self._warmed_up,
                "maximization": self.maximization,
            }

        # If not warmed up, we typically reject worsening unless credit was somehow set
        if not self._warmed_up and self.demon_credit <= 0:
            _accepted = bool(False)
            return _accepted, {
                "accepted": _accepted,
                "delta": candidate_obj - current_obj,
                "demon_credit": self.demon_credit,
                "warmed_up": self._warmed_up,
                "maximization": self.maximization,
            }

        # Worsening move: accepted if delta <= credit
        _accepted = bool(delta <= self.demon_credit)
        return _accepted, {
            "accepted": _accepted,
            "delta": candidate_obj - current_obj,
            "demon_credit": self.demon_credit,
            "warmed_up": self._warmed_up,
            "maximization": self.maximization,
        }

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        """Update the demon credit based on the transition result.

        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            accepted: Whether the move was accepted.
            kwargs: Additional context.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        delta = candidate_obj - current_obj
        if self.maximization:
            delta = -delta

        # State update logic
        if not self._warmed_up:
            self.history.append(abs(delta))
            if len(self.history) >= self.warm_up_steps:
                self.demon_credit = max(self.history) if self.history else 0.0
                self._warmed_up = True
            return

        # Regular operation
        if delta < 0:
            # Improvement: The demon "recharges" with the improvement magnitude
            self.demon_credit = abs(delta)
        elif accepted and delta > 0:
            # Worsening move accepted: Credit is depleted by the cost
            self.demon_credit = max(0.0, self.demon_credit - delta)

    def get_state(self) -> Dict[str, Any]:
        """Return the current demon credit and warm-up status.

        Returns:
            Dict[str, Any]: State dictionary.
        """
        return {
            "demon_credit": self.demon_credit,
            "warmed_up": self._warmed_up,
            "maximization": self.maximization,
        }
