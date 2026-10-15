"""Monte Carlo Acceptance (MCA) Criterion.

Stochastic criterion that accepts worsening moves with a fixed, unannealed
probability.

Attributes:
    MonteCarloAcceptance: The MCA criterion.

Example:
    >>> from logic.src.policies.acceptance_criteria.monte_carlo import MonteCarloAcceptance
    >>> criterion = MonteCarloAcceptance(p=0.1)
    >>> accepted, metrics = criterion.accept(current_obj=100.0, candidate_obj=98.0)
    False, {'accepted': False, 'delta': -2.0, 'acceptance_probability': 0.1}
"""

import random
from typing import Any, Dict, Optional, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("mc")
class MonteCarloAcceptance(IAcceptanceCriterion):
    """Fixed-probability stochastic acceptance.

    Deterministically accepts improving candidates. Worsening candidates are
    accepted with a fixed, unannealed probability `p`.

    Attributes:
        p (float): Probability of accepting a worsening move.
        rng (random.Random): Random number generator.
    """

    def __init__(self, p: float = 0.1, seed: Optional[int] = None) -> None:
        """Initialize the Monte Carlo criterion.

        Args:
            p: The probability of accepting a worsening move (0.0 to 1.0).
                Defaults to 0.1.
            seed: Random seed. Defaults to None.
        """
        self.p = max(0.0, min(1.0, p))
        self.rng = random.Random(seed)

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
        """Determine acceptance based on a roll against fixed probability p.

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
        if candidate_obj >= current_obj:
            _accepted = bool(True)
            return _accepted, {
                "accepted": _accepted,
                "delta": candidate_obj - current_obj,
                "acceptance_probability": self.p,
            }
        _accepted = bool(self.rng.random() < self.p)
        return _accepted, {
            "accepted": _accepted,
            "delta": candidate_obj - current_obj,
            "acceptance_probability": self.p,
        }

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
        """Return the fixed acceptance probability.

        Returns:
            Dict[str, Any]: State dictionary.
        """
        return {"acceptance_probability": self.p}
