"""Binary Tournament Acceptance (BTA) Criterion.

Stochastic criterion based on a competition between the current and candidate
solutions.

Attributes:
    BinaryTournamentAcceptance: The BTA acceptance criterion.

Example:
    >>> from logic.src.policies.acceptance_criteria.binary_tournament_acceptance import BinaryTournamentAcceptance
    >>> criterion = BinaryTournamentAcceptance(p=0.8)
    >>> criterion.setup(initial_objective=100.0)
    >>> accepted, metrics = criterion.accept(current_obj=100.0, candidate_obj=98.0)
    False, {'accepted': False, 'delta': -2.0, 'tournament_probability': 0.8}
"""

import random
from typing import Any, Dict, Optional, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("bta")
class BinaryTournamentAcceptance(IAcceptanceCriterion):
    """Binary Tournament stochastic acceptance.

    Evaluates the current solution against the candidate solution. The superior
    solution "wins" the tournament with probability `p` (where `p` > 0.5), and
    the inferior solution wins with probability `1 - p`.

    Attributes:
        p (float): Probability that the superior solution wins.
        rng (random.Random): Random number generator.
    """

    def __init__(self, p: float = 0.8, seed: Optional[int] = None) -> None:
        """Initialize the Binary Tournament criterion.

        Args:
            p: The probability that the superior solution wins.
                Typically in [0.5, 1.0]. Defaults to 0.8.
            seed: Random seed. Defaults to None.
        """
        self.p = p
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
        """Perform a binary tournament between current and candidate solutions.

        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            kwargs: Additional context.

        Returns:
            Tuple[bool, AcceptanceMetrics]: A tuple containing:
                - accepted (bool): True if candidate wins the tournament.
                - metrics (AcceptanceMetrics): Performance metadata.

        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        roll = self.rng.random()
        candidate_is_superior = candidate_obj > current_obj

        if candidate_is_superior:
            # Candidate gets the high probability
            _accepted = bool(roll < self.p)
            return _accepted, {
                "accepted": _accepted,
                "delta": candidate_obj - current_obj,
                "tournament_probability": self.p,
            }
        else:
            # Candidate is worse, so it gets the low probability
            _accepted = bool(roll > self.p)
            return _accepted, {
                "accepted": _accepted,
                "delta": candidate_obj - current_obj,
                "tournament_probability": self.p,
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
        """Return the current tournament probability.

        Returns:
            Dict[str, Any]: State dictionary.
        """
        return {"tournament_probability": self.p}
