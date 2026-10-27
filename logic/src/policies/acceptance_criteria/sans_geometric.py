"""Geometric simulated-annealing acceptance used by the improved SANS loop.

Attributes:
    SansGeometricAcceptance: Strict-improvement SA with the caller's random stream.

Example:
    >>> import random
    >>> from logic.src.policies.acceptance_criteria.sans_geometric import SansGeometricAcceptance
    >>> criterion = SansGeometricAcceptance(start_temp=10.0, cooling_rate=0.9, rng=random.Random(0))
    >>> criterion.accept(current_obj=1.0, candidate_obj=2.0)[0]
    True
"""

import math
from typing import Any, Dict, Optional, Tuple

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue


class SansGeometricAcceptance(IAcceptanceCriterion):
    """Acceptance of ``improved_simulated_annealing``, drawn from the caller's RNG.

    A candidate is accepted when its profit is strictly higher. Otherwise the
    move is accepted with probability ``exp(delta / T)``, and an overflow
    becomes probability 0 while still consuming one random draw. Cooling is
    ``T *= cooling_rate`` in :meth:`step`. The search loop reheats by assigning
    ``T`` directly.

    Attributes:
        T: Current temperature.
        cooling_rate: Geometric cooling factor applied once per temperature plateau.
        rng: The search's ``random.Random`` stream.
    """

    def __init__(self, start_temp: float, cooling_rate: float, rng: Any):
        """Initialize the geometric SANS acceptance.

        Args:
            start_temp: Initial temperature.
            cooling_rate: Factor applied by :meth:`step`.
            rng: The search's random stream. Worsening moves draw ``rng.random()``.
        """
        self.T = start_temp
        self.cooling_rate = cooling_rate
        self.rng = rng

    def setup(self, initial_objective: ObjectiveValue) -> None:
        """No-op: the temperature starts at ``start_temp``.

        Args:
            initial_objective: Objective of the starting solution (unused).
        """
        return None

    def accept(
        self,
        current_obj: ObjectiveValue,
        candidate_obj: ObjectiveValue,
        **kwargs: Any,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Decide acceptance without cooling.

        Args:
            current_obj: Profit of the incumbent.
            candidate_obj: Profit of the neighbour.
            kwargs: Additional context (unused).

        Returns:
            Tuple of (accepted, metrics). ``delta`` is candidate minus current.
        """
        delta = candidate_obj - current_obj
        if delta > 0:
            return True, {"accepted": True, "delta": delta, "temperature": self.T}
        try:
            prob = math.exp(delta / self.T)
        except OverflowError:
            prob = 0
        accepted = self.rng.random() < prob
        return accepted, {"accepted": accepted, "delta": delta, "temperature": self.T}

    def step(
        self,
        current_obj: Optional[ObjectiveValue] = None,
        candidate_obj: Optional[ObjectiveValue] = None,
        accepted: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        """Cool once per temperature plateau (``T *= cooling_rate``).

        Args:
            current_obj: Objective after the decision (unused).
            candidate_obj: Candidate objective (unused).
            accepted: Decision result (unused).
            kwargs: Additional context (unused).
        """
        self.T *= self.cooling_rate

    def get_state(self) -> Dict[str, Any]:
        """Return the current temperature.

        Returns:
            Dict with the current temperature.
        """
        return {"temperature": self.T}
