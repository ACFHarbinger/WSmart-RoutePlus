"""Acceptance criteria for the shared LNS engine (WP-11 P-31).

Attributes:
    LegacySAAcceptance: The pre-P-11 inline simulated-annealing acceptance of the
        PG-CLNS LNS, packaged behind :class:`IAcceptanceCriterion`.

Example:
    >>> import random
    >>> from logic.src.policies.routing.meta_heuristics.large_neighborhood_search.acceptance import (
    ...     LegacySAAcceptance,
    ... )
    >>> criterion = LegacySAAcceptance(start_temp=100.0, cooling_rate=0.95, rng=random.Random(0))
"""

import math
from typing import Any, Dict, Optional, Tuple

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue


class LegacySAAcceptance(IAcceptanceCriterion):
    """Simulated-annealing acceptance reproducing the pre-P-31 PG-CLNS LNS.

    Maximization form: improving candidates (delta < 0) are always accepted;
    worsening candidates are accepted with probability ``exp(-delta / T)``.
    Cooling is geometric and happens in :meth:`step`, so the temperature
    trajectory is bit-identical to the legacy inline loop.

    The random draws come from the *solver's* ``random`` stream (not a private
    generator), preserving the pre-P-31 RNG consumption order.

    Attributes:
        T: Current temperature.
        cooling_rate: Geometric cooling factor applied once per iteration.
        rng: The solver's random stream.
    """

    def __init__(self, start_temp: float, cooling_rate: float, rng: Any):
        """Initialize the legacy SA acceptance.

        Args:
            start_temp: Initial temperature (from the solver config).
            cooling_rate: Geometric cooling factor (from the solver config).
            rng: The solver's ``random.Random`` stream.
        """
        self.T = start_temp
        self.cooling_rate = cooling_rate
        self.rng = rng

    def setup(self, initial_objective: ObjectiveValue) -> None:
        """No-op: the legacy acceptance has no setup state.

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
        """Decide acceptance without side effects (no cooling here).

        Args:
            current_obj: Objective of the current incumbent.
            candidate_obj: Objective of the candidate solution.
            kwargs: Additional context (unused).

        Returns:
            Tuple of (accepted, metrics).
        """
        delta = float(current_obj) - float(candidate_obj)
        if delta < -1e-6:
            return True, {"accepted": True, "delta": delta, "temperature": self.T}
        prob = math.exp(-delta / self.T) if self.T > 0 else 0
        accepted = self.rng.random() < prob
        return accepted, {"accepted": accepted, "delta": delta, "temperature": self.T}

    def step(
        self,
        current_obj: Optional[ObjectiveValue] = None,
        candidate_obj: Optional[ObjectiveValue] = None,
        accepted: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        """Apply geometric cooling (``T *= cooling_rate``).

        Args:
            current_obj: Objective after the decision (unused).
            candidate_obj: Candidate objective (unused).
            accepted: Decision result (unused).
            kwargs: Additional context (unused).
        """
        self.T *= self.cooling_rate

    def get_state(self) -> Dict[str, Any]:
        """Return the current temperature for telemetry.

        Returns:
            Dict with the current temperature.
        """
        return {"temperature": self.T}
