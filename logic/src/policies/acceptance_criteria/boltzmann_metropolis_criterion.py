"""Boltzmann-Metropolis Criterion (BMC).

Implements the classic Simulated Annealing acceptance logic with a geometric
cooling schedule.

Attributes:
    BoltzmannAcceptance: The BMC acceptance criterion.

Example:
    >>> from logic.src.policies.acceptance_criteria.boltzmann_metropolis_criterion import BoltzmannAcceptance
    >>> criterion = BoltzmannAcceptance(initial_temp=1000.0, alpha=0.995, seed=42)
    >>> criterion.setup(initial_objective=10.0)
    >>> accepted, metrics = criterion.accept(current_obj=10.0, candidate_obj=12.0)
    >>> accepted
    True
    >>> criterion.accept(current_obj=10.0, candidate_obj=10.0)[0]  # ties are accepted
    True
"""

import math
import random
from typing import Any, Dict, Optional, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("bmc")
class BoltzmannAcceptance(IAcceptanceCriterion):
    """Metropolis-Boltzmann acceptance criterion.

    Improving moves are accepted deterministically. Worsening moves are accepted
    with probability P = exp(Δf / T), where Δf < 0 for maximization.

    Attributes:
        T (float): Current temperature parameter.
        alpha (float): Geometric cooling rate factor.
        rng (random.Random): Random number generator.
    """

    def __init__(self, initial_temp: float, alpha: float, seed: Optional[int] = 42):
        """Initialize the Boltzmann criterion.

        Args:
            initial_temp (float): The starting temperature parameter.
            alpha (float): The geometric cooling rate (typically 0.90 - 0.995).
            seed (Optional[int]): Random seed. Defaults to 42.
        """
        self.T = initial_temp
        self.alpha = alpha
        self.rng = random.Random(seed)

    def setup(self, initial_objective: ObjectiveValue) -> None:
        """Initialize the criterion state.

        Args:
            initial_objective (ObjectiveValue): The initial solution's objective.
        """
        initial_objective = cast(float, initial_objective)
        pass

    @property
    def get_temperature(self) -> float:
        """Get the current temperature parameter.

        Returns:
            float: The current temperature.
        """
        return self.T

    def set_temperature(self, value: float) -> None:
        """Set the current temperature parameter.

        Args:
            value (float): The new temperature value.
        """
        self.T = value

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        """Determine whether to accept a transition based on current temperature.

        Args:
            current_obj (ObjectiveValue): Objective of the current solution.
            candidate_obj (ObjectiveValue): Objective of the candidate solution.
            kwargs (Any): Additional context.

        Returns:
            Tuple[bool, AcceptanceMetrics]: A tuple containing:
                - accepted (bool): True if move is accepted.
                - metrics (AcceptanceMetrics): State metadata including temperature.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        delta = candidate_obj - current_obj
        if delta >= 0:
            _accepted = bool(True)
            return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj, "temperature": self.T}

        # delta is negative, T is positive -> exp(negative) gives valid [0, 1] probability
        if self.T <= 1e-9:
            _accepted = bool(False)
            return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj, "temperature": self.T}

        _accepted = bool(self.rng.random() < math.exp(delta / self.T))
        return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj, "temperature": self.T}

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        """Apply geometric cooling to the temperature.

        Args:
            current_obj (ObjectiveValue): Objective of the current solution.
            candidate_obj (ObjectiveValue): Objective of the candidate solution.
            accepted (bool): Whether the move was accepted.
            kwargs (Any): Additional context.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        # Geometric decay
        self.T *= self.alpha

    def get_state(self) -> Dict[str, Any]:
        """Return the current temperature.

        Returns:
            Dict[str, Any]: State dictionary containing 'temperature'.
        """
        return {"temperature": self.T}
