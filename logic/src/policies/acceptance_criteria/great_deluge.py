"""Great Deluge (GD) Acceptance Criterion.

Deterministic acceptance algorithm based on a rising 'water level' threshold.

Attributes:
    GreatDelugeAcceptance: The GD criterion.

Example:
    >>> from logic.src.policies.acceptance_criteria.great_deluge import GreatDelugeAcceptance
    >>> # Initialize for maximization with 10% target improvement over 1000 iterations
    >>> criterion = GreatDelugeAcceptance(target_fitness_multiplier=1.1, max_iterations=1000)
    >>> criterion.setup(initial_objective=100.0)
    >>> accepted, metrics = criterion.accept(current_obj=100.0, candidate_obj=105.0)
    True, {'accepted': True, 'delta': 5.0, 'water_level': 100.0, 'rain_speed': 0.01}
"""

from typing import Any, Dict, Optional, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("gd")
class GreatDelugeAcceptance(IAcceptanceCriterion):
    """Deterministic Great Deluge algorithm.

    Accepts a candidate if its objective is greater than or equal to a monotonically
    rising "water level". The water level rises linearly from the initial objective
    to a target objective over a predefined number of iterations.

    Attributes:
        target_multiplier (float): Expected relative improvement target.
        max_iterations (int): Total budget of iterations to reach the target.
        water_level (float): Current acceptance threshold.
        rain_speed (float): Increment added to the water level at each step.
    """

    def __init__(
        self,
        target_fitness_multiplier: Optional[float] = None,
        max_iterations: int = 1000,
        target_objective: Optional[float] = None,
    ):
        """Initialize the Great Deluge criterion.

        Args:
            target_fitness_multiplier: The expected relative improvement
                (e.g., 1.1 implies a 10% expected improvement over the initial solution).
            max_iterations: The total budget of iterations to reach the target.
            target_objective: Absolute terminal boundary, instead of a relative multiplier.
        """
        if (target_fitness_multiplier is None) == (target_objective is None):
            raise ValueError("Specify exactly one of target_fitness_multiplier and target_objective")
        self.target_objective = target_objective
        self.target_multiplier = target_fitness_multiplier
        self.max_iterations = max_iterations
        self.water_level = 0.0
        self.rain_speed = 0.0

    def setup(self, initial_objective: ObjectiveValue) -> None:
        """Initialize the criterion state.

        Args:
            initial_objective: The initial solution's objective.
        """
        initial_objective = cast(float, initial_objective)
        self.water_level = initial_objective
        target_objective = (
            self.target_objective
            if self.target_objective is not None
            else initial_objective * cast(float, self.target_multiplier)
        )

        # Calculate exactly how much the water should rise per step
        if self.max_iterations > 0:
            self.rain_speed = (target_objective - initial_objective) / self.max_iterations
        else:
            self.rain_speed = 0.0

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        """Determine whether to accept based on the current water level.

        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            kwargs: Additional context.

        Returns:
            Tuple[bool, AcceptanceMetrics]: A tuple containing:
                - accepted (bool): True if candidate improves the water level.
                - metrics (AcceptanceMetrics): State metadata including water level.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        _accepted = bool(candidate_obj >= self.water_level)
        return _accepted, {
            "accepted": _accepted,
            "delta": candidate_obj - current_obj,
            "water_level": self.water_level,
            "rain_speed": self.rain_speed,
        }

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        """Increment the water level by the rain speed.

        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            accepted: Whether the move was accepted.
            kwargs: Additional context.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        self.water_level += self.rain_speed

    def get_state(self) -> Dict[str, Any]:
        """Return the current water level and rain speed.

        Returns:
            Dict[str, Any]: State dictionary.
        """
        return {"water_level": self.water_level, "rain_speed": self.rain_speed}
