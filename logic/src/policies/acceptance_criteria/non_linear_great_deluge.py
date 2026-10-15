"""Non-Linear Great Deluge (NLGD) Acceptance Criterion.

Implements a convex water-level decay for convergence in metaheuristic search.

Attributes:
    NonLinearGreatDeluge: The NLGD criterion.

Example:
    >>> from logic.src.policies.acceptance_criteria.non_linear_great_deluge import NonLinearGreatDeluge
    >>> criterion = NonLinearGreatDeluge(t_max=1000, initial_tolerance=0.1, gap_epsilon=0.01, beta=5.0, maximization=False)
    >>> criterion.setup(initial_objective=100.0)
    >>> accepted, metrics = criterion.accept(current_obj=100.0, candidate_obj=98.0)
    True, {'accepted': True, 'delta': -2.0, 'water_level': 110.0, 'iteration': 0, 'progress': 0.0, 'maximization': False}
"""

import math
from typing import Any, Dict, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("nlgd")
class NonLinearGreatDeluge(IAcceptanceCriterion):
    """Non-Linear Great Deluge (NLGD) Acceptance Criterion.

    Implements a convex water-level decay that tracks the global best solution
    f_best with a small tolerance gap epsilon. This ensures exploration while
    forcing asymptotic convergence. Reference: Landa-Silva et al. (2004).

    Mathematical Formulation:
    Level_t = f_target + (Level_0 - f_target) * exp(-beta * t / t_max)
    where f_target = f_best * (1 - gap_epsilon) for minimization.

    Attributes:
        t_max (int): Total iteration budget.
        initial_tolerance (float): Factor for Level_0 relative to initial objective.
        gap_epsilon (float): Tolerance gap from f_best to prevent stagnation.
        beta (float): Decay factor (convexity of the decay curve).
        maximization (bool): Whether the problem is maximization.
        level_0 (float): Initial water level.
        water_level (float): Current acceptance threshold.
        t (int): Current iteration index.
        f_best_ever (float): Best objective observed during search.
    """

    def __init__(
        self,
        t_max: int,
        initial_tolerance: float = 0.1,
        gap_epsilon: float = 0.01,
        beta: float = 5.0,
        maximization: bool = False,
    ):
        """Initialize the NLGD criterion.

        Args:
            t_max: Total iteration budget.
            initial_tolerance: Factor for Level_0 relative to initial
                objective. Defaults to 0.1.
            gap_epsilon: Tolerance gap from f_best to prevent stagnation.
                Defaults to 0.01.
            beta: Decay factor (convexity). Defaults to 5.0.
            maximization: Whether the problem is maximization. Defaults to False.
        """
        self.t_max = t_max
        self.initial_tolerance = initial_tolerance
        self.gap_epsilon = gap_epsilon
        self.beta = beta
        self.maximization = maximization

        self.level_0 = 0.0
        self.water_level = 0.0
        self.t = 0
        self.f_best_ever = float("inf") if not maximization else float("-inf")

    def setup(self, initial_objective: ObjectiveValue) -> None:
        """Initialize the criterion state.

        Args:
            initial_objective: The initial solution's objective.
        """
        initial_objective = cast(float, initial_objective)
        if self.maximization:
            self.level_0 = initial_objective * (1.0 - self.initial_tolerance)
            self.f_best_ever = initial_objective
        else:
            self.level_0 = initial_objective * (1.0 + self.initial_tolerance)
            self.f_best_ever = initial_objective
        self.water_level = self.level_0
        self.t = 0

    def _get_f_best(self, **kwargs: Any) -> float:
        """Retrieve the best objective found so far.

        Args:
            kwargs: Should contain 'f_best' if external tracking is used.

        Returns:
            float: The best objective value.
        """
        f_best = kwargs.get("f_best")
        if f_best is not None:
            return float(f_best)
        return self.f_best_ever

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        """Determine whether to accept based on the non-linear water level.



        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            kwargs: Additional context.

        Returns:
            Tuple[bool, AcceptanceMetrics]: A tuple containing:
                - accepted (bool): True if candidate improves the water level.
                - metrics (AcceptanceMetrics): State metadata including progress.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        # f_best tracking for internal fallback
        if self.maximization:
            self.f_best_ever = max(self.f_best_ever, current_obj, candidate_obj)
        else:
            self.f_best_ever = min(self.f_best_ever, current_obj, candidate_obj)

        if self.maximization:
            _accepted = bool(candidate_obj >= self.water_level)
            return _accepted, {
                "accepted": _accepted,
                "delta": candidate_obj - current_obj,
                "water_level": self.water_level,
                "iteration": self.t,
                "progress": self.t / self.t_max if self.t_max > 0 else 0.0,
                "maximization": self.maximization,
            }
        else:
            _accepted = bool(candidate_obj <= self.water_level)
            return _accepted, {
                "accepted": _accepted,
                "delta": candidate_obj - current_obj,
                "water_level": self.water_level,
                "iteration": self.t,
                "progress": self.t / self.t_max if self.t_max > 0 else 0.0,
                "maximization": self.maximization,
            }

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        """Update the iteration counter and re-calculate the water level.

        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            accepted: Whether the move was accepted.
            kwargs: Additional context.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        self.t += 1
        f_best = self._get_f_best(**kwargs)

        # Calculate f_target with epsilon gap
        f_target = f_best * (1.0 + self.gap_epsilon) if self.maximization else f_best * (1.0 - self.gap_epsilon)

        # Non-linear decay schedule
        progress = min(1.0, self.t / self.t_max)
        decay = math.exp(-self.beta * progress)
        self.water_level = f_target + (self.level_0 - f_target) * decay

    def get_state(self) -> Dict[str, Any]:
        """Return the current water level and iteration progress.

        Returns:
            Dict[str, Any]: State dictionary.
        """
        return {
            "water_level": self.water_level,
            "iteration": self.t,
            "progress": self.t / self.t_max if self.t_max > 0 else 0.0,
            "maximization": self.maximization,
        }
