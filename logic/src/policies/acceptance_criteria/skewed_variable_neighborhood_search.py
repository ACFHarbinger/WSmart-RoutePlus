"""Skewed Variable Neighborhood Search (SVNS) Acceptance Criterion.

Accepts candidates based on a combination of objective value and structural
distance to prevent stagnation.

Attributes:
    SkewedVNSAcceptance: The Skewed Variable Neighborhood Search (SVNS) acceptance criterion.

Example:
    >>> from logic.src.policies.acceptance_criteria.skewed_variable_neighborhood_search import SkewedVNSAcceptance
    >>> from logic.src.interfaces.distance_metric import HammingDistance
    >>> metric = HammingDistance(n_customers=5)
    >>> criterion = SkewedVNSAcceptance(alpha=0.1, metric=metric)
    >>> accepted, metrics = criterion.accept(current_obj=100.0, candidate_obj=98.0)
    False, {'accepted': False, 'delta': -2.0, 'alpha': 0.1}
"""

from typing import Any, Dict, Optional, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics
from logic.src.interfaces.distance_metric import IDistanceMetric

from .base.registry import AcceptanceCriterionRegistry


class _ZeroDistance:
    """Structural distance of 0, used when the YAML names no metric."""

    def compute(self, current: Any, candidate: Any) -> float:
        del current, candidate
        return 0.0


@AcceptanceCriterionRegistry.register("svns")
class SkewedVNSAcceptance(IAcceptanceCriterion):
    """Skewed Variable Neighborhood Search (SVNS) Acceptance Criterion.

    Accepts a candidate solution if it is either better than the current solution
    or if it is sufficiently distant (structurally diverse) to justify a slight
    objective deterioration. This prevents the search from being trapped in
    isomorphic basins.

    Mathematical Formulation (Maximization):
    Accept if: f_cand > f_cur - alpha * rho(x_cur, x_cand)
    where rho is the structural distance calculated by an IDistanceMetric.

    Attributes:
        alpha (float): Scaling factor for structural distance.
        metric (IDistanceMetric): Metric for computing structural distance rho.
        maximization (bool): Whether the problem is maximization.
    """

    def __init__(
        self,
        alpha: float,
        metric: Optional[IDistanceMetric] = None,
        maximization: bool = True,
    ):
        """Initialize the Skewed VNS criterion.

        Args:
            alpha: Scaling factor for structural distance.
            metric: Metric for computing structural distance rho.
                ``ac_svns.yaml`` names none; a missing metric contributes 0
                until a caller supplies solutions.
            maximization: Whether the problem is maximization.
                Defaults to True.
        """
        self.alpha = alpha
        self.metric = metric if metric is not None else _ZeroDistance()
        self.maximization = maximization

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
        """Determine acceptance based on objective and structural distance.

        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            kwargs: Additional context.

        Returns:
            Tuple[bool, AcceptanceMetrics]: A tuple containing:
                - accepted (bool): True if candidate satisfies SVNS condition.
                - metrics (AcceptanceMetrics): Performance metadata.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        if self.maximization:
            if candidate_obj > current_obj:
                _accepted = bool(True)
                return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj, "alpha": self.alpha}
        else:
            if candidate_obj < current_obj:
                _accepted = bool(True)
                return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj, "alpha": self.alpha}

        current_sol = kwargs.get("current_sol")
        candidate_sol = kwargs.get("candidate_sol")

        if current_sol is None or candidate_sol is None:
            # Fallback to only improving if structural data is missing
            _accepted = bool(False)
            return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj, "alpha": self.alpha}

        rho = self.metric.compute(current_sol, candidate_sol)

        if self.maximization:
            _accepted = bool(candidate_obj > (current_obj - self.alpha * rho))
            return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj, "alpha": self.alpha}
        else:
            _accepted = bool(candidate_obj < (current_obj + self.alpha * rho))
            return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj, "alpha": self.alpha}

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        """No-op update step.

        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            accepted: Whether the move was accepted.
            kwargs: Additional context.
        """
        current_obj = cast(float, current_obj)

    def get_state(self) -> Dict[str, Any]:
        """Return the current alpha scaling factor.

        Returns:
            Dict[str, Any]: State dictionary.
        """
        return {"alpha": self.alpha}
