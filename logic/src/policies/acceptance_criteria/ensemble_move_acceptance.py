"""Ensemble Move Acceptance (EMA) Criterion.

Aggregates multiple move acceptance criteria into a single decision using
various logical rules (AND, OR, Voting).

Attributes:
    EnsembleAcceptance: The EMA acceptance criterion.

Example:
    >>> from logic.src.policies.acceptance_criteria.ensemble_move_acceptance import EnsembleAcceptance
    >>> from logic.src.policies.acceptance_criteria.boltzmann_metropolis_criterion import BoltzmannAcceptance
    >>> # Initialize sub-criteria
    >>> bm_crit = BoltzmannAcceptance(initial_temp=1000.0, alpha=0.99)
    >>> # Create ensemble with G-VOT rule
    >>> criterion = EnsembleAcceptance(criteria=[bm_crit], rule="G-VOT")
    >>> criterion.setup(initial_objective=100.0)
    >>> accepted, metrics = criterion.accept(current_obj=100.0, candidate_obj=98.0)
    True, {'accepted': True, 'delta': -2.0}
"""

from typing import Any, Dict, List, Optional, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("ema")
class EnsembleAcceptance(IAcceptanceCriterion):
    """Meta-decision architecture combining multiple criteria.

    Evaluates a candidate solution through a portfolio of heterogeneous criteria
    and aggregates the decision via logical ensemble rules (e.g., G-AND, G-OR, G-VOT).

    Attributes:
        criteria (List[IAcceptanceCriterion]): Portfolio of sub-criteria.
        rule (str): Aggregation logic ('G-AND', 'G-OR', 'G-VOT').
    """

    def __init__(
        self,
        criteria: Optional[List[IAcceptanceCriterion]] = None,
        rule: str = "G-VOT",
        methods: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
    ):
        """Initialize the Ensemble criterion.

        Args:
            criteria: The instantiated sub-criteria.
            rule: The aggregation logic ('G-AND', 'G-OR', 'G-VOT').
                Defaults to 'G-VOT'.
            methods: Registry names from ``ac_ema.yaml``. Each is built
                into a criterion when ``criteria`` is omitted.
            weights: Voting weights from the same file.

        Raises:
            ValueError: If the criteria list is empty.
        """
        if not criteria and methods:
            from logic.src.policies.acceptance_criteria.base.factory import AcceptanceCriterionFactory

            criteria = [AcceptanceCriterionFactory.create(str(name)) for name in methods]
        if not criteria:
            raise ValueError("EnsembleAcceptance requires at least one initialized criterion.")
        self.criteria = criteria
        self.rule = rule.upper()
        self.weights = list(weights) if weights is not None else []

    def setup(self, initial_objective: ObjectiveValue) -> None:
        """Initialize all sub-criteria in the ensemble.

        Args:
            initial_objective: The initial solution's objective.
        """
        initial_objective = cast(float, initial_objective)
        for crit in self.criteria:
            crit.setup(initial_objective)

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        """Aggregate acceptance decisions from all sub-criteria.

        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            kwargs: Additional context.

        Returns:
            Tuple[bool, AcceptanceMetrics]: A tuple containing:
                - accepted (bool): Combined decision based on the ensemble rule.
                - metrics (AcceptanceMetrics): Performance metadata.

        Raises:
            ValueError: If an unknown ensemble rule is specified.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        # Gather votes from all encapsulated criteria (List of Tuple[bool, AcceptanceMetrics])
        _results = [crit.accept(current_obj, candidate_obj) for crit in self.criteria]
        votes = [res[0] for res in _results]
        # Keep metrics for detailed analysis if needed (optional optimization)

        if self.rule == "G-AND":
            _accepted = bool(all(votes))  # Strict consensus
            return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj}
        elif self.rule == "G-OR":
            _accepted = bool(any(votes))  # Authority rule (at least one)
            return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj}
        elif self.rule == "G-VOT":
            _accepted = bool(sum(votes) >= (len(votes) / 2.0))  # Majority rule
            return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj}
        else:
            raise ValueError(f"Unknown EMA rule: {self.rule}")

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        """Advance the internal states of all sub-criteria.

        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            accepted: Whether the move was accepted.
            kwargs: Additional context.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        # Advance the state of all encapsulated criteria
        for crit in self.criteria:
            crit.step(current_obj, candidate_obj, accepted)

    def get_state(self) -> Dict[str, Any]:
        """Return the aggregated states of all underlying criteria.

        Returns:
            Dict[str, Any]: State dictionary mapping criterion index to its state.
        """
        # Aggregate the states of the underlying criteria
        return {f"criterion_{i}": crit.get_state() for i, crit in enumerate(self.criteria)}
