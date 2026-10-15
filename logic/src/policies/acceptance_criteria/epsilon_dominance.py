"""Epsilon-Dominance Acceptance Criterion.

Discretizes objective space into grid boxes to provide a finite archive of
non-dominated solutions.

Attributes:
    EpsilonDominanceCriterion: The Epsilon-Dominance criterion.

Example:
    >>> from logic.src.policies.acceptance_criteria.epsilon_dominance import EpsilonDominanceCriterion
    >>> # Initialize for minimization problem with epsilon=1.0 for both objectives
    >>> criterion = EpsilonDominanceCriterion(epsilon=1.0, maximization=False)
    >>> criterion.setup(initial_objective=100.0)
    >>> accepted, metrics = criterion.accept(current_obj=100.0, candidate_obj=98.0)
    True, {'accepted': True, 'archive_size': 1, 'epsilon': 1.0, 'maximization': False}
"""

import math
from typing import Any, Dict, List, Sequence, Tuple, Union, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("edc")
class EpsilonDominanceCriterion(IAcceptanceCriterion):
    """Epsilon-Dominance Acceptance Criterion with Grid Archiving.

    Ensures a finite archive size and mathematically guaranteed convergence limits
    by discretizing the objective space into epsilon-boxes (hyper-rectangles).

    A candidate solution f_cand is accepted if its epsilon-box is not dominated
    by any existing epsilon-box in the archive. If accepted, it replaces
    dominated solutions in the archive.

    Attributes:
        epsilon (Union[float, Sequence[float]]): Discretization step size for each
            objective.
        maximization (bool): Whether to maximize the objectives.
        archive (List[Tuple[Tuple[int, ...], Tuple[float, ...]]]): List of
            (epsilon_box, objective_vector) pairs representing the Pareto front.
    """

    def __init__(self, epsilon: Union[float, Sequence[float]], maximization: bool = True):
        """Initialize the Epsilon-Dominance criterion.

        Args:
            epsilon: The discretization step size
                for each objective. Can be a scalar for all objectives or a sequence.
            maximization: Whether to maximize the objectives. Defaults to True.
        """
        self.epsilon = epsilon
        self.maximization = maximization
        # Archive of (epsilon_box_tuple, objective_vector_tuple)
        self.archive: List[Tuple[Tuple[int, ...], Tuple[float, ...]]] = []

    def setup(self, initial_objective: ObjectiveValue) -> None:
        """Initialize the criterion state.

        Args:
            initial_objective: The initial solution's objective.
        """
        # Multi-objective handling usually passes a tuple as current_obj/candidate_obj
        # IAcceptanceCriterion interface signature uses 'float' for type-hinting compatibility,
        # but we treat it as an iterable.
        pass

    def _get_epsilon_box(self, objs: Sequence[float]) -> Tuple[int, ...]:
        """Calculate the epsilon-box coordinates for an objective vector.

        Args:
            objs: Vector of objective values.

        Returns:
            Tuple[int, ...]: Grid coordinates of the epsilon-box.
        """
        if isinstance(self.epsilon, (float, int)):
            eps_vec = [float(self.epsilon)] * len(objs)
        else:
            eps_vec = [float(e) for e in self.epsilon]

        if self.maximization:
            # For maximization, we want the grid box to favor higher values
            return tuple(int(math.floor(o / e)) for o, e in zip(objs, eps_vec, strict=False))
        else:
            # For minimization, we favor lower values
            return tuple(int(math.ceil(o / e)) for o, e in zip(objs, eps_vec, strict=False))

    def _dominates(self, box_a: Tuple[int, ...], box_b: Tuple[int, ...]) -> bool:
        """Determine if box_a dominates box_b.

        Args:
            box_a: Grid coordinates of the first box.
            box_b: Grid coordinates of the second box.

        Returns:
            bool: True if box_a dominates box_b.
        """
        if self.maximization:
            # a dominates b if every coord of a >= b AND at least one a > b
            greater_or_equal = all(a >= b for a, b in zip(box_a, box_b, strict=False))
            strictly_greater = any(a > b for a, b in zip(box_a, box_b, strict=False))
            return greater_or_equal and strictly_greater
        else:
            # a dominates b if every coord of a <= b AND at least one a < b
            less_or_equal = all(a <= b for a, b in zip(box_a, box_b, strict=False))
            strictly_less = any(a < b for a, b in zip(box_a, box_b, strict=False))
            return less_or_equal and strictly_less

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        """Determine whether to accept based on epsilon-dominance.

        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            kwargs: Additional context.

        Returns:
            Tuple[bool, AcceptanceMetrics]: A tuple containing:
                - accepted (bool): True if candidate is not dominated.
                - metrics (AcceptanceMetrics): Archive size and parameters.
        """
        cand_vec = cast(Sequence[float], candidate_obj)
        cand_box = self._get_epsilon_box(cand_vec)

        # A candidate is accepted if its box is not dominated by any box in the archive
        _accepted = bool(
            all(not (self._dominates(arc_box, cand_box) or arc_box == cand_box) for arc_box, _ in self.archive)
        )
        return _accepted, {
            "accepted": _accepted,
            "archive_size": len(self.archive),
            "epsilon": self.epsilon,
            "maximization": self.maximization,
        }

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        """Update the archive if the candidate was accepted.

        Args:
            current_obj: Objective of the current solution.
            candidate_obj: Objective of the candidate solution.
            accepted: Whether the move was accepted.
            kwargs: Additional context.
        """
        if not accepted:
            return

        cand_vec = cast(Sequence[float], candidate_obj)
        cand_box = self._get_epsilon_box(cand_vec)

        # Filter out boxes dominated by the new candidate box
        new_archive = []
        for arc_box, arc_vec in self.archive:
            if not self._dominates(cand_box, arc_box):
                new_archive.append((arc_box, arc_vec))

        new_archive.append((cand_box, tuple(cand_vec)))
        self.archive = new_archive

    def get_state(self) -> Dict[str, Any]:
        """Return the current archive size and parameters.

        Returns:
            Dict[str, Any]: State dictionary.
        """
        return {"archive_size": len(self.archive), "epsilon": self.epsilon, "maximization": self.maximization}
