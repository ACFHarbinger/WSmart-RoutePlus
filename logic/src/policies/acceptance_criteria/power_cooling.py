"""Power-cooling acceptance used by the original SANS annealing loop.

Attributes:
    PowerCoolingAcceptance: ``T = T_initial / i ** T_param`` with the caller's NumPy stream.

Example:
    >>> import numpy as np
    >>> from logic.src.policies.acceptance_criteria.power_cooling import PowerCoolingAcceptance
    >>> criterion = PowerCoolingAcceptance(np_rng=np.random.default_rng(0))
    >>> criterion.accept(current_obj=1.0, candidate_obj=1.0, iteration=1, t_initial=10.0, t_param=1.0)[0]
    True
"""

import math
from typing import Any, Dict, Tuple

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue


class PowerCoolingAcceptance(IAcceptanceCriterion):
    """Acceptance of ``run_annealing_loop``.

    Non-worsening moves (``delta >= 0``) are accepted with no random draw.
    Worsening moves recompute ``T = t_initial / iteration ** t_param`` and are
    accepted when ``T > 0`` and ``exp(delta / T) >= np_rng.uniform(0, 1)``.
    A non-positive temperature draws nothing. Temperature is not stored: the
    iteration index selects it, so :meth:`step` does not cool.

    Attributes:
        np_rng: NumPy generator the worsening branch draws from.
    """

    def __init__(self, np_rng: Any):
        """Initialize the power-cooling acceptance.

        Args:
            np_rng: The search's NumPy generator. Only a rejected-or-worsening
                move with positive temperature calls ``uniform(0, 1)``.
        """
        self.np_rng = np_rng

    def setup(self, initial_objective: ObjectiveValue) -> None:
        """No-op: each decision recomputes its temperature.

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
        """Decide acceptance for one annealing iteration.

        Args:
            current_obj: Profit of the incumbent (``previous_sol_profit``).
            candidate_obj: Profit of the neighbour (``current_sol_profit``).
            kwargs: ``iteration``, ``t_initial`` and ``t_param`` of the schedule.

        Returns:
            Tuple of (accepted, metrics). ``delta`` is candidate minus current.
        """
        # Scalars are the SANS profit. A sequence uses its first entry, same as LegacySAAcceptance.
        current = float(current_obj) if isinstance(current_obj, (int, float)) else float(current_obj[0])
        candidate = float(candidate_obj) if isinstance(candidate_obj, (int, float)) else float(candidate_obj[0])
        delta = candidate - current
        if delta >= 0:
            return True, {"accepted": True, "delta": delta}
        iteration = kwargs["iteration"]
        temperature = kwargs["t_initial"] / (iteration ** kwargs["t_param"])
        if temperature > 0 and math.exp(delta / temperature) >= self.np_rng.uniform(0, 1):
            return True, {"accepted": True, "delta": delta, "temperature": temperature}
        return False, {"accepted": False, "delta": delta, "temperature": temperature}

    def step(
        self,
        current_obj: ObjectiveValue,
        candidate_obj: ObjectiveValue,
        accepted: bool,
        **kwargs: Any,
    ) -> None:
        """No-op: the schedule depends on the iteration index, not stored heat.

        Args:
            current_obj: Objective of the incumbent (unused).
            candidate_obj: Objective of the candidate (unused).
            accepted: Decision result (unused).
            kwargs: Additional context (unused).
        """
        return None

    def get_state(self) -> Dict[str, Any]:
        """Return an empty state. Temperature is recomputed per decision.

        Returns:
            Empty dictionary.
        """
        return {}
