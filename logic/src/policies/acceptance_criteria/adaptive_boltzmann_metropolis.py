"""Adaptive Boltzmann Metropolis Criterion.

Provides a self-tuning cooling schedule based on the observed variance of
objective value deltas during the search process.

Attributes:
    AdaptiveBoltzmannMetropolis: Self-tuning Boltzmann-Metropolis acceptance
        criterion with adaptive temperature initialization.

Example:
    >>> from logic.src.policies.acceptance_criteria.adaptive_boltzmann_metropolis import AdaptiveBoltzmannMetropolis
    >>> criterion = AdaptiveBoltzmannMetropolis(p0=0.5, window_size=100, alpha=0.95)
    >>> criterion.setup(initial_objective=100.0)
    >>> accepted, metrics = criterion.accept(current_obj=100.0, candidate_obj=98.0)
"""

import math
import random
from collections import deque
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("abm")
class AdaptiveBoltzmannMetropolis(IAcceptanceCriterion):
    """Adaptive Boltzmann Metropolis Criterion.

    Statistically rigorously scales the temperature based on the search landscape's
    topography. Tracks the standard deviation of objective variations (sigma_delta_f)
    to dynamically adjust the cooling schedule and initial temperature.

    Formula for T_0:
    T_0 = -sigma_delta_f / ln(p_0)
    where p_0 is the desired probability of accepting a 1-sigma regressing move.

    Attributes:
        p0 (float): Initial target probability for accepting a 1-sigma move.
        window_size (int): Size of the moving window for statistical tracking.
        alpha (float): Cooling rate factor (geometric cooling).
        min_temp (float): Lower bound for temperature.
        maximization (bool): Whether the problem is maximization.
        rng (random.Random): Random number generator.
        deltas (deque): Moving window of observed objective deltas.
        temp (float): Current temperature.
        sigma (float): Observed standard deviation of deltas.
    """

    def __init__(
        self,
        p0: float = 0.5,
        window_size: int = 100,
        alpha: float = 0.95,
        min_temp: float = 1e-6,
        seed: Optional[int] = None,
        maximization: bool = True,
    ):
        """Initialize the ABM criterion.

        Args:
            p0: Initial target probability for accepting a 1-sigma move.
                Defaults to 0.5.
            window_size: Size of the moving window for statistical tracking.
                Defaults to 100.
            alpha: Cooling rate factor (geometric cooling). Defaults to 0.95.
            min_temp: Lower bound for temperature. Defaults to 1e-6.
            seed: Random seed. Defaults to None.
            maximization: Whether the problem is maximization. Defaults to True.
        """
        self.p0 = p0
        self.window_size = window_size
        self.alpha = alpha
        self.min_temp = min_temp
        self.maximization = maximization

        self.rng = random.Random(seed)
        self.deltas: deque[float] = deque(maxlen=window_size)

        self.temp = 0.0  # Initialized during first transitions
        self.sigma = 0.0

    def setup(self, initial_objective: ObjectiveValue) -> None:
        """Initialize the criterion state with the starting objective.

        Args:
            initial_objective: The initial solution's objective.
        """
        initial_objective = cast(float, initial_objective)
        pass

    def _update_stats(self, delta: float) -> None:
        """Append delta magnitude to the window and update sigma and initial temperature.

        Args:
            delta: Difference between candidate and current objective values.
        """
        self.deltas.append(abs(delta))
        if len(self.deltas) >= 5:  # Need a minimum window to compute sigma
            self.sigma = float(np.std(self.deltas))
            if self.sigma > 0 and self.temp == 0.0:
                # Initialize T0 based on first observed sigma
                self.temp = -self.sigma / math.log(self.p0)

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        """Determine whether to accept a transition based on Boltzmann probability.

        Args:
            current_obj: Objective value of the current incumbent solution.
            candidate_obj: Objective value of the proposed candidate solution.
            kwargs: Additional context passed through from the search loop.

        Returns:
            Tuple[bool, AcceptanceMetrics]: A tuple containing:
                - accepted (bool): True if move is accepted.
                - metrics (AcceptanceMetrics): Performance and state metadata.

        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        delta = candidate_obj - current_obj
        if not self.maximization:
            delta = -delta

        # Always accept improving or equal moves
        if delta >= 0:
            _accepted = bool(True)
            return _accepted, {
                "accepted": _accepted,
                "delta": candidate_obj - current_obj,
                "temperature": self.temp,
                "sigma": self.sigma,
                "window_len": len(self.deltas),
            }

        # If temperature is not yet initialized, reject worsening
        if self.temp <= self.min_temp:
            _accepted = bool(False)
            return _accepted, {
                "accepted": _accepted,
                "delta": candidate_obj - current_obj,
                "temperature": self.temp,
                "sigma": self.sigma,
                "window_len": len(self.deltas),
            }

        # Boltzmann probability: P = exp(delta / T)
        # Note: delta is negative here for worsening moves
        prob = math.exp(delta / self.temp)
        _accepted = bool(self.rng.random() < prob)
        return _accepted, {
            "accepted": _accepted,
            "delta": candidate_obj - current_obj,
            "temperature": self.temp,
            "sigma": self.sigma,
            "window_len": len(self.deltas),
        }

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        """Update the statistical window and apply the cooling schedule.

        Args:
            current_obj: Objective value of the current incumbent solution.
            candidate_obj: Objective value of the proposed candidate solution.
            accepted: Whether the candidate was accepted in this iteration.
            kwargs: Additional context passed through from the search loop.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        delta = candidate_obj - current_obj
        if not self.maximization:
            delta = -delta

        self._update_stats(delta)

        # Basic cooling schedule
        if self.temp > self.min_temp:
            self.temp *= self.alpha

    def get_state(self) -> Dict[str, Any]:
        """Return the current internal state for tracking.

        Returns:
            Dict[str, Any]: State dictionary containing temperature and sigma.
        """
        return {"temperature": self.temp, "sigma": self.sigma, "window_len": len(self.deltas)}
