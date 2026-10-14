"""
Main Neural Agent class assembling mixins.

Attributes:
    NeuralAgent: Neural Agent class.

Example:
    >>> from logic.src.policies.routing.learning_algorithms import NeuralAgent
    >>> agent = NeuralAgent(model)
    >>> routes, metrics = agent.run_day(env)
    >>> print(f"Best routes: {routes}")
    >>> print(f"Metrics: {metrics}")
"""

from __future__ import annotations

from typing import Optional

from .batch import BatchMixin
from .simulation import SimulationMixin


class NeuralAgent(BatchMixin, SimulationMixin):
    """
    Agent interface between simulator/environment and neural routing models.

    Handles model inference, hierarchical decision-making (HRL), and
    route improvement for waste collection routing.

    Attributes:
        model: The neural routing model (AttentionModel, etc.)
        problem: Problem instance (VRPP, WCVRP, etc.) for cost calculation
        seed: Random seed for reproducibility.
    """

    def __init__(self, model, seed: Optional[int] = None):
        """
        Initializes the NeuralAgent.

        Args:
            model: The neural routing model (e.g., AttentionModel).
            seed: Random seed for reproducibility.
        """
        self.model = model
        self.problem = model.problem
        self.seed = seed if seed is not None else 42
