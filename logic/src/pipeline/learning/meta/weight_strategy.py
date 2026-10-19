"""
Abstract Strategy Pattern for Weight Adjustment.

Attributes:
    WeightAdjustmentStrategy: Interface for strategies that propose and update cost weights for the RL env.

Example:
    None
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class WeightAdjustmentStrategy(ABC):
    """
    Interface for strategies that propose and update cost weights for the reinforcement learning environment.

    Attributes:
        None
    """

    @abstractmethod
    def propose_weights(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Returns the weights for the next simulation day or step.
        Args:
            context: Environment context.
        """
        pass

    @abstractmethod
    def feedback(
        self,
        reward: float,
        metrics: Any,
        day: Optional[int] = None,
        step: Optional[int] = None,
    ):
        """
        Receives feedback from the environment to update internal state.

        Args:
            reward: Reward from the environment.
            metrics: Metrics from the environment.
            day: Day of the simulation.
            step: Step of the simulation.
        """
        pass

    @abstractmethod
    def get_current_weights(self) -> Dict[str, float]:
        """
        Returns the currently held weights.
        """
        pass
