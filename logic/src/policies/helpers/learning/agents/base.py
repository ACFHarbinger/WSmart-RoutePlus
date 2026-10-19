"""Base Agent Module.

This module defines the abstract base class for all Reinforcement Learning (RL)
agents in the system, establishing a common interface for action selection,
updates, and state persistence.

Attributes:
    RLAgent: Abstract base class for all RL agents.

Example:
    >>> from logic.src.policies.helpers.learning.agents.base import RLAgent
    >>> # RLAgent is abstract and should be subclassed.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class RLAgent(ABC):
    """Abstract Base Class for all Reinforcement Learning agents.

    This interface defines the fundamental contract for agents used in
    operator selection, parameter control, and route optimization.
    Implementing classes must provide specific logic for action selection
    and model updates.

    Attributes:
        None (abstract class).
    """

    @abstractmethod
    def select_action(self, state: Any, rng: np.random.Generator) -> int:
        """
        Select an action based on the current state.

        Args:
            state: The current environment state or context vector.
            rng: Random number generator for stochastic exploration.

        Returns:
            The index of the selected action.
        """
        pass

    @abstractmethod
    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        """
        Update the agent's internal model based on observed transition.

        Args:
            state: The state where the action was taken.
            action: The action that was performed.
            reward: The reward received after taking the action.
            next_state: The state reached after the action.
            done: Whether the episode or sequence has ended.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Serialize the agent state to disk.

        Args:
            path: Absolute path to the destination file.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        De-serialize the agent state from disk.

        Args:
            path: Absolute path to the source file.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the agent's internal tracking and counters to initial state.
        """
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieve diagnostic statistics and performance metrics.

        Returns:
            A dictionary containing key-value pairs of agent metrics.
        """
        return {}
