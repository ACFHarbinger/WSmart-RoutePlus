"""Softmax Bandit Module.

This module implements the Softmax bandit algorithm, which selects arms with a
probability proportional to their estimated values using the Gibbs (Boltzmann)
distribution.

Attributes:
    SoftmaxBandit: Bandit agent using the Softmax policy.

Example:
    >>> from logic.src.policies.helpers.learning.agents.bandits.softmax import SoftmaxBandit
    >>> agent = SoftmaxBandit(n_arms=3, temperature=0.5)
    >>> action = agent.select_action(state=None, rng=agent.rng)
"""

from typing import Any, Dict, Optional

import numpy as np

from .base import BanditAgent


class SoftmaxBandit(BanditAgent):
    """
    Softmax Sampling Bandit policy.

    Selects arms with a probability proportional to their estimated values using
    the Gibbs (Boltzmann) distribution. Controlled by a 'temperature' parameter.

    Attributes:
        temperature: Exploration temperature.
        probs: Current probability distribution over arms.
    """

    def __init__(self, n_arms: int, temperature: float = 1.0, seed: Optional[int] = None, history_size: int = 50):
        """
        Initialize the Softmax agent.

        Args:
            n_arms: Number of available actions.
            temperature: Exploration temperature (higher = uniform, lower = greedy).
            seed: Optional seed for the RNG.
            history_size: Size of reward history buffer.
        """
        super().__init__(n_arms, seed, history_size)
        self.temperature = temperature
        self.probs = np.ones(n_arms) / n_arms

    def select_action(self, state: Any, rng: np.random.Generator) -> int:
        """
        Select an arm using Softmax sampling.

        Args:
            state: Contextual state (unused).
            rng: Random number generator.

        Returns:
            The selected arm index.
        """

        # Calculate exponentiated values for probability distribution
        # clip values to prevent overflow in exp
        v = np.clip(self.values / max(self.temperature, 1e-8), -100, 100)
        exp_v = np.exp(v)

        # Normalize to generate probabilities
        self.probs = exp_v / np.sum(exp_v)

        return int(rng.choice(self.n_arms, p=self.probs))

    def get_statistics(self) -> Dict[str, Any]:
        """
        Augment statistics with temperature and selection probabilities.

        Returns:
            Dictionary containing model parameters and current distributions.
        """
        stats = super().get_statistics()
        stats.update({"temperature": self.temperature, "probs": self.probs.tolist()})
        return stats
