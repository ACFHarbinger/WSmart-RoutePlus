"""
Bandit utilities for operator selection.

Attributes:
    ThompsonBandit: Implementation of Thompson Sampling for operator selection.

Example:
    >>> bandit = ThompsonBandit(["op1", "op2"])
    >>> selected = bandit.select()
    >>> bandit.update(selected, 1)
"""

from typing import Dict, List, Tuple

import numpy as np


class ThompsonBandit:
    """Thompson Sampling bandit using Beta-Bernoulli conjugate prior.

    Attributes:
        names (List[str]): Names of the alternative operators.
        rng (np.random.Generator): Local random number generator.
        params (Dict[str, List[float]]): Alpha and beta parameters for each operator.
    """

    def __init__(self, names: List[str], seed: int = 42):
        """Initialise the bandit.

        Args:
            names: List of operator names to choose from.
            seed: Random seed for reproducibility.
        """
        self.names = names
        self.rng = np.random.default_rng(seed)
        # (alpha, beta) parameters
        self.params = {name: [1.0, 1.0] for name in names}

    def select(self) -> str:
        """Select an operator by sampling from Beta distributions.

        Returns:
            str: The name of the selected operator with the highest sampled value.
        """
        samples: Dict[str, float] = {}
        for name, (a, b) in self.params.items():
            samples[name] = float(self.rng.beta(a, b))
        return max(samples, key=lambda x: samples[x])

    def update(self, name: str, success: int):
        """Update parameters based on success (1) or failure (0).

        Args:
            name: Name of the operator to update.
            success: Binary indicator of success (1) or failure (0).
        """
        if name in self.params:
            if success:
                self.params[name][0] += 1
            else:
                self.params[name][1] += 1

    def get_weights(self) -> Dict[str, Tuple[float, float]]:
        """Return the current alpha and beta parameters.

        Returns:
            Dict[str, Tuple[float, float]]: Mapping from operator name to (alpha, beta) tuple.
        """
        return {name: (a, b) for name, (a, b) in self.params.items()}

    def load_weights(self, weights: Dict[str, Tuple[float, float]]):
        """Load external weights.

        Args:
            weights: Dictionary of weights to load.
        """
        for name, (a, b) in weights.items():
            if name in self.params:
                self.params[name] = [a, b]
