"""
Statistical sampling distributions - Uniform.

Attributes:
    Uniform(low, high):
        Uniform distribution sampling.

Example:
    dist = Uniform()
    data = dist._sample_tensor(torch.Size((10, 50)))
"""

from typing import Optional, Tuple

import numpy as np
import torch

from .base import BaseDistribution


class Uniform(BaseDistribution):
    """Discrete uniform sampling distribution.

    Attributes:
        low (int): Minimum value for randint.
        high (int): Maximum value for randint.
    """

    def __init__(self, low: int = 0, high: int = 100):
        """Initialize Class.

        Args:
            low: Minimum value for randint.
            high: Maximum value for randint.
        """
        self.low = low
        self.high = high

    def _sample_tensor(self, size: Tuple[int, ...], generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample from discrete uniform distribution.

        Args:
            size: Sampling shape (e.g., (batch_size, num_loc, components))
            generator: Description of generator.

        Returns:
            torch.Tensor: Sampled values in range [0.01, 1.0].
        """
        if generator is None:
            generator = torch.Generator()
        res = torch.randint(self.low, self.high, size, generator=generator)
        return res.float() / 100.0

    def _sample_array(self, size: Tuple[int, ...], rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample from discrete uniform distribution.

        Args:
            size: Sampling shape (e.g., (batch_size, num_loc, components))
            rng: Optional numpy default_rng for reproducibility.

        Returns:
            np.ndarray: Sampled values in range [1, 100].
        """
        if rng is None:
            rng = np.random.default_rng()
        return rng.integers(self.low, self.high, size=size).astype(float) / 100.0
