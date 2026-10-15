"""
Statistical sampling distributions - Constant.

Attributes:
    Constant(value: Union[float, torch.Tensor] = 1.0):
        Constant value sampling.

Example:
    constant = Constant()
    constant.sample(torch.Size((10, 50)))
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch

from .base import BaseDistribution


class Constant(BaseDistribution):
    """Constant value sampling.

    Attributes:
        value (Union[float, torch.Tensor]): Constant value to return.
    """

    def __init__(self, value: Union[float, torch.Tensor] = 1.0):
        """Initialize Class.

        Args:
            value: Constant value to return.
        """
        self.value = value

    def _sample_tensor(self, size: Tuple[int, ...], generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Return constant tensor.

        Args:
            size: Sampling shape (e.g., (batch_size, num_loc, components))
            generator: Description of generator.

        Returns:
            torch.Tensor: Constant values
        """
        if isinstance(self.value, torch.Tensor):
            return self.value.expand(size)
        return torch.full(size, float(self.value))

    def _sample_array(self, size: Tuple[int, ...], rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Return constant array.

        Args:
            size: Sampling shape (e.g., (batch_size, num_loc, components))
            rng: Optional numpy default_rng (ignored, for interface consistency).

        Returns:
            np.ndarray: Constant values
        """
        return np.full(size, self.value)
