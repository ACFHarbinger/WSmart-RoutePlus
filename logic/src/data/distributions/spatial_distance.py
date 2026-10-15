"""
Spatial sampling distributions - Distance-based.

Attributes:
    Distance: Distance-based sampling.

Example:
    >>> from logic.src.data.distributions import Distance
    >>> distance = Distance()
    >>> print(distance.sample())
"""

from typing import Any, Optional, Tuple, Union

import numpy as np
import torch

from .base import BaseDistribution


class Distance(BaseDistribution):
    """Distance-based sampling.

    Attributes:
        graph: Graph containing depot and locations.
    """

    def __init__(self, graph: Tuple[Any, Any]):
        """Initialize Class.

        Args:
            graph: (depot, loc) coordinates.
        """
        self.depot, self.loc = graph

    def _sample_array(
        self, size: Tuple[int, ...], rng: Optional[Union[torch.Generator, np.random.Generator]] = None
    ) -> np.ndarray:
        """Sample from distance-based distribution.

        Args:
            size: Description of size.
            rng: Description of rng.

        Returns:
            np.ndarray: Sampled values [num_loc] or [batch_size, num_loc]
        """
        if self.depot.ndim == 1:
            # Single instance: depot is [coords], loc is [num_loc, coords]
            # Reshape for broadcasting
            wp = np.linalg.norm(self.depot[None, :] - self.loc, axis=-1)
            # Clip or repeat to match size if necessary, but usually size matches [num_loc]
            res = (1 + (wp / wp.max() * 99).astype(int)) / 100.0
            return np.broadcast_to(res, size)
        else:
            # Batch: depot is [batch_size, coords], loc is [batch_size, num_loc, coords]
            wp = np.linalg.norm(self.depot[:, None, :] - self.loc, axis=-1)
            res = (1 + (wp / wp.max(axis=-1, keepdims=True) * 99).astype(int)) / 100.0
            return np.broadcast_to(res, size)

    def _sample_tensor(self, size: Tuple[int, ...], generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """PyTorch version of distance-based sampling.

        Args:
            size: Description of size.
            generator: Description of generator.

        Returns:
            Description of return value.
        """
        # Ensure inputs are tensors
        depot = torch.as_tensor(self.depot).float()
        loc = torch.as_tensor(self.loc).float()
        if depot.ndim == 1:
            diff = depot[None, :] - loc
            wp = torch.linalg.vector_norm(diff, dim=-1)
            wp_max = wp.max()
            scaled = 1 + (wp / wp_max * 99).to(torch.int)
            res = scaled.float() / 100.0
            return res.expand(size)
        else:
            diff = depot[:, None, :] - loc
            wp = torch.linalg.vector_norm(diff, dim=-1)
            wp_max = wp.max(dim=-1, keepdim=True).values
            scaled = 1 + (wp / wp_max * 99).to(torch.int)
            res = scaled.float() / 100.0
            return res.expand(size)
