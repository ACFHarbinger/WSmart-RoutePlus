"""spatial_cluster.py module.

Attributes:
    Cluster: Multiple Gaussian distributed clusters.

Example:
    >>> from logic.src.data.distributions import Cluster
    >>> cluster = Cluster(n_cluster=3)
    >>> print(cluster.sample())
"""

from typing import Optional, Tuple

import numpy as np
import torch

from .base import BaseDistribution


class Cluster(BaseDistribution):
    """Multiple Gaussian distributed clusters.

    Attributes:
        n_cluster: Number of clusters.
        lower: Lower bound for cluster centers.
        upper: Upper bound for cluster centers.
        std: Standard deviation for Gaussian distribution.
    """

    def __init__(self, n_cluster: int = 3):
        """Initialize Class.

        Args:
            n_cluster: Description of n_cluster.
        """
        self.n_cluster = n_cluster
        self.lower, self.upper = 0.2, 0.8
        self.std = 0.07

    def _sample_tensor(self, size: Tuple[int, ...], generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample clustered locations.

        Args:
            size: Description of size.
            generator: Description of generator.

        Returns:
            torch.Tensor: Sampled values.
        """
        if generator is None:
            generator = torch.Generator()
        batch_size, num_loc, _ = size

        center = self.lower + (self.upper - self.lower) * torch.rand(
            batch_size, self.n_cluster * 2, generator=generator
        )

        coords = torch.zeros(batch_size, num_loc, 2)
        cluster_sizes = [num_loc // self.n_cluster] * self.n_cluster
        for i in range(num_loc % self.n_cluster):
            cluster_sizes[i] += 1

        current_index = 0
        for i in range(self.n_cluster):
            means = center[:, i * 2 : (i + 1) * 2]
            stds = torch.full((batch_size, 2), self.std)
            points = torch.normal(
                means.unsqueeze(1).expand(-1, cluster_sizes[i], -1),
                stds.unsqueeze(1).expand(-1, cluster_sizes[i], -1),
                generator=generator,
            )
            coords[:, current_index : (current_index + cluster_sizes[i]), :] = points
            current_index += cluster_sizes[i]

        return coords.clamp_(0, 1)

    def _sample_array(self, size: Tuple[int, ...], rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """NumPy version of clustered location sampling.

        Args:
            size: Description of size.
            rng: Description of rng.

        Returns:
            Description of return value.
        """
        if rng is None:
            rng = np.random.default_rng()

        batch_size, num_loc, _ = size

        # 1. Sample cluster centers uniformly within [lower, upper]
        # Shape: (batch_size, n_cluster * 2)
        center = self.lower + (self.upper - self.lower) * rng.random(size=(batch_size, self.n_cluster * 2))

        coords = np.zeros((batch_size, num_loc, 2))

        # 2. Distribute points among clusters (handle remainder)
        cluster_sizes = [num_loc // self.n_cluster] * self.n_cluster
        for i in range(num_loc % self.n_cluster):
            cluster_sizes[i] += 1

        current_index = 0
        for i in range(self.n_cluster):
            size_i = cluster_sizes[i]

            # 3. Extract centers for current cluster and expand for all points in it
            # Extract x, y for this cluster: (batch_size, 2)
            means = center[:, i * 2 : (i + 1) * 2]

            # Expand means to (batch_size, size_i, 2) to match point dimensions
            # NumPy's broadcasting handles this cleanly with np.newaxis
            means_expanded = means[:, np.newaxis, :]

            # 4. Sample Gaussian points
            # rng.normal can take arrays for loc and scale to sample in bulk
            points = rng.normal(loc=means_expanded, scale=self.std, size=(batch_size, size_i, 2))

            # 5. Assign to coordinate tensor
            coords[:, current_index : (current_index + size_i), :] = points
            current_index += size_i

        # Ensure points stay within the unit square [0, 1]
        return np.clip(coords, 0, 1)
