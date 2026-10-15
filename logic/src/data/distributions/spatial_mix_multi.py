"""spatial_mix_multi.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

"""

from typing import Optional, Tuple

import numpy as np
import torch

from .base import BaseDistribution
from .spatial_cluster import Cluster
from .spatial_gaussian_mixture import GaussianMixture
from .spatial_mixed import Mixed


class MixMultiDistributions(BaseDistribution):
    """Batch-wise sampling from multiple distribution variants.

    Attributes:
        distributions: List of distribution classes and their arguments.
    """

    def __init__(self):
        """Initialize Class.

        Args:
            None.
        """
        self.distributions = [
            (None, {}),
            (Cluster, {"n_cluster": 3}),
            (Mixed, {"n_cluster_mix": 1}),
            (GaussianMixture, {"num_modes": 0, "cdist": 0}),
            (GaussianMixture, {"num_modes": 1, "cdist": 1}),
            (Cluster, {"n_cluster": 4}),
            (Cluster, {"n_cluster": 5}),
            (Mixed, {"n_cluster_mix": 2}),
            (Mixed, {"n_cluster_mix": 3}),
            (Cluster, {"n_cluster": 6}),
            (Mixed, {"n_cluster_mix": 4}),
        ]

    def _sample_tensor(self, size: Tuple[int, ...], generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample.

        Args:
            size: Description of size.
            generator: Description of generator.

        Returns:
            torch.Tensor: Sampled values.
        """
        if generator is None:
            generator = torch.Generator()

        batch_size, num_loc, _ = size
        coords = torch.zeros(batch_size, num_loc, 2)
        dist_indices = torch.randint(0, len(self.distributions), (batch_size,), generator=generator)

        for i, (cls, kwargs) in enumerate(self.distributions):
            mask = dist_indices == i
            n_samples = int(mask.sum().item())
            if n_samples == 0:
                continue

            if cls is None:
                coords[mask] = torch.rand(n_samples, num_loc, 2, generator=generator)
            else:
                dist = cls(**kwargs)
                coords[mask] = dist._sample_tensor((n_samples, num_loc, 2), generator=generator)

        return coords.clamp_(0, 1)

    def _sample_array(self, size: Tuple[int, ...], rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample from multiple distributions.

        Args:
            size: Tuple of size (batch_size, num_loc, 2).
            rng: Random number generator.

        Returns:
            np.ndarray: Sampled values.
        """
        if rng is None:
            rng = np.random.default_rng()

        batch_size, num_loc, _ = size
        # Initialize as zeros
        coords = np.zeros((batch_size, num_loc, 2))

        # Sample distribution indices for each batch item
        dist_indices = rng.integers(0, len(self.distributions), size=batch_size)

        for i, (cls, kwargs) in enumerate(self.distributions):
            # Boolean mask for all batch items assigned to this distribution
            mask = dist_indices == i
            n_samples = np.sum(mask)

            if n_samples == 0:
                continue

            if cls is None:
                # Fallback to standard uniform [0, 1)
                coords[mask] = rng.random(size=(n_samples, num_loc, 2))
            else:
                # Instantiate the distribution class and call its array strategy
                dist = cls(**kwargs)
                # Ensure the class implements _sample_array
                coords[mask] = dist._sample_array((n_samples, num_loc, 2), rng=rng)

        return np.clip(coords, 0, 1)
