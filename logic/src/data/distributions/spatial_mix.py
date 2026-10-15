"""spatial_mix.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

"""

from typing import Optional, Tuple

import numpy as np
import torch

from .base import BaseDistribution
from .spatial_cluster import Cluster
from .spatial_mixed import Mixed


class MixDistribution(BaseDistribution):
    """33/33/33 mix of Uniform/Cluster/Mixed.

    Attributes:
        n_cluster: Number of clusters.
        n_cluster_mix: Number of clusters in the mixed distribution.
    """

    def __init__(self, n_cluster: int = 3, n_cluster_mix: int = 1):
        """Initialize Class.

        Args:
            n_cluster: Description of n_cluster.
            n_cluster_mix: Description of n_cluster_mix.
        """
        self.Mixed = Mixed(n_cluster_mix=n_cluster_mix)
        self.Cluster = Cluster(n_cluster=n_cluster)

    def _sample_tensor(self, size: Tuple[int, ...], generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample.

        Args:
            size: Description of size.
            generator: Description of generator.

        Returns:
            Any: Description of return value.
        """
        if generator is None:
            generator = torch.Generator()

        batch_size, num_loc, _ = size
        coords = torch.FloatTensor(batch_size, num_loc, 2).uniform_(0, 1)

        p = torch.rand(batch_size, generator=generator)

        mask = p <= 0.33
        n_mixed = int(mask.sum().item())
        if n_mixed > 0:
            coords[mask] = self.Mixed._sample_tensor((n_mixed, num_loc, 2), generator=generator)

        mask = (p > 0.33) & (p <= 0.66)
        n_cluster = int(mask.sum().item())
        if n_cluster > 0:
            coords[mask] = self.Cluster._sample_tensor((n_cluster, num_loc, 2), generator=generator)

        return coords

    def _sample_array(self, size: Tuple[int, ...], rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample from a 33/33/33 mix of Uniform/Cluster/Mixed distributions.

        Args:
            size: Tuple of size (batch_size, num_loc, 2).
            rng: Random number generator.

        Returns:
            np.ndarray: Sampled values.
        """
        if rng is None:
            rng = np.random.default_rng()

        batch_size, num_loc, _ = size

        # 1. Initialize coordinates with uniform distribution [0, 1)
        coords = rng.random(size=(batch_size, num_loc, 2))

        # 2. Sample probabilities for the distribution mixture
        p = rng.random(size=(batch_size,))

        # 3. Handle 'Mixed' distribution (p <= 0.33)
        mask_mixed = p <= 0.33
        n_mixed = np.sum(mask_mixed)
        if n_mixed > 0:
            # Strategy Pattern: call the array version of the Mixed sampler
            coords[mask_mixed] = self.Mixed._sample_array((n_mixed, num_loc, 2), rng=rng)  # type: ignore[arg-type]

        # 4. Handle 'Cluster' distribution (0.33 < p <= 0.66)
        mask_cluster = (p > 0.33) & (p <= 0.66)
        n_cluster = np.sum(mask_cluster)
        if n_cluster > 0:
            # Strategy Pattern: call the array version of the Cluster sampler
            coords[mask_cluster] = self.Cluster._sample_array((n_cluster, num_loc, 2), rng=rng)  # type: ignore[arg-type]

        # Note: (p > 0.66) remains the initial uniform distribution (coords)
        return coords
