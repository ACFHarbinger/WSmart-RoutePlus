"""spatial_mixed.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

"""

from typing import Optional, Tuple

import numpy as np
import torch

from .base import BaseDistribution


class Mixed(BaseDistribution):
    """50% uniform + 50% Gaussian clusters.

    Attributes:
        n_cluster_mix: Number of clusters to mix.
        lower: Lower bound for cluster centers.
        upper: Upper bound for cluster centers.
        std: Standard deviation for Gaussian clusters.
    """

    def __init__(self, n_cluster_mix: int = 1):
        """Initialize Class.

        Args:
            n_cluster_mix: Description of n_cluster_mix.
        """
        self.n_cluster_mix = n_cluster_mix
        self.lower, self.upper = 0.2, 0.8
        self.std = 0.07

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

        center = self.lower + (self.upper - self.lower) * torch.rand(
            batch_size, self.n_cluster_mix * 2, generator=generator
        )

        coords = torch.FloatTensor(batch_size, num_loc, 2).uniform_(0, 1)
        mutate_idx = torch.stack(
            [torch.randperm(num_loc, generator=generator)[: num_loc // 2] for _ in range(batch_size)]
        )

        segment_size = num_loc // (2 * self.n_cluster_mix)
        remaining = num_loc // 2 - segment_size * (self.n_cluster_mix - 1)
        sizes = [segment_size] * (self.n_cluster_mix - 1) + [remaining]

        for i in range(self.n_cluster_mix):
            indices = mutate_idx[:, sum(sizes[:i]) : sum(sizes[: i + 1])]

            means_x = center[:, 2 * i].unsqueeze(1).expand(-1, sizes[i])
            means_y = center[:, 2 * i + 1].unsqueeze(1).expand(-1, sizes[i])

            new_points = torch.stack(
                [
                    torch.normal(means_x, self.std, generator=generator),
                    torch.normal(means_y, self.std, generator=generator),
                ],
                dim=2,
            )

            coords.scatter_(
                1,
                indices.unsqueeze(-1).expand(-1, -1, 2),
                new_points,
            )

        return coords.clamp_(0, 1)

    def _sample_array(self, size: Tuple[int, ...], rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample from 50% uniform + 50% Gaussian clusters.

        Args:
            size: Tuple of size (batch_size, num_loc, 2).
            rng: Random number generator.

        Returns:
            np.ndarray: Sampled values.
        """
        if rng is None:
            rng = np.random.default_rng()

        batch_size, num_loc, _ = size

        # 1. Generate cluster centers
        center = self.lower + (self.upper - self.lower) * rng.random(size=(batch_size, self.n_cluster_mix * 2))

        # 2. Initialize coords with uniform distribution [0, 1)
        coords = rng.random(size=(batch_size, num_loc, 2))

        # 3. Create mutation indices (equivalent to torch.randperm)
        # We generate a permutation for each batch and take the first half
        mutate_idx = np.array([rng.permutation(num_loc)[: num_loc // 2] for _ in range(batch_size)])

        # 4. Calculate segment sizes for clusters
        segment_size = num_loc // (2 * self.n_cluster_mix)
        remaining = num_loc // 2 - segment_size * (self.n_cluster_mix - 1)
        sizes = [segment_size] * (self.n_cluster_mix - 1) + [remaining]

        # 5. Apply Gaussian clusters to the selected indices
        for i in range(self.n_cluster_mix):
            start_idx = sum(sizes[:i])
            end_idx = sum(sizes[: i + 1])

            # Extract the specific indices for this cluster segment
            current_indices = mutate_idx[:, start_idx:end_idx]  # (batch_size, segment)

            # Prepare means
            means_x = center[:, 2 * i, np.newaxis]
            means_y = center[:, 2 * i + 1, np.newaxis]

            # Sample from Normal distribution
            # Size matches (batch_size, segment)
            new_x = rng.normal(means_x, self.std, size=(batch_size, sizes[i]))
            new_y = rng.normal(means_y, self.std, size=(batch_size, sizes[i]))

            # 6. Advanced Indexing (NumPy's scatter_ equivalent)
            # Create batch indices to align with current_indices
            batch_indices = np.arange(batch_size)[:, np.newaxis]

            coords[batch_indices, current_indices, 0] = new_x
            coords[batch_indices, current_indices, 1] = new_y

        return np.clip(coords, 0, 1)
