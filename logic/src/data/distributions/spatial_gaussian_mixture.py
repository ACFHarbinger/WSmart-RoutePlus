"""spatial_gaussian_mixture.py module.

Attributes:
    GaussianMixture: Configurable Gaussian Mixture Model.

Example:
    >>> from logic.src.data.distributions import GaussianMixture
    >>> gaussian_mixture = GaussianMixture(num_modes=3)
    >>> print(gaussian_mixture.sample())
"""

from typing import Optional, Tuple, Union, cast

import numpy as np
import torch

from .base import BaseDistribution


class GaussianMixture(BaseDistribution):
    """Configurable Gaussian Mixture Model.

    Attributes:
        num_modes: Number of modes in the Gaussian mixture.
        cdist: Scale of the Gaussian distribution.
    """

    def __init__(self, num_modes: int = 0, cdist: int = 0):
        """Initialize Class.

        Args:
            num_modes: Description of num_modes.
            cdist: Description of cdist.
        """
        self.num_modes = num_modes
        self.cdist = cdist

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
        if self.num_modes == 0:
            return torch.rand((batch_size, num_loc, 2), generator=generator)
        elif self.num_modes == 1 and self.cdist == 1:
            return self._generate_gaussian(batch_size, num_loc, generator=generator)
        else:
            return torch.stack(
                [self._generate_gaussian_mixture(num_loc, generator=generator) for _ in range(batch_size)]
            )

    def _sample_array(self, size: Tuple[int, ...], rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """NumPy version of the Gaussian/Mixture spatial sampler.

        Args:
            size: Description of size.
            rng: Description of rng.

        Returns:
            Description of return value.
        """
        if rng is None:
            rng = np.random.default_rng()

        batch_size, num_loc, _ = size

        # Case 1: Simple Uniform Distribution
        if self.num_modes == 0:
            return rng.random(size=(batch_size, num_loc, 2))

        # Case 2: Single Mode Gaussian (Vectorized)
        elif self.num_modes == 1 and self.cdist == 1:
            # Ensure _generate_gaussian_array is implemented using rng
            return self._generate_gaussian_array(batch_size, num_loc, rng=rng)

        # Case 3: Gaussian Mixture (Iterative/Batch)
        else:
            # We use a list comprehension and stack, similar to the torch.stack approach
            # Note: If _generate_gaussian_mixture can be vectorized for batches,
            # that would be more efficient than this loop.
            samples = [self._generate_gaussian_mixture_array(num_loc, rng=rng) for _ in range(batch_size)]
        return np.stack(samples)

    def _generate_gaussian_mixture(self, num_loc: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """generate gaussian mixture.

        Args:
            num_loc: Description of num_loc.
            generator: Description of generator.

        Returns:
            Any: Description of return value.
        """
        nums = torch.multinomial(
            input=torch.ones(self.num_modes) / self.num_modes,
            num_samples=num_loc,
            replacement=True,
            generator=generator,
        )

        coords = torch.empty((0, 2))

        for i in range(self.num_modes):
            num = int((nums == i).sum().item())
            if num > 0:
                center = torch.rand((1, 2), generator=generator) * self.cdist
                cov = torch.eye(2)
                nxy = torch.distributions.MultivariateNormal(center.squeeze(), covariance_matrix=cov).sample(
                    torch.Size([num])
                )
                coords = torch.cat((coords, nxy), dim=0)

        return self._global_min_max_scaling(coords)  # type: ignore[return-value]

    def _generate_gaussian(
        self, batch_size: int, num_loc: int, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """generate gaussian.

        Args:
            batch_size: Description of batch_size.
            num_loc: Description of num_loc.
            generator: Description of generator.

        Returns:
            Any: Description of return value.
        """
        mean = torch.full((batch_size, num_loc, 2), 0.5)
        covs = torch.rand(batch_size, generator=generator)

        coords = torch.zeros((batch_size, num_loc, 2))
        for i in range(batch_size):
            cov_matrix = torch.tensor([[1.0, covs[i]], [covs[i], 1.0]])
            m = torch.distributions.MultivariateNormal(mean[i], covariance_matrix=cov_matrix)
            coords[i] = m.sample()

        indices = torch.randperm(coords.size(0), generator=generator)
        coords = coords[indices]

        return self._batch_normalize_and_center(coords)  # type: ignore[return-value]

    def _generate_gaussian_mixture_array(self, num_loc: int, rng: np.random.Generator) -> np.ndarray:
        """Generate a mixture of Gaussian distributions.

        Args:
            num_loc: Number of locations.
            rng: Random number generator.

        Returns:
            np.ndarray: Sampled values.
        """
        probs = np.ones(self.num_modes) / self.num_modes
        nums = rng.choice(self.num_modes, size=num_loc, p=probs)

        coords_list = []

        for i in range(self.num_modes):
            num = np.sum(nums == i)
            if num > 0:
                # Random center scaled by cdist
                center = rng.random(2) * self.cdist
                cov = np.eye(2)

                # Sample from Multivariate Normal
                nxy = rng.multivariate_normal(mean=center, cov=cov, size=num)
                coords_list.append(nxy)

        # Concatenate all sampled clusters
        coords = np.vstack(coords_list)

        return self._global_min_max_scaling(coords)  # type: ignore[return-value]

    def _generate_gaussian_array(self, batch_size: int, num_loc: int, rng: np.random.Generator) -> np.ndarray:
        """Generate a single Gaussian distribution.

        Args:
            batch_size: Batch size.
            num_loc: Number of locations.
            rng: Random number generator.

        Returns:
            np.ndarray: Sampled values.
        """
        coords = np.zeros((batch_size, num_loc, 2))
        cov_values = rng.random(batch_size)

        # In NumPy, we still iterate over the batch for Multivariate Normal
        # as its built-in multivariate_normal doesn't broadcast different cov matrices easily
        for i in range(batch_size):
            mean = np.array([0.5, 0.5])
            # Construct cov matrix: [[1.0, rho], [rho, 1.0]]
            cov_matrix = np.array([[1.0, cov_values[i]], [cov_values[i], 1.0]])

            # Sample directly into the coordinate slice
            coords[i] = rng.multivariate_normal(mean=mean, cov=cov_matrix, size=num_loc)

        # Shuffling the batch (equivalent to randperm)
        indices = rng.permutation(batch_size)
        coords = coords[indices]

        return self._batch_normalize_and_center(coords)  # type: ignore[return-value]

    def _global_min_max_scaling(self, coords: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """global min max scaling.

        Args:
            coords: Description of coords.

        Returns:
            Any: Description of return value.
        """
        if isinstance(coords, torch.Tensor):
            coords_min = torch.amin(coords, dim=0, keepdim=True)
            coords_max = torch.amax(coords, dim=0, keepdim=True)
        else:
            coords_min = coords.min(axis=0, keepdims=True)
            coords_max = coords.max(axis=0, keepdims=True)

        coords = (coords - coords_min) / (coords_max - coords_min + 1e-8)
        return coords

    def _batch_normalize_and_center(self, coords: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """batch normalize and center.

        Args:
            coords: Description of coords.

        Returns:
            Any: Description of return value.
        """
        if isinstance(coords, torch.Tensor):
            coords_min = torch.amin(coords, dim=1, keepdim=True)
            coords_max = torch.amax(coords, dim=1, keepdim=True)
        else:
            coords_min = coords.min(axis=1, keepdims=True)
            coords_max = coords.max(axis=1, keepdims=True)

        coords = coords - coords_min
        if isinstance(coords, torch.Tensor):
            range_max = torch.amax(coords_max - coords_min, dim=-1, keepdim=True)
        else:
            diff = cast(np.ndarray, coords_max - coords_min)
            range_max = diff.max(axis=-1, keepdims=True)

        coords = coords / (range_max + 1e-8)

        if isinstance(coords, torch.Tensor):
            coords = coords + (1 - torch.amax(coords, dim=1, keepdim=True)) / 2
        else:
            arr_coords = cast(np.ndarray, coords)
            coords = coords + (1 - arr_coords.max(axis=1, keepdims=True)) / 2

        return coords
