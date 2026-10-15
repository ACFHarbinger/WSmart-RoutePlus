"""
Statistical sampling distributions - Compound Poisson-Gamma.

Attributes:
    CompoundPoissonGamma(lam, alpha, theta):
        Compound Poisson-Gamma distribution sampling.

Example:
    cpg = CompoundPoissonGamma()
    cpg.sample(torch.Size((10, 50)))
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch

from .base import BaseDistribution


class CompoundPoissonGamma(BaseDistribution):
    """Compound Poisson-Gamma distribution sampling.

    Attributes:
        lam (Union[float, torch.Tensor]): Poisson rate parameter (expected number of events).
        alpha (Union[float, torch.Tensor]): Gamma shape parameter for each event.
        theta (Union[float, torch.Tensor]): Gamma scale parameter for each event.
    """

    def __init__(
        self,
        lam: Union[float, torch.Tensor] = 1.0,
        alpha: Union[float, torch.Tensor] = 2.0,
        theta: Union[float, torch.Tensor] = 2.0,
    ):
        """Initialize Class.

        A compound Poisson-Gamma distribution represents the sum of N independent Gamma
        random variables, where N is Poisson distributed.

        Args:
            lam: Poisson rate parameter (expected number of events).
            alpha: Gamma shape parameter for each event.
            theta: Gamma scale parameter for each event.
        """
        super().__init__()
        self.lam = lam
        self.alpha = alpha
        self.theta = theta

    def _sample_tensor(self, size: Tuple[int, ...], generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample from Compound Poisson-Gamma distribution.

        Args:
            size: Sampling shape (e.g., (batch_size, num_loc))
            generator: Optional torch.Generator for reproducibility.

        Returns:
            torch.Tensor: Sampled values
        """
        if generator is None:
            generator = torch.Generator()

        lam_tensor = torch.as_tensor(self.lam, dtype=torch.float32)

        # Sample N ~ Poisson(lam)
        poisson_dist = torch.distributions.Poisson(lam_tensor)
        # N will have shape `size` if lam_tensor is scalar.
        N = poisson_dist.sample(torch.Size(size))

        mask = N > 0
        safe_N = torch.where(mask, N, torch.ones_like(N))

        alpha_tensor = torch.as_tensor(self.alpha, dtype=torch.float32)
        theta_tensor = torch.as_tensor(self.theta, dtype=torch.float32)

        gamma_shape = safe_N * alpha_tensor
        gamma_rate = 1.0 / theta_tensor

        gamma_dist = torch.distributions.Gamma(gamma_shape, gamma_rate)

        # gamma_dist's batch_shape is already at least matching N's shape
        # so we don't pass torch.Size(size) again to avoid doubling dimensions
        Y_sampled = gamma_dist.sample()

        return torch.where(mask, Y_sampled, torch.zeros_like(Y_sampled)) / 100.0

    def _sample_array(self, size: Tuple[int, ...], rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample from Compound Poisson-Gamma distribution.

        Args:
            size: Sampling shape (e.g., (batch_size, num_loc))
            rng: Optional numpy default_rng for reproducibility.

        Returns:
            np.ndarray: Sampled values
        """
        if rng is None:
            rng = np.random.default_rng()

        N = rng.poisson(lam=self.lam, size=size)

        mask = N > 0
        safe_N = np.where(mask, N, 1.0)

        gamma_shape = safe_N * self.alpha

        # If gamma_shape has the requested size (via safe_N), the output will match it.
        Y_sampled = rng.gamma(shape=gamma_shape, scale=self.theta)

        return np.where(mask, Y_sampled, 0.0) / 100.0
