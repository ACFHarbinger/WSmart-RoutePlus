"""
Statistical sampling distributions - Bernoulli-Gamma Mixture.

Attributes:
    p (Union[float, torch.Tensor]): Probability of a non-zero (Gamma-distributed) event.
    alpha (Union[float, torch.Tensor]): Gamma shape parameter.
    theta (Union[float, torch.Tensor]): Gamma scale parameter.

Example:
    bgm = BernoulliGammaMixture()
    bgm.sample(torch.Size((10, 50)))
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch

from .base import BaseDistribution


class BernoulliGammaMixture(BaseDistribution):
    """Bernoulli-Gamma Mixture distribution sampling.

    Attributes:
        p (Union[float, torch.Tensor]): Probability of a non-zero (Gamma-distributed) event.
        alpha (Union[float, torch.Tensor]): Gamma shape parameter.
        theta (Union[float, torch.Tensor]): Gamma scale parameter.
    """

    def __init__(
        self,
        p: Union[float, torch.Tensor] = 0.5,
        alpha: Union[float, torch.Tensor] = 2.0,
        theta: Union[float, torch.Tensor] = 2.0,
    ):
        """Initialize Class.

        A Bernoulli-Gamma mixture models an event that occurs with probability p,
        and when it occurs, its magnitude is drawn from a Gamma distribution.

        Args:
            p: Probability of a non-zero (Gamma-distributed) event.
            alpha: Gamma shape parameter.
            theta: Gamma scale parameter.
        """
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.theta = theta

    def _sample_tensor(self, size: Tuple[int, ...], generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample from Bernoulli-Gamma Mixture distribution.

        Args:
            size: Sampling shape (e.g., (batch_size, num_loc))
            generator: Optional torch.Generator for reproducibility.

        Returns:
            torch.Tensor: Sampled values
        """
        if generator is None:
            generator = torch.Generator()

        p_tensor = torch.as_tensor(self.p, dtype=torch.float32)

        # Sample N ~ Bernoulli(p)
        bernoulli_dist = torch.distributions.Bernoulli(probs=p_tensor)
        N = bernoulli_dist.sample(torch.Size(size))

        mask = N > 0

        alpha_tensor = torch.as_tensor(self.alpha, dtype=torch.float32)
        theta_tensor = torch.as_tensor(self.theta, dtype=torch.float32)

        gamma_shape = torch.ones_like(N) * alpha_tensor
        gamma_rate = torch.ones_like(N) * (1.0 / theta_tensor)

        gamma_dist = torch.distributions.Gamma(gamma_shape, gamma_rate)

        # gamma_dist's batch_shape is already at least matching N's shape
        Y_sampled = gamma_dist.sample()

        return torch.where(mask, Y_sampled, torch.zeros_like(Y_sampled)) / 100.0

    def _sample_array(self, size: Tuple[int, ...], rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample from Bernoulli-Gamma Mixture distribution.

        Args:
            size: Sampling shape (e.g., (batch_size, num_loc))
            rng: Optional numpy default_rng for reproducibility.

        Returns:
            np.ndarray: Sampled values
        """
        if rng is None:
            rng = np.random.default_rng()

        N = rng.binomial(n=1, p=self.p, size=size)

        mask = N > 0

        # Create shape array to ensure gamma sample has expected size even with scalar params
        gamma_shape = np.ones(size) * self.alpha

        # If gamma_shape has the requested size, the output will match it.
        Y_sampled = rng.gamma(shape=gamma_shape, scale=self.theta)

        return np.where(mask, Y_sampled, 0.0) / 100.0
