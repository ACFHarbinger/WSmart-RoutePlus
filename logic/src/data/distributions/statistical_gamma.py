"""
Statistical sampling distributions - Gamma.

Attributes:
    Gamma(alpha, theta, option):
        Gamma distribution sampling.

Example:
    gamma = Gamma()
    gamma.sample(torch.Size((10, 50)))
"""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from logic.src.constants.data import GAMMA_PRESETS

from .base import BaseDistribution


class Gamma(BaseDistribution):
    """Gamma distribution sampling with optional per-node preset parameters.

    Attributes:
        alpha (Union[float, torch.Tensor]): Shape parameter.
        theta (Union[float, torch.Tensor]): Scale parameter.
        option (Optional[int]): Preset index for per-node parameters.
    """

    def __init__(
        self,
        alpha: Union[float, torch.Tensor] = 2.0,
        theta: Union[float, torch.Tensor] = 2.0,
        option: Optional[int] = None,
    ):
        """Initialize Class.

        When ``option`` is provided, ``alpha`` and ``theta`` are ignored
        and per-node heterogeneous parameters are loaded from ``GAMMA_PRESETS``.

        Args:
            alpha: Shape parameter (ignored if ``option`` is set).
            theta: Scale parameter (ignored if ``option`` is set).
            option: Preset index (0-3) for per-node heterogeneous params.
        """
        self.option = option
        self.alpha = alpha
        self.theta = theta

    def _tile_param(self, size: int, param: List[int]) -> List[int]:
        """
        Tile a parameter list to match a target size.

        Args:
            size: Target size.
            param: Parameter list to tile.

        Returns:
            Tiled parameter list of length ``size``.
        """
        param_len = len(param)
        if size == param_len:
            return param

        tiled = param * math.ceil(size / param_len)
        return tiled[:size]

    def _sample_tensor(self, size: Tuple[int, ...], generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample from Gamma distribution.

        Args:
            size: Sampling shape (e.g., (batch_size, num_loc))
            generator: Description of generator.

        Returns:
            torch.Tensor: Sampled values
        """
        if generator is None:
            # The global RNG: a fresh ``torch.Generator()`` has the same fixed seed on every call.
            generator = torch.default_generator
        if self.option is not None:
            # Use per-node heterogeneous params; fall back to numpy then convert
            # Draw a fresh seed per call: seeding from initial_seed() repeated the same sample,
            # so every day of a multi-day dataset held identical waste.
            seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator, device=generator.device).item())
            rng = np.random.default_rng(seed)
            return torch.from_numpy(self._sample_array(size, rng=rng)).float()

        m = torch.distributions.Gamma(self.alpha, 1 / self.theta)
        return m.sample(torch.Size(size)) / 100.0

    def _sample_array(self, size: Tuple[int, ...], rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample from Gamma distribution.

        When ``option`` is set, uses per-node heterogeneous alpha/theta
        parameters tiled to match ``problem_size`` (the last dimension of ``size``).

        Args:
            size: Sampling shape (e.g., (batch_size, num_loc))
            rng: Optional numpy Generator for reproducibility.

        Returns:
            np.ndarray: Sampled values
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.option is not None:
            alpha_pattern, theta_pattern = GAMMA_PRESETS[self.option]
            problem_size = size[-1]
            k = self._tile_param(problem_size, alpha_pattern)
            th = self._tile_param(problem_size, theta_pattern)
            return rng.gamma(k, th, size=size) / 100.0

        return rng.gamma(self.alpha, self.theta, size=size) / 100.0
