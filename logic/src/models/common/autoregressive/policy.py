"""Autoregressive Policy module.

This module provides the implementation of the AutoregressivePolicy, which
integrates an autoregressive encoder and decoder into a unified constructive
policy architecture.

Attributes:
    AutoregressivePolicy: Base class for autoregressive policies.

Example:
    >>> policy = AutoregressivePolicy()
    >>> reward, log_p, actions = policy(td, env)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict

from logic.src.envs.base.base import RL4COEnvBase

from .constructive import ConstructivePolicy
from .decoder import AutoregressiveDecoder
from .encoder import AutoregressiveEncoder


class AutoregressivePolicy(ConstructivePolicy):
    """Base class for autoregressive policies.

    Combines an AR encoder with an AR decoder to form a complete policy.
    Inherits from `ConstructivePolicy` to leverage standardized decoding
    strategies and simulation utilities.

    Attributes:
        encoder: Problem state encoder.
        decoder: Step-by-step action decoder.
    """

    def __init__(
        self,
        encoder: Optional[AutoregressiveEncoder] = None,
        decoder: Optional[AutoregressiveDecoder] = None,
        env_name: Optional[str] = None,
        embed_dim: int = 128,
        seed: int = 42,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        """Initialize the AutoregressivePolicy.

        Args:
            encoder: Optional problem state encoder.
            decoder: Optional step-by-step action decoder.
            env_name: Name of the RL4CO environment.
            embed_dim: Internal dimensionality for the latent embeddings.
            seed: Random seed for reproducibility.
            device: Computing device for the neural network.
            kwargs: Additional keyword arguments.
        """
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            embed_dim=embed_dim,
            seed=seed,
            device=device,
            **kwargs,
        )

    def forward(
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        strategy: str = "sampling",
        num_starts: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Full forward pass: problem encoding followed by solution decoding.

        Args:
            td: TensorDict containing the current problem/environment state.
            env: Environment object for reward calculation and masking.
            strategy: Decoding strategy ("sampling", "greedy", etc.).
            num_starts: Number of initial solutions to generate.
            kwargs: Additional keyword arguments for the encoder/decoder.

        Returns:
            Dict[str, Any]: Policy outputs including:
                - reward (torch.Tensor): Calculated reward for the full tours.
                - log_likelihood (torch.Tensor): Log probability of chosen actions.
                - actions (torch.Tensor): The sequence of selected node indices.

        Raises:
            ValueError: If a decoder has not been provided to the policy.
        """
        # Encode static/initial state
        embeddings = self.encoder(td, **kwargs) if self.encoder is not None else None

        # Decode actions sequentially
        if self.decoder is not None:
            # Note: Many decoders in WSmart-Route implement their own selection loop.
            # We assume the decoder handles the entire AR process.
            log_p, actions = self.decoder(td, embeddings, env, strategy=strategy, num_starts=num_starts, **kwargs)
        else:
            raise ValueError("AutoregressivePolicy requires a decoder.")

        # Calculate reward/cost for the final solution
        reward = env.get_reward(td, actions) if env is not None else torch.zeros(td.batch_size[0], device=td.device)

        return {
            "reward": reward,
            "log_likelihood": log_p,
            "actions": actions,
        }
