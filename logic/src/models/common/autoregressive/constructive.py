"""Base constructive policy modules.

This module provides the foundation for constructive routing policies, which
build solutions by selecting nodes sequentially.

Attributes:
    ConstructivePolicy: Base class for constructive policies.

Example:
    None
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase


class ConstructivePolicy(nn.Module, ABC):
    """Base class for constructive (autoregressive) policies.

    Constructive policies build solutions step-by-step by selecting
    one node at a time until the solution is complete.

    Attributes:
        encoder: Neural encoder for problem state.
        decoder: Neural decoder for action selection.
        env_name: Name of the environment.
        embed_dim: Dimensionality of embeddings.
        seed: Random seed for reproducibility.
        device: Device on which the policy resides.
        generator: Local random number generator for sampling.
        rng: Standard library random generator.
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        env_name: Optional[str] = None,
        embed_dim: int = 128,
        seed: int = 42,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        """Initialize the ConstructivePolicy.



        Args:
            encoder: Encoder instance for state processing.
            decoder: Decoder instance for step-by-step selection.
            env_name: Name of the environment associated with this policy.
            embed_dim: Feature dimensionality for latent representations.
            seed: Seed for random initialization and sampling.
            device: Computing device ('cpu', 'cuda').
            kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.env_name = env_name
        self.embed_dim = embed_dim
        self.seed = seed
        self.device = torch.device(device)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.rng = random.Random(seed) if seed is not None else random.Random()

    def __getstate__(self) -> Dict[str, Any]:
        """Prepare the policy state for pickling.

        Custom handler that serializes the torch.Generator state correctly
        since it is not directly picklable.

        Returns:
            Dict[str, Any]: A serializable dictionary of the policy's state.
        """
        state = self.__dict__.copy()
        state["generator_state"] = self.generator.get_state()
        state["generator_device"] = str(self.generator.device)
        del state["generator"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore the policy state from a pickled object.

        Restores the torch.Generator on the original device with its
        saved internal state.

        Args:
            state: Serialized dictionary of the policy's attributes.
        """
        gen_state = state.pop("generator_state")
        gen_device = state.pop("generator_device")
        self.__dict__.update(state)
        self.generator = torch.Generator(device=gen_device)
        self.generator.set_state(gen_state)

    @abstractmethod
    def forward(
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        strategy: str = "sampling",
        num_starts: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute a full forward pass: encoding followed by sequential decoding.



        Args:
            td: TensorDict containing the problem instance metadata.
            env: Environment object for state transitions and masking.
            strategy: Decoding strategy ("sampling", "greedy", "beam_search").
            num_starts: Number of parallel solution attempts.
            kwargs: Additional control arguments for decoding.

        Returns:
            Dict[str, Any]: Results dictionary containing:
                - reward (torch.Tensor): Final reward/cost for the constructed tours.
                - log_likelihood (torch.Tensor): Log probability of the selections.
                - actions (torch.Tensor): The sequence of visited node indices.
        """
        raise NotImplementedError

    def _select_action(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        strategy: str = "sampling",
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select an action based on logits and the specified strategy.

        Uses the `get_decoding_strategy` utility to apply logic like greedy
        selection, multinomial sampling, or beam search.



        Args:
            logits: Predicted action logits of shape [batch, num_nodes].
            mask: Valid action mask of shape [batch, num_nodes].
            strategy: Name of the decoding strategy to apply.
            kwargs: Additional parameters (e.g., temperature) for the strategy.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - action: Selected node index.
                - log_prob: Log probability of the selection.
                - entropy: Entropy of the action distribution.
        """
        # Get strategy (can be cached if needed)
        from logic.src.utils.decoding import get_decoding_strategy

        decoder_strategy = get_decoding_strategy(strategy, **kwargs)

        # Step
        result = decoder_strategy.step(logits, mask)
        if len(result) == 3:
            action, log_prob, entropy = result
        else:
            action, log_prob = result[:2]  # type: ignore[misc]
            entropy = result[2] if len(result) > 2 else torch.tensor(0.0)
        return action, log_prob, entropy
