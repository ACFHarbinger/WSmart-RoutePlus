"""Non-autoregressive Policy module.

This module provides the `NonAutoregressivePolicy` base class, which decouples
global graph feature prediction (heatmap generation) from solution
construction (decoding from heatmaps).

Attributes:
    NonAutoregressivePolicy: Base class for non-autoregressive policies.

Example:
    >>> encoder = NonAutoregressiveEncoder(embed_dim=128)
    >>> decoder = NonAutoregressiveDecoder(embed_dim=128)
    >>> policy = NonAutoregressivePolicy(encoder, decoder, env_name="tsp")
    >>> out = policy(td)
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Optional, Tuple

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase

from .decoder import NonAutoregressiveDecoder
from .encoder import NonAutoregressiveEncoder


class NonAutoregressivePolicy(nn.Module, ABC):
    """Base class for non-autoregressive policies.

    NAR policies predict graph feature matrices (heatmaps) in a single step using
    an encoder, then utilize a decoder to construct a valid tour from these
    features using algorithms such as greedy search, sampling, or ACO.

    Attributes:
        encoder: Global heatmap predictor.
        decoder: Solution constructor.
        env_name: Name of the target environment.
        embed_dim: Feature vector dimensionality.
    """

    def __init__(
        self,
        encoder: Optional[NonAutoregressiveEncoder] = None,
        decoder: Optional[NonAutoregressiveDecoder] = None,
        env_name: Optional[str] = None,
        embed_dim: int = 128,
        **kwargs: Any,
    ) -> None:
        """Initialize the NonAutoregressivePolicy.

        Args:
            encoder: Encoder module for global heatmap prediction.
            decoder: Decoder module for solution construction.
            env_name: Optional environment identifier.
            embed_dim: Dimensionality of latent embeddings.
            kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.env_name = env_name
        self.embed_dim = embed_dim

    def forward(
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        strategy: str = "sampling",
        num_starts: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Perform a full forward pass: heatmap prediction followed by decoding.

        Args:
            td: TensorDict containing problem instance data.
            env: Environment managing problem logic and rules.
            strategy: Decoding strategy identifier (e.g., "greedy", "sampling").
            num_starts: Number of parallel construction starts.
            kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Results dictionary containing:
                - reward (torch.Tensor): Calculated reward/cost for the solution.
                - log_likelihood (torch.Tensor): Log prob of chosen actions.
                - actions (torch.Tensor): Selected node indices.
                - heatmap (torch.Tensor): The predicted graph features used.
        """
        # Encode: predict heatmap
        encoder_out = self.encoder(td, **kwargs) if self.encoder is not None else None
        heatmap = encoder_out[0] if isinstance(encoder_out, tuple) else encoder_out

        # Decode: construct solution(s) from heatmap
        if self.decoder is not None and heatmap is not None:
            out = self.decoder(td, heatmap, env, strategy=strategy, num_starts=num_starts, **kwargs)
        else:
            # Fallback for subclasses that override forward entirely
            out = {}

        out["heatmap"] = heatmap
        return out

    def set_strategy(self, strategy: str, **kwargs: Any) -> None:
        """Set the decoding strategy for subsequent inference calls.

        Useful for aligning with evaluation pipelines that vary decoding
        parameters (e.g., temperature, beam width).

        Args:
            strategy: Strategy name (e.g., "beam_search", "greedy").
            kwargs: Additional strategy-specific parameters (e.g., temperature).
        """
        self._strategy = strategy
        for k, v in kwargs.items():
            setattr(self, f"_{k}", v)

    def common_decoding(
        self,
        strategy: str,
        td: TensorDict,
        env: RL4COEnvBase,
        heatmap: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        **decoding_kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict, RL4COEnvBase]:
        """Provide a standardized construction loop for NAR models using heatmaps.

        This utility manages strategy instantiation, pre/post-decoding hooks,
        and the iterative selection loop driven by the pre-computed heatmap.

        Args:
            strategy: Name of the decoding strategy.
            td: TensorDict with current environment state.
            env: Environment providing step logic.
            heatmap: Pre-computed edge probability matrix.
            actions: Optional pre-defined actions for evaluation.
            decoding_kwargs: Strategy-specific parameters.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, TensorDict, RL4COEnvBase]:
                - logprobs: Cumulative log probabilities.
                - actions: Final solution action sequence.
                - td: Resulting environment state TensorDict.
                - env: Environment object post-construction.

        Raises:
            ValueError: If a decoder has not been defined for the policy.
        """
        if actions is not None:
            strategy = "evaluate"
            decoding_kwargs["actions"] = actions

        from logic.src.utils.decoding import batchify, get_decoding_strategy

        strategy_obj = get_decoding_strategy(strategy, **decoding_kwargs)

        # Pre-decoding hook
        td, env, num_starts_hook = strategy_obj.pre_decoder_hook(td, env)

        # Determine num_starts
        num_starts = decoding_kwargs.get("num_starts", decoding_kwargs.get("num_samples", num_starts_hook))
        if num_starts is None:
            num_starts = 1
        num_starts = int(num_starts)

        # Update heatmap and td to match num_starts if needed
        if num_starts > 1:
            if td.size(0) != heatmap.size(0) * num_starts and td.size(0) == heatmap.size(0):
                # This might happen if pre_decoder_hook didn't batchify
                td = batchify(td, num_starts)

            if heatmap.size(0) != td.size(0):
                heatmap = batchify(heatmap, num_starts)

        if self.decoder is None:
            raise ValueError("common_decoding requires a decoder to be set")

        # Main decoding loop
        step = 0
        actions_list = []
        log_probs_list = []
        while not td["done"].all():
            # In NAR models, the decoder uses the pre-computed heatmap
            logits, mask = self.decoder(td, heatmap, env)
            action, log_prob, _ = strategy_obj.step(logits, mask, td)

            actions_list.append(action)
            log_probs_list.append(log_prob)

            td.set("action", action)
            td.update(env.step(td)["next"])
            step += 1

        actions_result = torch.stack(actions_list, dim=1)
        logprobs = torch.stack(log_probs_list, dim=1)

        # Post-decoding hook
        logprobs, actions_result, td, env = strategy_obj.post_decoder_hook(td, env, logprobs, actions_result)
        return logprobs, actions_result, td, env

    def eval(self) -> NonAutoregressivePolicy:
        """Set the model to evaluation mode.

        Returns:
            NonAutoregressivePolicy: Self, in evaluation mode.
        """
        super().eval()
        return self
