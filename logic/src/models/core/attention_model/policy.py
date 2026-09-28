"""Attention Model Policy for RL training.

This module provides an RL4CO-compatible wrapper around the GATEncoder and
AttentionDecoder components. It utilizes TensorDict for standardized state
management across different reinforcement learning environments.

Attributes:
    AttentionModelPolicy: A constructive policy combining GAT and attention decoding.

Example:
    >>> from logic.src.models.core.attention_model.policy import AttentionModelPolicy
    >>> policy = AttentionModelPolicy(env_name="vrp", embed_dim=128)
    >>> out = policy(td, env, strategy="greedy")
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
from tensordict import TensorDict

from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.data.processor.transforms import batchify
from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.autoregressive.policy import AutoregressivePolicy
from logic.src.models.subnets.decoders.glimpse.decoder import GlimpseDecoder
from logic.src.models.subnets.embeddings import get_init_embedding
from logic.src.models.subnets.encoders.gat import GraphAttentionEncoder
from logic.src.utils.data.td_state_wrapper import TensorDictStateWrapper
from logic.src.utils.tasks.dummy_problem import DummyProblem


class AttentionModelPolicy(AutoregressivePolicy):
    """RL4CO-style Policy for constructive routing.

    Leverages a Graph Attention Network (GAT) to encode nodes and a Glimpse-based
    decoder to autoregressively select the next stop in a route.

    Attributes:
        init_embedding (nn.Module): Problem-specific initial feature projection.
        encoder (GraphAttentionEncoder): Graph-level feature extractor.
        decoder (GlimpseDecoder): Step-wise decision maker with attention glimpse.
        env_name (str): Persistent identifier for specific routing domains.
        embed_dim (int): feature vector dimensionality.
    """

    encoder: GraphAttentionEncoder
    decoder: GlimpseDecoder

    def __init__(
        self,
        env_name: str,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        n_encode_layers: int = 3,
        n_heads: int = 8,
        normalization: str = "batch",
        **kwargs: Any,
    ) -> None:
        """Initializes the AttentionModelPolicy.

        Args:
            env_name: Name of the environment identifier.
            embed_dim: Dimensionality of node features.
            hidden_dim: Dimensionality of hidden layers.
            n_encode_layers: Number of GAT encoder layers.
            n_heads: Number of attention heads.
            normalization: Type of normalization ("batch", "layer").
            kwargs: Additional keyword arguments.
        """
        super().__init__(env_name=env_name, embed_dim=embed_dim)

        self.init_embedding = get_init_embedding(env_name, embed_dim)

        if "norm_config" in kwargs:
            norm_config = kwargs.pop("norm_config")
        elif normalization is not None:
            norm_config = NormalizationConfig(norm_type=normalization)
        else:
            norm_config = None

        self.encoder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embed_dim,
            feed_forward_hidden=hidden_dim,
            n_layers=n_encode_layers,
            norm_config=norm_config,
            **kwargs,
        )

        self.decoder = GlimpseDecoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            problem=DummyProblem(env_name),
            n_heads=n_heads,
            **kwargs,
        )

    def forward(  # type: ignore[override]
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        strategy: str = "sampling",
        num_starts: int = 1,
        actions: Optional[torch.Tensor] = None,
        start_nodes: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Executes one-shot or multi-start constructive decoding.

        Args:
            td: TensorDict containing problem instance data.
            env: Environment managing problem physics.
            strategy: Decoding strategy identifier (e.g., "greedy").
            num_starts: Number of parallel construction starts.
            actions: Optional pre-selected actions.
            start_nodes: Optional forced first-action indices of shape
                ``[batch * num_starts]``.  When provided, step 0 uses these
                node indices as the action (log-prob is still computed from
                the decoder logits so gradients flow correctly).  Subsequent
                steps decode normally.  Ignored when ``actions`` is given.
            kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Policy results including:
                - reward (torch.Tensor): Calculated reward for the routes.
                - log_likelihood (torch.Tensor): Cumulative log prob of actions.
                - actions (torch.Tensor): Selected node indices.
                - entropy (torch.Tensor): Total policy entropy across steps.
                - td (TensorDict): Final environment state after decoding.

        Raises:
            RuntimeError: If encoder or decoder subnets are uninitialized.
            ValueError: If `env_name` is missing from the state.
        """
        # 1. Initialize embeddings
        init_embeds = self.init_embedding(td)

        # 2. Encoder
        edges = td.get("edges", None)
        if self.encoder is None:
            raise RuntimeError("Encoder is not initialized")
        embeddings = self.encoder(init_embeds, edges)

        # 3. Multi-start expansion if needed
        if num_starts > 1:
            td = batchify(td, num_starts)
            embeddings = embeddings.unsqueeze(1).repeat(1, num_starts, 1, 1).reshape(-1, *embeddings.shape[1:])

        # 4. Decoder Precomputation
        if self.decoder is None:
            raise RuntimeError("Decoder is not initialized")
        fixed = self.decoder._precompute(embeddings)

        # 5. Decoding Loop
        log_likelihood: Union[int, float, torch.Tensor] = 0
        entropy: Union[int, float, torch.Tensor] = 0
        output_actions = []
        step_idx = 0

        # Assuming environment is already reset
        if hasattr(env, "batch_size"):
            env.batch_size = td.batch_size

        while not td["done"].all():
            # Wrap state for legacy compatibility
            if self.env_name is None:
                raise ValueError("env_name must be set")
            state_wrapper = TensorDictStateWrapper(td, self.env_name)

            # Get logits from decoder (mask=True for invalid)
            logits, mask = self.decoder._get_log_p(fixed, state_wrapper)

            # Flatten head dimension if present (e.g. DeepGATDecoder)
            if logits.dim() == 3:
                logits = logits[:, 0, :] if logits.size(1) > 1 else logits.squeeze(1)

            # Invert mask for _select_action (expects True=VALID)
            if mask.dim() == 3:
                mask = mask.squeeze(1)
            valid_mask = ~mask

            if actions is not None:
                # Teacher forcing
                action = actions[:, step_idx]
                # Compute log_prob of this action
                probs = torch.softmax(logits.masked_fill(~valid_mask, float("-inf")), dim=-1)
                log_p = torch.log(probs.gather(1, action.unsqueeze(-1)) + 1e-10).squeeze(-1)
            elif step_idx == 0 and start_nodes is not None:
                # Forced start node (POMO mandatory-only mode): pin first action to
                # the pre-selected mandatory node while still computing log_p from
                # the decoder distribution so that gradients flow correctly.
                action = start_nodes.to(td.device)
                probs = torch.softmax(logits.masked_fill(~valid_mask, float("-inf")), dim=-1)
                log_p = torch.log(probs.gather(1, action.unsqueeze(-1)) + 1e-10).squeeze(-1)
            else:
                # Select action
                action, log_p, entropy_step = self._select_action(logits, valid_mask, strategy)
                if isinstance(entropy_step, torch.Tensor):
                    entropy = entropy + entropy_step

            # Update state
            td["action"] = action
            td = env.step(td)["next"].clone()

            # update log likelihood
            log_likelihood = log_likelihood + log_p
            output_actions.append(action)
            step_idx += 1

        # Collect results
        if len(output_actions) > 0:
            actions_tensor = torch.stack(output_actions, dim=1)
        else:
            actions_tensor = torch.zeros((td.batch_size[0], 0), device=td.device, dtype=torch.long)

        reward = env.get_reward(td, actions_tensor)

        out = {
            "reward": reward,
            "log_likelihood": log_likelihood,
            "actions": actions_tensor,
            "entropy": entropy,
            "td": td,
        }

        if kwargs.get("return_init_embeds", False):
            out["init_embeds"] = init_embeds

        return out
