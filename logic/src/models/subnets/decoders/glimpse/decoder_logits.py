"""Logit and likelihood helpers for GlimpseDecoder.

Moved from ``logic.src.models.subnets.decoders.glimpse.decoder``.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

import torch
from torch import nn

from logic.src.models.subnets.decoders.common import AttentionDecoderCache
from logic.src.models.subnets.decoders.glimpse.attention import make_heads, one_to_many_logits


class GlimpseDecoderLogitsMixin:
    """Moved methods of :class:`GlimpseDecoder`."""

    project_node_embeddings: nn.Linear
    project_step_context: nn.Module
    context_embedding: Optional[nn.Module]
    n_heads: int
    tanh_clipping: float
    embed_dim: int

    def _precompute(self, embeddings: torch.Tensor, num_steps: int = 1) -> AttentionDecoderCache:
        """Precompute K, V for the attention mechanism.

        Args:
            embeddings: Node embeddings from the encoder.
            num_steps: Unused.

        Returns:
            AttentionDecoderCache: Cache object containing keys and values.
        """
        node_embeddings = embeddings

        # Joint projection
        qkv = self.project_node_embeddings(node_embeddings)
        glimpse_K, glimpse_V, logit_K = qkv.chunk(3, dim=-1)

        fixed = AttentionDecoderCache(
            node_embeddings=node_embeddings,
            graph_context=None,
            glimpse_key=make_heads(glimpse_K, self.n_heads),
            glimpse_val=make_heads(glimpse_V, self.n_heads),
            logit_key=make_heads(logit_K, self.n_heads),
        )
        return fixed

    def _get_log_p(
        self,
        fixed: AttentionDecoderCache,
        state: Any,
        normalize: bool = True,
        mask_val: float = -math.inf,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities for the current state.

        Args:
            fixed: Fixed keys/values in the attention mechanism.
            state: Current environment state.
            normalize: Whether to apply log-softmax.
            mask_val: Value for masked-out logits.
            mask: Optional additional mask.

        Returns:
            Tuple: Log probabilities and the resolved mask.

        Raises:
            ValueError: If no mask is available from state or argument.
        """
        query = self._get_parallel_step_context(fixed.node_embeddings, state)

        # 1. Resolve the mask and ensure it's not None for the type checker
        # We combine the state mask (visited nodes) with the optional external mask.
        current_mask = state.get_mask()
        if mask is not None:
            current_mask = current_mask | mask if current_mask is not None else mask

        if current_mask is None:
            raise ValueError("A mask must be provided either as an argument or via the state.")

        # 2. Handle the unsqueeze logic on the guaranteed tensor
        # We use a temporary variable so the analyzer sees 'Tensor', not 'Optional[Tensor]'
        input_mask = current_mask.unsqueeze(1) if current_mask.dim() == 2 else current_mask

        logits = one_to_many_logits(
            query.unsqueeze(1),
            fixed.glimpse_key,  # type: ignore[arg-type]
            fixed.glimpse_val,  # type: ignore[arg-type]
            fixed.logit_key,  # type: ignore[arg-type]
            input_mask,  # Now guaranteed to be a Tensor
            self.n_heads,
            tanh_clipping=self.tanh_clipping,
            mask_val=mask_val,
            mask_inner=getattr(self, "mask_inner", True),
        ).squeeze(1)

        if normalize:
            return torch.log_softmax(logits, dim=-1), current_mask
        return logits, current_mask

    def _get_parallel_step_context(
        self, embeddings: torch.Tensor, state: Any, from_depot: bool = False
    ) -> torch.Tensor:
        """Extract step context from state and project.

        Args:
            embeddings: Node embeddings.
            state: Current environment state.
            from_depot: Flag for starting from depot.

        Returns:
            torch.Tensor: Projected step context query.
        """
        if hasattr(self, "context_embedding") and self.context_embedding is not None:
            step_context = self.context_embedding(embeddings, state)
            query = self.project_step_context(step_context)
            if query.dim() == 3:
                query = query.squeeze(1)
            return query

        current_node = state.get_current_node() if hasattr(state, "get_current_node") else state.get("current_node")
        if current_node.dim() > 1:
            current_node = current_node.squeeze(-1)
        batch_size = current_node.size(0)

        if embeddings.size(0) != batch_size:
            if embeddings.size(0) == 1:
                embeddings = embeddings.expand(batch_size, -1, -1)
            elif batch_size % embeddings.size(0) == 0:
                embeddings = embeddings.repeat_interleave(batch_size // embeddings.size(0), dim=0)
            else:
                embeddings = embeddings.expand(batch_size, -1, -1)

        # Simple context: [batch, embed]
        # AM typically uses [embeddings(current_node), embeddings(first_node), capacity_left]
        # Here we follow a simplified version matching the original code
        step_context = embeddings.gather(
            1, current_node.unsqueeze(1).unsqueeze(-1).expand(batch_size, 1, self.embed_dim)
        ).squeeze(1)

        return self.project_step_context(step_context)

    def _calc_log_likelihood(
        self,
        _log_p: torch.Tensor,
        a: torch.Tensor,
        mask: Optional[torch.Tensor],
        return_entropy: bool = False,
        kl_loss: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Utility for log-likelihood and entropy calculation.

        Args:
            _log_p: Log probability predictions.
            a: Actions taken.
            mask: Valid sample mask.
            return_entropy: Whether to return entropy of distribution.
            kl_loss: Unused flag.

        Returns:
            Union: Log-likelihood or (log-likelihood, entropy).
        """
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        if mask is not None:
            log_p[mask] = 0

        if return_entropy:
            entropy = -(_log_p.exp() * _log_p).sum(2)
            if mask is not None:
                entropy[mask] = 0
            return log_p, entropy

        return log_p
