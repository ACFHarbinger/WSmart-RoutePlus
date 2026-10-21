"""Multi-Head Attention (MHA) logic for constructive decoders.

This module provides utility functions for computing attention-based logits
and reshaping tensors for multi-head attention mechanisms used in decoders.

Attributes:
    one_to_many_logits: Function to compute attention logits for multiple nodes.
    make_heads: Function to reshape tensors for multi-head attention.

Example:
    >>> from logic.src.models.subnets.decoders.glimpse.attention import one_to_many_logits, make_heads
    >>> k_heads = make_heads(k_tensor, n_heads=8)
    >>> logits = one_to_many_logits(query, k_heads, v_heads, logit_k, mask, n_heads=8)
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def one_to_many_logits(
    query: torch.Tensor,
    glimpse_K: torch.Tensor,
    glimpse_V: torch.Tensor,
    logit_K: torch.Tensor,
    mask: torch.Tensor,
    n_heads: int,
    graph_mask: Optional[torch.Tensor] = None,
    dist_bias: Optional[torch.Tensor] = None,
    mask_val: float = -math.inf,
    tanh_clipping: float = 10.0,
    mask_inner: bool = True,
) -> torch.Tensor:
    """Computes attention logits for a query against multiple nodes.

    Args:
        query: Query embeddings of shape (batch, steps, embed_dim).
        glimpse_K: Key embeddings for glimpse attention.
        glimpse_V: Value embeddings for glimpse attention.
        logit_K: Key embeddings for final logit computation.
        mask: Mask for visited/invalid nodes.
        n_heads: Number of attention heads.
        graph_mask: Optional graph-level mask.
        dist_bias: Optional distance or cost bias.
        mask_val: Value to use for masked nodes.
        tanh_clipping: Scaling factor for tanh logit clipping.
        mask_inner: Whether to mask invalid nodes in the inner glimpse attention.

    Returns:
        torch.Tensor: Logits for node selection of shape (batch, graph_size).
    """
    batch_size, num_steps, embed_dim = query.size()
    key_size = embed_dim // n_heads
    # Compute heads: [batch_size, n_heads, num_steps, key_size]
    q_heads = query.view(batch_size, num_steps, n_heads, key_size).transpose(1, 2)

    # glimpse_K: [batch_size, n_heads, graph_size, key_size]
    # glimpse_V: [batch_size, n_heads, graph_size, key_size]
    # logit_K: [batch_size, n_heads, graph_size, key_size]

    # Attention scores: [batch_size, n_heads, num_steps, graph_size]
    attn_scores = torch.matmul(q_heads, glimpse_K.transpose(-2, -1)) / math.sqrt(key_size)

    # Masking
    if mask is not None and mask_inner:
        # mask: [batch_size, num_steps, graph_size]
        mask_expanded = mask.unsqueeze(1)  # [batch_size, 1, num_steps, graph_size]
        attn_scores = attn_scores.masked_fill(mask_expanded, mask_val)

    if graph_mask is not None:
        attn_scores = attn_scores.masked_fill(graph_mask.unsqueeze(1).unsqueeze(2), mask_val)

    attn_probs = torch.softmax(attn_scores, dim=-1)

    # Glimpse: [batch_size, n_heads, num_steps, key_size]
    glimpse = torch.matmul(attn_probs, glimpse_V)

    # Project back
    glimpse_combined = glimpse.transpose(1, 2).contiguous().view(batch_size, num_steps, embed_dim)

    # Final logits
    # query_final: [batch_size, n_heads, num_steps, key_size]
    # We use a separate projection for the final logit K as in AM
    q_final = glimpse_combined.view(batch_size, num_steps, n_heads, key_size).transpose(1, 2)
    logits = torch.matmul(q_final, logit_K.transpose(-2, -1)) / math.sqrt(key_size)

    # Combine heads by averaging? No, usually AM uses MHA and then one final projection.
    # Actually, the implementation in GlimpseDecoder averages heads for logits.
    logits = logits.mean(dim=1)

    if dist_bias is not None:
        logits = logits + dist_bias

    if tanh_clipping > 0:
        logits = torch.tanh(logits) * tanh_clipping

    if mask is not None:
        logits = logits.masked_fill(mask, mask_val)

    return logits


def make_heads(v: torch.Tensor, n_heads: int, num_steps: Optional[int] = None) -> torch.Tensor:
    """Reshapes tensor into heads for multi-head attention.

    Args:
        v: Input tensor of shape (batch, size, embed_dim).
        n_heads: Number of attention heads.
        num_steps: Optional number of steps (defaults to graph size).

    Returns:
        torch.Tensor: Reshaped tensor of shape (batch, n_heads, size, head_dim).
    """
    batch_size, graph_size, embed_dim = v.size()
    if num_steps is None:
        num_steps = graph_size

    # heads: [batch_size, n_heads, graph_size, key_size]
    return v.view(batch_size, graph_size, n_heads, embed_dim // n_heads).transpose(1, 2)
