"""MDAM Multi-Head Attention module.

This module provides the MultiHeadAttentionMDAM layer, which implements the
specialized multi-head attention mechanism used in the Multi-Decoder Attention
Model (MDAM). It supports returning intermediate tensors required for dynamic
embedding updates during decoding.

Attributes:
    MultiHeadAttentionMDAM: MDAM-specific attention mechanism with coupling support.

Example:
    >>> import torch
    >>> from logic.src.models.subnets.modules.mdam_attention import MultiHeadAttentionMDAM
    >>> model = MultiHeadAttentionMDAM(embed_dim=128, n_heads=8)
    >>> q = torch.randn(1, 4, 128)
    >>> out = model(q)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadAttentionMDAM(nn.Module):
    """Multi-Head Attention for MDAM with optional routing feedback.

    Unlike standard MHA, this specialized module can return intermediate attention
    weights and value matrices. These are leveraged by the MDAM encoder to
    perform "dynamic re-encoding" where embeddings are updated based on the
    decoder's attention patterns.

    Reference:
        Xin, L., et al. (2021). Multi-Decoder Attention Model with Embedding
        Glimpse for Solving Vehicle Routing Problems. AAAI Conference on
        Artificial Intelligence.

    Attributes:
        embed_dim (int): Total dimensionality of the input/output embeddings.
        n_heads (int): Number of parallel attention heads.
        val_dim (int): Dimensionality of values per head (full `embed_dim` for MDAM).
        norm_factor (float): Scaling factor for scores (1 / sqrt(embed_dim)).
        W_query (nn.Parameter): Learnable query projection weights for each head.
        W_key (nn.Parameter): Learnable key projection weights for each head.
        W_val (nn.Parameter): Learnable value projection weights for each head.
        W_out (nn.Parameter): Learnable output projection weights for consolidation.
        last_one (bool): Whether to return intermediate attention tensors.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        last_one: bool = False,
    ) -> None:
        """Initializes MultiHeadAttentionMDAM.

        Args:
            embed_dim: Feature dimensionality of input embeddings.
            n_heads: Number of parallel attention heads.
            last_one: If True, `forward` returns (output, attention, values).
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.val_dim = embed_dim  # Full embed_dim per head for MDAM
        self.norm_factor = 1 / math.sqrt(embed_dim)

        # Separate weight matrices for each head
        self.W_query = nn.Parameter(torch.Tensor(n_heads, embed_dim, embed_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, embed_dim, embed_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, embed_dim, embed_dim))
        self.W_out = nn.Parameter(torch.Tensor(n_heads, embed_dim, embed_dim))

        self._init_parameters()
        self.last_one = last_one

    def _init_parameters(self) -> None:
        """Initializes weights using a uniform distribution."""
        for param in self.parameters():
            if param.dim() > 0:
                stdv = 1.0 / math.sqrt(param.size(-1))
                param.data.uniform_(-stdv, stdv)

    def forward(
        self,
        q: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Computes the MDAM multi-head attention update.

        Args:
            q: Query sequence tensor of shape (batch, n_query, dim).
            h: Key/Value sequence tensor of shape (batch, n_graph, dim).
                If None, defaults to self-attention on `q`.
            mask: Boolean attention mask (batch, n_query, n_graph).
                True indicates blocked positions.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                If `last_one` is False:
                    - Transformed output tensor of shape (batch, n_query, dim).
                If `last_one` is True:
                    - output: Transformed features (batch, n_query, dim).
                    - attn: Attention weights (heads, batch, n_query, n_graph).
                    - V: Internal value matrices (heads, batch, n_graph, dim).
        """
        if h is None:
            h = q  # Self-attention

        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.embed_dim, "Wrong embedding dimension of input"

        # Flatten for matrix multiplication
        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # Shapes for reshaping after matmul
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Compute Q, K, V projections
        # Shape: (n_heads, batch_size, seq_len, embed_dim)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Compute attention scores
        # Shape: (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask_expanded] = float("-inf")

        # Compute attention weights
        attn = F.softmax(compatibility, dim=-1)

        # Handle NaN from softmax when all values are masked
        if mask is not None:
            attnc = attn.clone()
            mask_expanded = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            attnc[mask_expanded] = 0
            attn = attnc

        # Compute attention output
        # heads: (n_heads, batch_size, n_query, embed_dim)
        heads = torch.matmul(attn, V)

        # Project and reshape to output
        # (batch_size, n_query, n_heads * embed_dim) -> (batch_size, n_query, embed_dim)
        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.embed_dim),
            self.W_out.view(-1, self.embed_dim),
        ).view(batch_size, n_query, self.embed_dim)

        if self.last_one:
            return out, attn, V
        return out
