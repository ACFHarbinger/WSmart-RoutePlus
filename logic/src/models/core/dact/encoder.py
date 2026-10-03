"""DACT Encoder implementation.

This module provides the paper-faithful `DACTEncoder` (Ma et al., NeurIPS 2021, §4.1-4.2),
which implements the Dual-Aspect Collaborative Transformer architecture. It maintains
two distinct encoding streams throughout all stacked layers:
1. Node Feature Embeddings (NFEs, h): captures spatial coordinates and topology.
2. Positional Feature Embeddings (PFEs, g): captures cyclic solution order.

In each DAC layer (Eq. 5-10):
- Self-attention correlation matrices are computed individually from each aspect.
- Cross-aspect referential attention shares attention correlations as mutual references.
- Independent feed-forward sub-layers (FFN_h, FFN_g) and LayerNorm refine each aspect.
- The encoder returns a tuple (h, g) that flows directly to the DACTDecoder.

Attributes:
    DACAttLayer: Dual-Aspect Collaborative Attention layer with referential attention.
    DACTEncoder: Neural encoder for collaborative node-position modelling.

Example:
    >>> from logic.src.models.core.dact.encoder import DACTEncoder
    >>> encoder = DACTEncoder(embed_dim=128)
    >>> h, g = encoder(td)
"""

from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from logic.src.models.common.improvement.encoder import ImprovementEncoder
from logic.src.models.subnets.modules import Normalization


class DACAttLayer(nn.Module):
    """Dual-Aspect Collaborative Attention (DAC-Att) Layer (Ma et al. 2021, §4.2).

    Simultaneously refines node feature embeddings (h) and positional feature
    embeddings (g) using cross-aspect referential attention (Eq. 7-10) and
    aspect-specific feed-forward networks (Eq. 5-6).
    """

    def __init__(self, embed_dim: int, num_heads: int = 4) -> None:
        """Initializes the DAC-Att layer.

        Args:
            embed_dim: Feature dimension of embeddings.
            num_heads: Number of attention heads (default: 4 per paper).
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            divisors = [d for d in range(1, num_heads + 1) if embed_dim % d == 0]
            num_heads = max(divisors) if divisors else 1
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / (self.head_dim**0.5)

        # Projections for node aspect h (Eq. 7, 8, 10)
        self.W_h_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_h_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_h_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_h_vref = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_h_o = nn.Linear(embed_dim * 2, embed_dim, bias=False)

        # Projections for positional aspect g (Eq. 7, 9, 10)
        self.W_g_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_g_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_g_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_g_vref = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_g_o = nn.Linear(embed_dim * 2, embed_dim, bias=False)

        # Normalization and aspect-specific FFNs (Eq. 5 & 6)
        self.norm_h1 = Normalization(embed_dim, "layer")
        self.norm_g1 = Normalization(embed_dim, "layer")

        self.ffn_h = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm_h2 = Normalization(embed_dim, "layer")

        self.ffn_g = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm_g2 = Normalization(embed_dim, "layer")

    def forward(self, h: torch.Tensor, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refines both aspect embeddings via DAC-Att.

        Args:
            h: Node feature embeddings [B, N, d].
            g: Positional feature embeddings [B, N, d].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (h_out [B, N, d], g_out [B, N, d]).
        """
        bs, n, d = h.shape
        m, dk = self.num_heads, self.head_dim

        # 1. Self-attention correlations for h and g (Eq. 7)
        qh = self.W_h_q(h).view(bs, n, m, dk).permute(0, 2, 1, 3)
        kh = self.W_h_k(h).view(bs, n, m, dk).permute(0, 2, 1, 3)
        alpha_h = F.softmax(torch.matmul(qh, kh.transpose(-2, -1)) * self.scale, dim=-1)

        qg = self.W_g_q(g).view(bs, n, m, dk).permute(0, 2, 1, 3)
        kg = self.W_g_k(g).view(bs, n, m, dk).permute(0, 2, 1, 3)
        alpha_g = F.softmax(torch.matmul(qg, kg.transpose(-2, -1)) * self.scale, dim=-1)

        # 2. Cross-aspect referential attention (Eq. 8 & 9)
        vh = self.W_h_v(h).view(bs, n, m, dk).permute(0, 2, 1, 3)
        vh_ref = self.W_h_vref(h).view(bs, n, m, dk).permute(0, 2, 1, 3)
        val_h = torch.matmul(alpha_h, vh).permute(0, 2, 1, 3).contiguous().view(bs, n, d)
        val_h_ref = torch.matmul(alpha_g, vh_ref).permute(0, 2, 1, 3).contiguous().view(bs, n, d)
        h_attn = self.W_h_o(torch.cat([val_h, val_h_ref], dim=-1))

        vg = self.W_g_v(g).view(bs, n, m, dk).permute(0, 2, 1, 3)
        vg_ref = self.W_g_vref(g).view(bs, n, m, dk).permute(0, 2, 1, 3)
        val_g = torch.matmul(alpha_g, vg).permute(0, 2, 1, 3).contiguous().view(bs, n, d)
        val_g_ref = torch.matmul(alpha_h, vg_ref).permute(0, 2, 1, 3).contiguous().view(bs, n, d)
        g_attn = self.W_g_o(torch.cat([val_g, val_g_ref], dim=-1))

        # 3. Skip connections & LayerNorm (Eq. 5 & 6)
        h_mid = self.norm_h1(h + h_attn)
        g_mid = self.norm_g1(g + g_attn)

        # 4. Independent FFN sub-layers
        h_out = self.norm_h2(h_mid + self.ffn_h(h_mid))
        g_out = self.norm_g2(g_mid + self.ffn_g(g_mid))

        return h_out, g_out


class DACTEncoder(ImprovementEncoder):
    """DACT Dual-Aspect Collaborative Encoder (Ma et al. 2021).

    Processes the problem instance from two separate, collaborative streams:
    - Node aspect (coordinates and depot).
    - Positional aspect (cyclic positional encoding of solution order).

    Attributes:
        node_init (nn.Linear): Initial projection for spatial coordinates.
        pos_init (nn.Linear): Initial projection for cyclic positional coordinates.
        layers (nn.ModuleList): Stack of DACAttLayer layers.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        pos_type: str = "CPE",
        **kwargs: Any,
    ) -> None:
        """Initializes the paper-faithful DACT encoder.

        Args:
            embed_dim: Dimensionality of latent embeddings.
            num_layers: Number of collaborative transformer layers.
            num_heads: Number of attention heads (default: 4 per paper).
            pos_type: Positional encoding type.
            kwargs: Additional keyword arguments.
        """
        super().__init__(embed_dim)
        self.node_init = nn.Linear(2, embed_dim)

        # Positional aspect initialization: linear projection of multi-dimensional CPE (Eq. 4 & 13)
        self.pos_init = nn.Linear(embed_dim, embed_dim, bias=False)

        self.layers = nn.ModuleList([DACAttLayer(embed_dim=embed_dim, num_heads=num_heads) for _ in range(num_layers)])

    def _compute_cyclic_positions(self, solution: torch.Tensor, n_nodes: int) -> torch.Tensor:
        """Computes multi-dimensional Cyclic Positional Encoding (CPE) per Ma et al. 2021, Eq. (4) & (13).

        Args:
            solution: Tour sequence indices [B, N].
            n_nodes: Number of nodes in the problem instance.

        Returns:
            torch.Tensor: Multi-dimensional CPE features [B, N, embed_dim].
        """
        bs = solution.size(0)
        device = solution.device
        dim = self.embed_dim
        half_dim = dim // 2

        # Invert permutation: find position k of node i in solution
        rank = torch.zeros(bs, n_nodes, dtype=torch.float32, device=device)
        positions = torch.arange(n_nodes, dtype=torch.float32, device=device).unsqueeze(0).expand(bs, -1)
        rank.scatter_(1, solution, positions)

        d = torch.arange(dim, device=device, dtype=torch.float32)

        # Wavelengths per dimension lambda_d per Appendix B Eq. (13)
        n_pow = max(float(n_nodes) ** (1.0 / max(half_dim, 1)), 1.0)
        d_div_3 = torch.floor(d / 3.0)
        term1 = (3.0 * d_div_3 + 1.0) / dim
        lambda_d_low = term1 * (float(n_nodes) - n_pow) + n_pow
        lambda_d = torch.where(d < half_dim, lambda_d_low, torch.tensor(float(n_nodes), device=device))
        lambda_d = torch.clamp(lambda_d, min=1e-4)

        # Eq. (4): z(i) = (i / N) * lambda_d * ceil(N / lambda_d)
        rank_exp = rank.unsqueeze(-1)  # [B, N, 1]
        lambda_exp = lambda_d.unsqueeze(0).unsqueeze(0)  # [1, 1, dim]

        ceil_term = torch.ceil(float(n_nodes) / lambda_exp)
        z = (rank_exp / max(float(n_nodes), 1.0)) * lambda_exp * ceil_term

        mod_term = 2.0 * lambda_exp
        pattern = torch.abs((z % mod_term) - lambda_exp)
        phase = (2.0 * torch.pi / lambda_exp) * pattern

        is_even = (torch.arange(dim, device=device) % 2 == 0).unsqueeze(0).unsqueeze(0)
        cpe = torch.where(is_even, torch.sin(phase), torch.cos(phase))  # [B, N, dim]
        return cpe

    def forward(self, td: TensorDict, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes problem instance into dual-aspect embeddings (h, g).

        Args:
            td: TensorDict containing 'depot', 'locs', and 'solution' keys.
            kwargs: Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - h: Refined node feature embeddings [B, N, d].
                - g: Refined positional feature embeddings [B, N, d].
        """
        # 1. Initialize node aspect (h)
        depot = td["depot"].unsqueeze(1) if td["depot"].dim() == 2 else td["depot"]
        locs = td["locs"]
        nodes = torch.cat([depot, locs], dim=1)  # [B, N, 2]
        h = self.node_init(nodes)  # [B, N, d]
        bs, n, _ = nodes.shape

        # 2. Initialize positional aspect (g)
        if "solution" in td.keys():
            pos_coords = self._compute_cyclic_positions(td["solution"], n)
        else:
            identity_tour = torch.arange(n, device=h.device).unsqueeze(0).expand(bs, -1)
            pos_coords = self._compute_cyclic_positions(identity_tour, n)

        g = self.pos_init(pos_coords)  # [B, N, d]

        # 3. Stacked collaborative Dual-Aspect attention (Eq. 5-10)
        for layer in self.layers:
            h, g = layer(h, g)

        return h, g
