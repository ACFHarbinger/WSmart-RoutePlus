"""DACT Decoder implementation.

This module implements the paper-faithful `DACTDecoder` (Ma et al., NeurIPS 2021),
which selects node pairs (i, j) to perform local improvement moves (e.g., 2-opt)
using dual-aspect collaborative representations (node aspect and position aspect).

Architecture (§4.3, Eq. 11, Footnotes 6 & 7):
- Max-pooling context augmentation for both node (h) and position (g) aspects.
- Multi-Head Compatibility (MHC) sub-layer (m=4 heads) producing proposal matrices.
- Feed-Forward Aggregation (FFA) 4-layer MLP (2m -> 32 -> 32 -> 1) synthesizing proposals.
- Entropy control via C * tanh clipping (C=6) and diagonal identity masking.

Attributes:
    DACTDecoder: Pairwise action decoder for iterative improvement.

Example:
    >>> from logic.src.models.core.dact.decoder import DACTDecoder
    >>> decoder = DACTDecoder(embed_dim=128, num_heads=4)
    >>> log_p, actions = decoder(td, h, my_env)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.improvement.policy import ImprovementDecoder


class DACTDecoder(ImprovementDecoder):
    """DACT Pairwise Action Decoder (Ma et al. NeurIPS 2021, §4.3).

    Processes dual-aspect node embeddings to predict the optimal pair of indices
    for iterative improvement moves. It generates multi-head compatibility proposals
    from both node and position aspects, and synthesizes them using a feed-forward
    aggregation (FFA) MLP.

    Attributes:
        embed_dim (int): Dimensionality of latent embeddings.
        num_heads (int): Count of attention heads in MHC (default: 4 per paper).
        tanh_clipping (float): Entropy control constant C (default: 6.0).
        seed (int): Random seed for action sampling.
        generator (torch.Generator): PRNG instance for reproducible sampling.
        W_h_local (nn.Linear): Local projection for node aspect.
        W_h_global (nn.Linear): Global max-pooling projection for node aspect.
        W_g_local (nn.Linear): Local projection for position aspect.
        W_g_global (nn.Linear): Global max-pooling projection for position aspect.
        project_qh (nn.Linear): Query projection for node aspect.
        project_kh (nn.Linear): Key projection for node aspect.
        project_qg (nn.Linear): Query projection for position aspect.
        project_kg (nn.Linear): Key projection for position aspect.
        ffa (nn.Sequential): 4-layer Feed-Forward Aggregation MLP (2m -> 32 -> 32 -> 1).
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        tanh_clipping: float = 6.0,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initializes the paper-faithful DACT decoder.

        Args:
            embed_dim: Dimensionality of latent embeddings.
            num_heads: Number of compatibility attention heads (paper default: 4).
            tanh_clipping: Clipping factor C for entropy control (paper default: 6.0).
            seed: Random seed for action sampling.
            kwargs: Additional keyword arguments.
        """
        super().__init__(embed_dim)
        self.num_heads = num_heads
        self.tanh_clipping = tanh_clipping
        self.seed = seed
        init_device = kwargs.get("device", "cpu")
        self.generator = torch.Generator(device=init_device).manual_seed(self.seed)

        if embed_dim % num_heads != 0:
            divisors = [d for d in range(1, num_heads + 1) if embed_dim % d == 0]
            num_heads = max(divisors) if divisors else 1
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / (head_dim**0.5)

        # Max-pooling sub-layer projections (Footnote 6)
        self.W_h_local = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_h_global = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_g_local = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_g_global = nn.Linear(embed_dim, embed_dim, bias=False)

        # Multi-Head Compatibility (MHC) projections for node (h) and position (g) aspects
        self.project_qh = nn.Linear(embed_dim, embed_dim, bias=False)
        self.project_kh = nn.Linear(embed_dim, embed_dim, bias=False)
        self.project_qg = nn.Linear(embed_dim, embed_dim, bias=False)
        self.project_kg = nn.Linear(embed_dim, embed_dim, bias=False)

        # Position aspect adapter (used when encoder outputs single embedding tensor)
        self.pos_adapter = nn.Linear(embed_dim, embed_dim)

        # Feed-Forward Aggregation (FFA) MLP (Eq. 11): 2m -> 32 -> 32 -> 1
        ffa_in_dim = 2 * num_heads
        self.ffa = nn.Sequential(
            nn.Linear(ffa_in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    @property
    def device(self) -> torch.device:
        """Determines the current hardware placement of the model.

        Returns:
            torch.device: Current placement of the model parameters.
        """
        return next(self.parameters()).device

    def __getstate__(self) -> Dict[str, Any]:
        """Serializes the state, handling non-picklable components.

        Returns:
            Dict[str, Any]: Model state dictionary with generator metadata.
        """
        state = self.__dict__.copy()
        state["generator_state"] = self.generator.get_state()
        state["generator_device"] = str(self.generator.device)
        del state["generator"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restores the model state including the RNG generator.

        Args:
            state: Serialized attribute map.
        """
        gen_state = state.pop("generator_state")
        gen_device = state.pop("generator_device")
        self.__dict__.update(state)
        self.generator = torch.Generator(device=gen_device)
        self.generator.set_state(gen_state)

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Validates checkpoint keys against legacy generic decoder weights."""
        legacy_keys = {prefix + "project_q.weight", prefix + "project_k.weight"}
        present_legacy = [k for k in legacy_keys if k in state_dict]
        has_new_keys = any(k.startswith(prefix + "W_h_") or k.startswith(prefix + "project_qh") for k in state_dict)
        if present_legacy and not has_new_keys:
            raise RuntimeError(
                f"Detected legacy generic pairwise decoder checkpoint with keys {present_legacy} "
                f"for {self.__class__.__name__}. DACTDecoder has been upgraded to the paper-faithful "
                f"Dual-Aspect Collaborative architecture (Ma et al. NeurIPS 2021). "
                f"Legacy checkpoints cannot be loaded silently and must be retrained."
            )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(
        self,
        td: TensorDict,
        embeddings: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        env: RL4COEnvBase,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts a node pair for an improvement operator via Dual-Aspect MHC + FFA.

        Args:
            td: TensorDict containing problem instance data.
            embeddings: Contextual node features [B, N, d] or tuple (h, g).
            env: Environment managing problem physics.
            kwargs: Additional keyword arguments including 'strategy'.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - log_p: Log-probability of the selected pair [B].
                - actions: Integer indices [i, j] for the move [B, 2].
        """
        # 1. Unpack or derive dual aspects: node aspect h and position aspect g
        if isinstance(embeddings, (tuple, list)) and len(embeddings) >= 2:
            h = embeddings[0]
            g = embeddings[1]
        else:
            h = embeddings[0] if isinstance(embeddings, (tuple, list)) else embeddings
            g = self.pos_adapter(h)

        bs, n, d = h.shape
        m = self.num_heads
        dk = self.head_dim

        # 2. Max-pooling context augmentation (Footnote 6)
        h_max = h.max(dim=1, keepdim=True).values  # [B, 1, d]
        h_hat = self.W_h_local(h) + self.W_h_global(h_max)  # [B, N, d]

        g_max = g.max(dim=1, keepdim=True).values  # [B, 1, d]
        g_hat = self.W_g_local(g) + self.W_g_global(g_max)  # [B, N, d]

        # 3. Multi-Head Compatibility (MHC) Sub-layer (§4.3)
        # Node aspect MHC proposals: Y_h [B, m, N, N]
        qh = self.project_qh(h_hat).view(bs, n, m, dk).permute(0, 2, 1, 3)
        kh = self.project_kh(h_hat).view(bs, n, m, dk).permute(0, 2, 1, 3)
        y_h = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale

        # Position aspect MHC proposals: Y_g [B, m, N, N]
        qg = self.project_qg(g_hat).view(bs, n, m, dk).permute(0, 2, 1, 3)
        kg = self.project_kg(g_hat).view(bs, n, m, dk).permute(0, 2, 1, 3)
        y_g = torch.matmul(qg, kg.transpose(-2, -1)) * self.scale

        # 4. Feed-Forward Aggregation (FFA) MLP (Eq. 11)
        # Concatenate 2m proposals along head dimension: [B, 2m, N, N] -> [B, N, N, 2m]
        proposals = torch.cat([y_h, y_g], dim=1).permute(0, 2, 3, 1)
        y_tilde = self.ffa(proposals).squeeze(-1)  # [B, N, N]

        # 5. Entropy control (clipping) and diagonal masking
        scores = self.tanh_clipping * torch.tanh(y_tilde)
        mask = torch.eye(n, device=h.device, dtype=torch.bool).unsqueeze(0).expand(bs, -1, -1)
        scores = scores.masked_fill(mask, float("-inf"))

        logits = scores.view(bs, -1)

        # 6. Action selection
        strategy = kwargs.get("strategy", "greedy")
        if strategy == "greedy":
            action_indices = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            action_indices = torch.multinomial(probs, 1, generator=self.generator).squeeze(-1)

        # Coordinate conversion
        idx_i = torch.div(action_indices, n, rounding_mode="floor")
        idx_j = action_indices % n
        actions = torch.stack([idx_i, idx_j], dim=-1)

        log_p = F.log_softmax(logits, dim=-1).gather(1, action_indices.unsqueeze(-1)).squeeze(-1)

        return log_p, actions
