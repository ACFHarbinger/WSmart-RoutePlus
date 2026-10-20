"""N2S Decoder implementation.

The forward pass lives in ``decoder_forward``.

Partner-resolution methods live in ``decoder_partners``.

This module implements the paper-faithful `N2SDecoder` (Li et al., IJCAI 2021),
which factorizes neighborhood search moves into:
1. Node-Pair / Request Removal Decoder (§4.2, Eq. 12-13):
   - Computes tour-closeness scores lambda_i (Eq. 12) measuring connection strength to tour neighbors.
   - Evaluates removal distribution via MLP_lambda (Eq. 13, 2m + 4 -> 32 -> 32 -> 1) taking
     multi-head closeness and historical frequency/recency features.
   - Samples the node / request to remove with log P_remove.
2. Conditional Reinsertion Decoder (§4.2, Eq. 14-15):
   - Computes directional preference matrices mu_p and mu_s (Eq. 14).
   - Evaluates conditional reinsertion position distribution via MLP_mu (Eq. 15, 4m -> 32 -> 32 -> 1)
     conditioned on the removed node and its tour partner.
   - Enforces precedence and validity constraints (masking immediate predecessor and identity).
   - Samples reinsertion target with log P_reinsert|remove.
3. Joint action [i_plus, i_minus, j, k] and factorized log-likelihood log P_rem + log P_reinsert.
   The request pickup/delivery nodes are reinserted after residual-tour anchor nodes j/k.

Attributes:
    N2SDecoder: Paper-faithful removal and conditional reinsertion decoder.

Example:
    >>> decoder = N2SDecoder(embed_dim=128, num_heads=4)
    >>> log_p, actions = decoder(td, h, env)
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from logic.src.models.common.improvement.decoder import ImprovementDecoder

from .decoder_forward import (
    N2SDecoderForwardMixin as N2SDecoderForwardMixin,
)
from .decoder_partners import (
    N2SDecoderPartnersMixin as N2SDecoderPartnersMixin,
)


class N2SDecoder(N2SDecoderForwardMixin, N2SDecoderPartnersMixin, ImprovementDecoder):
    """N2S Decoder with Request-Removal and Conditional Reinsertion (Li et al. IJCAI 2021, §4.2).

    Conditioned on the incumbent tour permutation, it explicitly factorizes the neighborhood move:
    1. Evaluates candidate request/node removal using neighbor closeness and history via MLP_lambda.
    2. Conditions joint reinsertion positions on the selected removal request via MLP_mu with
       precedence and topological validity masking.

    Attributes:
        embed_dim (int): Dimensionality of node embeddings.
        num_heads (int): Count of attention heads (default: 4 per paper).
        tanh_clipping (float): Entropy control constant C (default: 6.0).
        seed (int): RNG seed for sampling.
        generator (torch.Generator): PRNG instance for reproducible sampling.
        W_h_local (nn.Linear): Local projection for max-pooling layer (Eq. 11).
        W_h_global (nn.Linear): Global projection for max-pooling layer (Eq. 11).
        project_q_lambda (nn.Linear): Query projection for neighbor closeness (Eq. 12).
        project_k_lambda (nn.Linear): Key projection for neighbor closeness (Eq. 12).
        mlp_lambda (nn.Sequential): 3-layer MLP for removal distribution (Eq. 13, 2m + 4 -> 32 -> 32 -> 1).
        project_qp (nn.Linear): Predecessor query projection (Eq. 14).
        project_kp (nn.Linear): Predecessor key projection (Eq. 14).
        project_qs (nn.Linear): Successor query projection (Eq. 14).
        project_ks (nn.Linear): Successor key projection (Eq. 14).
        mlp_mu (nn.Sequential): 4-layer MLP for conditional reinsertion (Eq. 15, 4m -> 32 -> 32 -> 1).
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        tanh_clipping: float = 6.0,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initializes the paper-faithful N2S decoder.

        Args:
            embed_dim: Dimensionality of the node features.
            num_heads: Number of attention heads (paper default: 4).
            tanh_clipping: Clipping factor C for entropy control (paper default: 6.0).
            seed: Random seed for action sampling.
            kwargs: Additional keyword arguments.
        """
        super().__init__(embed_dim=embed_dim)
        self.embed_dim = embed_dim
        if embed_dim % num_heads != 0:
            divisors = [d for d in range(1, num_heads + 1) if embed_dim % d == 0]
            num_heads = max(divisors) if divisors else 1
        self.num_heads = num_heads
        self.tanh_clipping = tanh_clipping
        self.seed = seed
        init_device = kwargs.get("device", "cpu")
        self.generator = torch.Generator(device=init_device).manual_seed(self.seed)

        head_dim = embed_dim // num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / (head_dim**0.5)

        # Max-pooling sub-layer (Eq. 11)
        self.W_h_local = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_h_global = nn.Linear(embed_dim, embed_dim, bias=False)

        # Closeness scoring projections (Eq. 12)
        self.project_q_lambda = nn.Linear(embed_dim, embed_dim, bias=False)
        self.project_k_lambda = nn.Linear(embed_dim, embed_dim, bias=False)

        # Node-Pair Removal MLP_lambda (Eq. 13): (2m + 4 -> 32 -> 32 -> 1)
        mlp_lambda_in = 2 * num_heads + 4
        self.mlp_lambda = nn.Sequential(
            nn.Linear(mlp_lambda_in, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Directional insertion preference projections (Eq. 14)
        self.project_qp = nn.Linear(embed_dim, embed_dim, bias=False)
        self.project_kp = nn.Linear(embed_dim, embed_dim, bias=False)
        self.project_qs = nn.Linear(embed_dim, embed_dim, bias=False)
        self.project_ks = nn.Linear(embed_dim, embed_dim, bias=False)

        # Conditional Reinsertion MLP_mu (Eq. 15): (4m -> 32 -> 32 -> 1)
        mlp_mu_in = 4 * num_heads
        self.mlp_mu = nn.Sequential(
            nn.Linear(mlp_mu_in, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    @property
    def device(self) -> torch.device:
        """Determines the current hardware placement of parameters.

        Returns:
            torch.device: The active device (CPU/CUDA).
        """
        return next(self.parameters()).device

    def __getstate__(self) -> Dict[str, Any]:
        """Serializes the state, handling non-picklable components.

        Returns:
            Dict[str, Any]: Attribute map with generator state recorded.
        """
        state = self.__dict__.copy()
        state["generator_state"] = self.generator.get_state()
        state["generator_device"] = str(self.generator.device)
        del state["generator"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restores policy state including the RNG generator.

        Args:
            state: Serialized dictionary of attributes.
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
        has_new_keys = any(k.startswith(prefix + "W_h_") or k.startswith(prefix + "mlp_lambda") for k in state_dict)
        if present_legacy and not has_new_keys:
            raise RuntimeError(
                f"Detected legacy generic pairwise decoder checkpoint with keys {present_legacy} "
                f"for {self.__class__.__name__}. N2SDecoder has been upgraded to the paper-faithful "
                f"tour-aware request-removal and conditional reinsertion architecture (Li et al. IJCAI 2021). "
                f"Legacy checkpoints cannot be loaded silently and must be retrained."
            )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
