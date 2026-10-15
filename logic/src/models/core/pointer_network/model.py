"""Pointer Network for routing problems.

This module provides the classic `PointerNetwork` (Vinyals et al. 2015), an
LSTM-based architecture that uses attention as a pointer to select nodes from
the input sequence, allowing it to handle variable-length sequences effectively.

Attributes:
    PointerNetwork: Classic sequence-to-sequence pointer architecture.

Example:
    >>> model = PointerNetwork(embed_dim=128, hidden_dim=512, problem=tsp)
    >>> cost, ll = model(inputs)
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

import torch
from torch import nn

from logic.src.models.subnets import PointerDecoder, PointerEncoder


class PointerNetwork(nn.Module):
    """Pointer Network with Attention-based node selection.

    Maintains a hidden state through an LSTM encoder and uses a glimpse-based
    decoder to select input nodes competitively. Heavily used as a baseline for
    neural combinatorial optimization.

    Attributes:
        problem (Any): Optimization problem wrapper.
        input_dim (int): Dimensionality of raw spatial input (e.g., 2 for xy).
        encoder (PointerEncoder): LSTM encoder for input nodes.
        decoder (PointerDecoder): Pointer-based constructor.
        decoder_in_0 (nn.Parameter): Initial hidden state for the decoder.
        embedding (nn.Parameter): Feature projection weights.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        problem: Any,
        n_encode_layers: Optional[int] = None,
        tanh_clipping: float = 10.0,
        mask_inner: bool = True,
        mask_logits: bool = True,
        normalization: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the Pointer Network.

        Args:
            embed_dim: Dimensionality of latent embeddings.
            hidden_dim: Dimensionality of LSTM hidden states.
            problem: Environment managing rewards and problem constraints.
            n_encode_layers: Number of encoder layers (defaults to 1).
            tanh_clipping: Range for attention logit clipping.
            mask_inner: Whether to mask glimpses during pointer calculation.
            mask_logits: Whether to mask invalid actions in the output layer.
            normalization: Type of normalization (e.g., "batch", "layer").
            kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.problem = problem
        self.input_dim = 2
        self.encoder = PointerEncoder(embed_dim, hidden_dim)
        self.decoder = PointerDecoder(
            embed_dim,
            hidden_dim,
            tanh_exploration=tanh_clipping,
            use_tanh=tanh_clipping > 0,
            n_glimpses=1,
            mask_glimpses=mask_inner,
            mask_logits=mask_logits,
        )

        # Trainable initial hidden states
        std = 1.0 / math.sqrt(embed_dim)
        self.decoder_in_0 = nn.Parameter(torch.FloatTensor(embed_dim))
        self.decoder_in_0.data.uniform_(-std, std)

        self.embedding = nn.Parameter(torch.FloatTensor(self.input_dim, embed_dim))
        self.embedding.data.uniform_(-std, std)

    def set_strategy(self, strategy: str) -> None:
        """Configures the current action selection tactic.

        Args:
            strategy: Mode identifier (e.g., 'greedy', 'sampling').
        """
        self.decoder.strategy = strategy

    def forward(
        self,
        inputs: torch.Tensor,
        eval_tours: Optional[torch.Tensor] = None,
        return_pi: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Performs a forward pass to construct a tour.

        Args:
            inputs: Physical coordinates [Batch, Nodes, Dim].
            eval_tours: Pre-defined tour indices for teacher forcing.
            return_pi: Toggle return of constructed action indices.

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                - cost: total solution length [Batch].
                - ll: path log likelihood [Batch].
                - pi: action indices (if return_pi) [Batch, SeqLen].
        """
        batch_size, graph_size, input_dim = inputs.size()
        embedded_inputs = torch.mm(inputs.transpose(0, 1).contiguous().view(-1, input_dim), self.embedding).view(
            graph_size, batch_size, -1
        )

        # 1. Recursive decoding pass
        _log_p, pi = self._inner(embedded_inputs, eval_tours)

        # 2. Score construction
        res = self.problem.evaluate_tour(inputs, pi)
        if isinstance(res, dict):
            cost = res["cost"]
            mask = res.get("mask", None)
        elif isinstance(res, tuple):
            cost = res[0]
            mask = res[1] if len(res) > 1 else None
        else:
            cost = res
            mask = None

        # 3. Log likelihood calculation for training
        ll = self._calc_log_likelihood(_log_p, pi, mask)

        if return_pi:
            return cost, ll, pi

        return cost, ll

    def _calc_log_likelihood(self, _log_p: torch.Tensor, a: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Computes the total log probability of an action sequence.

        Args:
            _log_p: Predicted log probabilities across all steps [Batch, SeqLen, Nodes].
            a: Selected action indices [Batch, SeqLen].
            mask: Binary mask for cost-relevant steps [Batch, SeqLen].

        Returns:
            torch.Tensor: Cumulative log likelihood per instance [Batch].
        """
        # Gather probabilities of specific actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Masking redundant actions (e.g. padding in variable length)
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).all(), "Logprobs should not be -inf, check sampling procedure!"

        return log_p.sum(1)

    def _inner(
        self, inputs: torch.Tensor, eval_tours: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Core encoder-decoder execution loop.

        Args:
            inputs: Embedded node features [Nodes, Batch, Dim].
            eval_tours: Teacher forcing targets.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - log_probabilities: [Nodes, Batch, Nodes].
                - action_indices: [Nodes, Batch].
        """
        # Initial LSTM states
        encoder_hx = encoder_cx = torch.zeros(1, inputs.size(1), self.encoder.hidden_dim, device=inputs.device)

        # Encoder forward pass
        enc_h, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx, encoder_cx))
        dec_init_state = (enc_h_t[-1], enc_c_t[-1])

        # Initial decoder input (learned parameter)
        decoder_input = self.decoder_in_0.unsqueeze(0).repeat(inputs.size(1), 1)

        # Sequential pointer-selection pass
        (pointer_probs, input_idxs), _ = self.decoder(decoder_input, inputs, dec_init_state, enc_h, eval_tours)
        return pointer_probs, input_idxs
