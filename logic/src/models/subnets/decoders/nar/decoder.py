"""Simple NAR Decoder: Constructs solutions using greedy/sampling from heatmaps.

This module provides a non-autoregressive decoder that directly utilizes
edge or node heatmaps to produce selection logits.

Attributes:
    SimpleNARDecoder: Decoder for Non-Autoregressive models using heatmaps.

Example:
    >>> from logic.src.models.subnets.decoders.nar.decoder import SimpleNARDecoder
    >>> decoder = SimpleNARDecoder()
    >>> logits, mask = decoder(td, heatmap, env)
"""

from __future__ import annotations

from typing import Any, Tuple

import torch
from tensordict import TensorDict

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.non_autoregressive.decoder import NonAutoregressiveDecoder


class SimpleNARDecoder(NonAutoregressiveDecoder):
    """Simple decoder for Non-Autoregressive models.

    Directly uses a pre-computed heatmap (e.g., from a GNN) as logits
    for selecting the next action in a constructive manner.

    Attributes:
        kwargs (Any): Additional configuration parameters.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the SimpleNARDecoder.

        Args:
            kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)

    def forward(
        self,
        td: TensorDict,
        heatmap: torch.Tensor,
        env: RL4COEnvBase,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Produces logits for the current step based on the heatmap.

        Args:
            td: TensorDict containing the current state.
            heatmap: Pre-computed heatmap of shape (batch, n, n) or (batch, n).
            env: Environment instance.
            kwargs: Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Logits for selecting the next node
                and the current selection mask.
        """
        # Current node
        current_node = td.get(
            "current_node",
            torch.zeros(td.batch_size, dtype=torch.long, device=heatmap.device),
        )

        if heatmap.dim() == 3:
            # Edge-based heatmap: [batch, num_nodes, num_nodes]
            # Select logits for current node
            logits = heatmap.gather(1, current_node.view(-1, 1, 1).expand(-1, 1, heatmap.size(-1))).squeeze(1)
        else:
            # Node-based heatmap: [batch, num_nodes]
            logits = heatmap

        return logits, td["mask"]
