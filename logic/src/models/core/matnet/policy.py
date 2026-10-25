"""MatNet Policy for matrix-form problems.

This module provides the `MatNetPolicy`, which integrates the specialized
MatNet encoder and decoder components to solve problems where the input is
primarily a cost or distance matrix (e.g., ATSP, FFSP).

Attributes:
    MatNetPolicy: Autoregressive policy for matrix-based constructive search.

Example:
    >>> from logic.src.models.core.matnet.policy import MatNetPolicy
    >>> policy = MatNetPolicy(embed_dim=256, hidden_dim=512, problem=env.problem)
    >>> out = policy(td, env)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from tensordict import TensorDict

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.autoregressive.policy import (
    AutoregressivePolicy,
)
from logic.src.models.subnets.decoders.matnet import MatNetDecoder
from logic.src.models.subnets.embeddings.matnet import MatNetInitEmbedding
from logic.src.models.subnets.encoders.matnet.encoder import MatNetEncoder


class MatNetPolicy(AutoregressivePolicy):
    """Matrix-aware neural policy.

    Implements the row and column embedding architecture from Kwon et al. (2021).
    The encoder processes the cost matrix into dual embeddings, which the
    decoder then uses to construct a solution.

    Attributes:
        encoder (MatNetEncoder): Specialized matrix-score attention encoder.
        decoder (MatNetDecoder): Autoregressive constructor for matrix problems.
        init_embedding (MatNetInitEmbedding): Statistical feature projection.
        problem (Any): Problem domain context.
        embed_dim (int): feature vector size.
    """

    encoder: MatNetEncoder
    decoder: MatNetDecoder

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        problem: Any,
        num_layers: int = 5,
        n_heads: int = 8,
        tanh_clipping: float = 10.0,
        normalization: str = "instance",
        **kwargs: Any,
    ) -> None:
        """Initializes the MatNetPolicy.

        Args:
            embed_dim: Dimensionality of latent embeddings.
            hidden_dim: Dimensionality of hidden layers.
            problem: Environment or problem logic wrapper.
            num_layers: Number of transformer encoder layers.
            n_heads: Number of attention heads.
            tanh_clipping: Range for logit clipping.
            normalization: Type of layer normalization.
            kwargs: Additional keyword arguments.
        """
        super().__init__(env_name=None)
        self.problem = problem
        self.embed_dim = embed_dim

        self.init_embedding = MatNetInitEmbedding(embed_dim, normalization)

        self.encoder = MatNetEncoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            n_heads=n_heads,
            feed_forward_hidden=hidden_dim,
            normalization=normalization,
        )

        self.decoder = MatNetDecoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            problem=problem,
            n_heads=n_heads,
            tanh_clipping=tanh_clipping,
            **kwargs,
        )

    def set_strategy(self, strategy: str, temp: Optional[float] = None) -> None:
        """Configures the current action selection tactic.

        Args:
            strategy: Identifier for the mode (e.g. 'greedy', 'sampling').
            temp: soft-max temperature for sampling diversity.
        """
        if self.decoder is not None and hasattr(self.decoder, "set_strategy"):
            self.decoder.set_strategy(strategy, temp)

    def forward(
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        strategy: str = "sampling",
        num_starts: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Constructs a solution from the provided matrix data.

        Identifies a cost matrix in `td` (keys: 'dist', 'cost_matrix'),
        encodes it into row/col latents, and construction is handled by the decoder.

        Args:
            td: TensorDict containing the cost matrix and problem data.
            env: Environment managing problem physics.
            strategy: Decoding strategy identifier (e.g., "sampling").
            num_starts: Number of parallel construction starts.
            kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Construction results containing:
                - log_p (torch.Tensor): Cumulative log likelihood.
                - actions (torch.Tensor): Output node indices.

        Raises:
            ValueError: If no cost matrix is found in the input.
            AssertionError: If subnets are not properly initialized.
        """
        # Search for matrix (e.g. distance or cost matrix)
        matrix = td.get("dist")
        if matrix is None:
            matrix = td.get("cost_matrix")

        if matrix is None:
            raise ValueError("MatNetPolicy requires a cost matrix in 'dist' or 'cost_matrix' key.")

        if matrix.dim() == 2:
            matrix = matrix.unsqueeze(0)

        # 1. Initial row and column embeddings from matrix stats
        row_emb, col_emb = self.init_embedding(matrix)

        # 2. Encoding using mixed-score attention
        assert self.encoder is not None, "Encoder is not initialized"
        row_emb, col_emb = self.encoder(row_emb, col_emb, matrix)

        # 3. Decoding
        assert self.decoder is not None, "Decoder is not initialized"
        cost_weights = kwargs.get("cost_weights")
        log_p, actions = self.decoder(td, row_emb, cost_weights=cost_weights, col_embeddings=col_emb, **kwargs)

        return {"log_p": log_p, "actions": actions}
