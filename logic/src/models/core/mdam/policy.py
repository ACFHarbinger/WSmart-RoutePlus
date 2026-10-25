"""MDAM Policy: Multi-Path Construction Policy.

This module implements `MDAMPolicy`, which uses a shared graph encoder and multiple
parallel decoder paths to generate a diverse set of solutions for a single
problem instance.

Attributes:
    MDAMPolicy: Autoregressive policy with multi-decoder expansion.

Example:
    >>> policy = MDAMPolicy(env_name="ptp", num_paths=5)
    >>> out = policy(td, env)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from tensordict import TensorDict

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.autoregressive.policy import AutoregressivePolicy
from logic.src.models.subnets.decoders.mdam import MDAMDecoder
from logic.src.models.subnets.embeddings import get_init_embedding
from logic.src.models.subnets.encoders.mdam.encoder import MDAMGraphAttentionEncoder


class MDAMPolicy(AutoregressivePolicy):
    """Multi-Decoder Attention Model (MDAM) Policy.

    Generates diverse solutions by branching the construction process into
    multiple independent paths (decoders). Employs KL-divergence as an auxiliary
    loss during training to discourage decoder mode collapse.

    Attributes:
        embed_dim (int): feature vector width.
        env_name (str): optimization task identifier.
        num_paths (int): count of parallel decoders.
        init_embedding (nn.Module): initial feature projector.
        train_strategy (str): decoding mode for training.
        val_strategy (str): decoding mode for validation.
        test_strategy (str): decoding mode for testing.
        encoder (MDAMGraphAttentionEncoder): shared graph context extractor.
        decoder (MDAMDecoder): multi-path construction unit.
    """

    def __init__(
        self,
        encoder: Optional[MDAMGraphAttentionEncoder] = None,
        decoder: Optional[MDAMDecoder] = None,
        embed_dim: int = 128,
        env_name: str = "ptp",
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        num_paths: int = 5,
        normalization: str = "batch",
        train_strategy: str = "sampling",
        val_strategy: str = "greedy",
        test_strategy: str = "greedy",
        **decoder_kwargs: Any,
    ) -> None:
        """Initializes the MDAM policy.

        Args:
            encoder: Optional pre-instantiated MDAM encoder.
            decoder: Optional pre-instantiated MDAM decoder.
            embed_dim: Dimensionality of latent embeddings.
            env_name: Name of the environment identifier.
            num_encoder_layers: Number of transformer encoder layers.
            num_heads: Number of attention heads.
            num_paths: Number of parallel decoder paths.
            normalization: Type of layer normalization ("batch", "instance").
            train_strategy: Default strategy for training.
            val_strategy: Default strategy for validation.
            test_strategy: Default strategy for testing.
            decoder_kwargs: Additional keyword arguments for the decoder.
        """
        if encoder is None:
            encoder = MDAMGraphAttentionEncoder(
                num_heads=num_heads,
                embed_dim=embed_dim,
                num_layers=num_encoder_layers,
                normalization=normalization,
            )

        if decoder is None:
            decoder = MDAMDecoder(
                env_name=env_name,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_paths=num_paths,
                **decoder_kwargs,
            )

        super().__init__(env_name=env_name, encoder=encoder, decoder=decoder)  # type: ignore[arg-type]

        self.embed_dim = embed_dim
        self.env_name = env_name
        self.num_paths = num_paths

        self.init_embedding = get_init_embedding(env_name, embed_dim)

        self.train_strategy = train_strategy
        self.val_strategy = val_strategy
        self.test_strategy = test_strategy

    def forward(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        strategy: str = "sampling",
        num_starts: int = 1,
        phase: str = "train",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Calculates diverse construction paths through the multi-decoder.

        Args:
            td: TensorDict containing problem instance data.
            env: Environment managing problem physics.
            strategy: Decoding strategy identifier (e.g., "sampling").
            num_starts: Number of parallel construction starts.
            phase: Current execution phase ("train", "val", "test").
            kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Result map including:
                - reward: Tensor of rewards per path [B, num_paths].
                - log_likelihood: Cumulative log probabilities [B, num_paths].
                - kl_divergence: Diversity auxiliary score.
                - actions: Constructive sequences [B, num_paths, SeqLen].
        """
        embedding = self.init_embedding(td)

        # Shared encoding phase
        encoded_inputs, graph_embed, attn, V, h_old = self.encoder(td, x=embedding)  # type: ignore[misc]

        # Multi-path decoding phase
        log_p, actions, reward, kl_divergence = self.decoder(  # type: ignore[misc]
            td,
            (encoded_inputs, graph_embed, attn, V, h_old),
            env,
            self.encoder,
            strategy=strategy,
            num_starts=num_starts,
            **kwargs,
        )

        return {
            "reward": reward,
            "log_likelihood": log_p,
            "kl_divergence": kl_divergence,
            "actions": actions,
        }
