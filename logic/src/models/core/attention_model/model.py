"""Attention Model (AM) core module.

This module provides the implementation of the Attention Model (Kool et al. 2019),
a graph-based neural network that uses multi-head attention to constructively
solve Vehicle Routing Problems. It supports various problem domains
including TSP, VRPP, and WCVRP.

Attributes:
    AttentionModel: The primary constructive neural routing policy.

Example:
    >>> from logic.src.models.core.attention_model.model import AttentionModel
    >>> model = AttentionModel(embed_dim=128, hidden_dim=512, problem=env.problem, ...)
    >>> out = model(td, env, strategy="greedy")
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union, cast

import torch
import torch.utils.checkpoint
from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.constants.models import (
    FEED_FORWARD_EXPANSION,
    NODE_DIM,
    TANH_CLIPPING,
)
from logic.src.interfaces.tensor_dict_like import ITensorDictLike
from logic.src.models.subnets.embeddings import (
    GenericContextEmbedder,
    VRPPContextEmbedder,
    WCVRPContextEmbedder,
)
from logic.src.models.subnets.factories import NeuralComponentFactory
from logic.src.utils.functions.problem import is_tsp_problem, is_vrpp_problem, is_wc_problem

from .decoding import DecodingMixin


class AttentionModel(DecodingMixin, nn.Module):
    """Attention Model for neural combinatorial optimization.

    This model implements an Encoder-Decoder architecture where the encoder
    processes node features into a latent graph representation, and the decoder
    sequentially constructs routes by attending to the encoded nodes and
    the current search state.

    Attributes:
        embed_dim (int): Dimensionality of node and graph embeddings.
        hidden_dim (int): Hidden dimension for feed-forward sublayers.
        n_heads (int): Number of multi-head attention heads.
        problem (Any): Problem definition object (e.g., VRPP instance).
        encoder (nn.Module): The graph attention encoder.
        decoder (nn.Module): The autoregressive attention decoder.
        context_embedder (nn.Module): Initial feature projection module.
        pomo_size (int): Parallel start size for POMO (if enabled).
        checkpoint_encoder (bool): Whether to use gradient checkpointing on encoding.
        aggregation_graph (str): Global aggregation method ('avg', 'max', 'sum').
        temporal_horizon (int): Number of future steps to consider for dynamics.
        tanh_clipping (float): Logit clipping value for stable training.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        problem: Any,
        component_factory: NeuralComponentFactory,
        n_encode_layers: int = 3,
        n_encode_sublayers: Optional[int] = None,
        n_decode_layers: Optional[int] = None,
        dropout_rate: float = 0.1,
        aggregation: str = "sum",
        aggregation_graph: str = "avg",
        tanh_clipping: float = TANH_CLIPPING,
        mask_inner: bool = True,
        mask_logits: bool = True,
        mask_graph: bool = False,
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
        n_heads: int = 8,
        checkpoint_encoder: bool = False,
        shrink_size: Optional[int] = None,
        pomo_size: int = 0,
        temporal_horizon: int = 0,
        spatial_bias: bool = False,
        spatial_bias_scale: float = 1.0,
        entropy_weight: float = 0.0,
        predictor_layers: Optional[int] = None,
        connection_type: str = "residual",
        hyper_expansion: int = FEED_FORWARD_EXPANSION,
        decoder_type: str = "attention",
        **kwargs: Any,
    ) -> None:
        """Initializes the multi-head Attention Model.

        Args:
            embed_dim: Dimensionality of node and graph embeddings.
            hidden_dim: Hidden dimension for feed-forward sublayers.
            problem: Domain problem instance definition.
            component_factory: Factory for building encoder/decoder subnets.
            n_encode_layers: Number of Transformer encoder layers.
            n_encode_sublayers: Optional sub-depth override for encoder blocks.
            n_decode_layers: Number of Transformer decoder layers.
            dropout_rate: Dropout probability.
            aggregation: GNN node pooling type for local features.
            aggregation_graph: Global graph pooling strategy ("avg", "sum", "max").
            tanh_clipping: Logit clipping value for stable training.
            mask_inner: Enable masking inside transformer blocks.
            mask_logits: Enable node visit masking in final softmax.
            mask_graph: Global graph structure masking flag.
            norm_config: Layer normalization configuration.
            activation_config: Activation function configuration.
            n_heads: Number of attention heads.
            checkpoint_encoder: Whether to use gradient checkpointing on encoding.
            shrink_size: Compression factor for efficient decoders.
            pomo_size: Parallel start size for POMO.
            temporal_horizon: Look-ahead depth for dynamic problems.
            spatial_bias: Enable relative distance injection.
            spatial_bias_scale: Multiplier for spatial bias.
            entropy_weight: Weight for entropy exploration bonus.
            predictor_layers: MLP depth for final logit prediction.
            connection_type: Skip connection architecture style.
            hyper_expansion: Expansion factor for hidden MLPs.
            decoder_type: Architectural style of the decoder ("attention", etc.).
            kwargs: Additional keyword arguments.
        """
        nn.Module.__init__(self)
        DecodingMixin.__init__(self)

        norm_config = norm_config or NormalizationConfig()
        activation_config = activation_config or ActivationConfig()

        self._init_parameters(
            embed_dim,
            hidden_dim,
            problem,
            n_heads,
            pomo_size,
            checkpoint_encoder,
            aggregation_graph,
            temporal_horizon,
            tanh_clipping,
        )

        self._init_context_embedder(temporal_horizon)

        # Baseline step context dim matches graph latent size
        step_context_dim = 2 * embed_dim + embed_dim

        self._init_components(
            component_factory,
            step_context_dim,
            n_encode_layers,
            n_encode_sublayers,
            n_decode_layers,
            norm_config,
            activation_config,
            dropout_rate,
            aggregation,
            hyper_expansion,
            connection_type,
            predictor_layers,
            tanh_clipping,
            mask_inner,
            mask_logits,
            mask_graph,
            shrink_size,
            spatial_bias,
            spatial_bias_scale,
            decoder_type,
        )

        self.set_strategy("greedy")

    def _init_parameters(
        self,
        embed_dim: int,
        hidden_dim: int,
        problem: Any,
        n_heads: int,
        pomo_size: int,
        checkpoint_encoder: bool,
        aggregation_graph: str,
        temporal_horizon: int,
        tanh_clipping: float,
    ) -> None:
        """Helper to initialize class member parameters.

        Args:
            embed_dim: Embedding size.
            hidden_dim: Subnet hidden size.
            problem: Domain problem object.
            n_heads: Attention head count.
            pomo_size: POMO parallel construction count.
            checkpoint_encoder: Encoder checkpointing flag.
            aggregation_graph: Pooling strategy identifier.
            temporal_horizon: Look-ahead depth.
            tanh_clipping: Softmax clipping constant.
        """
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.problem = problem
        self.pomo_size = pomo_size
        self.checkpoint_encoder = checkpoint_encoder
        self.aggregation_graph = aggregation_graph
        self.temporal_horizon = temporal_horizon
        self.tanh_clipping = tanh_clipping

    def _init_context_embedder(self, temporal_horizon: int) -> None:
        """Initializes the problem-specific initial projection network.

        Args:
            temporal_horizon: Look-ahead steps for dynamic features.
        """
        if is_vrpp_problem(self.problem):
            self.context_embedder = VRPPContextEmbedder(self.embed_dim, temporal_horizon=self.temporal_horizon)
        elif is_wc_problem(self.problem):
            self.context_embedder = WCVRPContextEmbedder(self.embed_dim, temporal_horizon=self.temporal_horizon)  # type: ignore[assignment]
        else:
            node_dim = 2 if is_tsp_problem(self.problem) else NODE_DIM
            self.context_embedder = GenericContextEmbedder(  # type: ignore[assignment]
                self.embed_dim,
                node_dim=node_dim,
                temporal_horizon=self.temporal_horizon,
            )

    @property
    def is_vrpp(self) -> bool:
        """Determines if the model is configured for VRP with Profits.

        Returns:
            bool: True if the problem is a VRPP variant.
        """
        return is_vrpp_problem(self.problem)

    @property
    def is_wc(self) -> bool:
        """Determines if the model is configured for Waste Collection.

        Returns:
            bool: True if the problem is a WCVRP variant.
        """
        return is_wc_problem(self.problem)

    def _init_components(
        self,
        component_factory: NeuralComponentFactory,
        step_context_dim: int,
        n_encode_layers: int,
        n_encode_sublayers: Optional[int],
        n_decode_layers: Optional[int],
        norm_config: NormalizationConfig,
        activation_config: ActivationConfig,
        dropout_rate: float,
        aggregation: str,
        hyper_expansion: int,
        connection_type: str,
        predictor_layers: Optional[int],
        tanh_clipping: float,
        mask_inner: bool,
        mask_logits: bool,
        mask_graph: bool,
        shrink_size: Optional[int],
        spatial_bias: bool,
        spatial_bias_scale: float,
        decoder_type: str = "attention",
    ) -> None:
        """Initializes the graph encoder and sequential decoder using the factory.

        Args:
            component_factory: The object responsible for building subnets.
            step_context_dim: Input size for decoder's current search state.
            n_encode_layers: Graph attention encoder depth.
            n_encode_sublayers: Optional sub-depth override for encoder blocks.
            n_decode_layers: Autoregressive decoder depth.
            norm_config: Layer normalization configuration.
            activation_config: Activation function configuration.
            dropout_rate: Network dropout probability.
            aggregation: GNN node pooling type.
            hyper_expansion: Internal MLP expansion factor.
            connection_type: Skip connection architecture.
            predictor_layers: MLP depth for final logit prediction.
            tanh_clipping: Value of TANH clipping for attention.
            mask_inner: Enable masking inside transformer blocks.
            mask_logits: Enable node visit masking in final softmax.
            mask_graph: Global graph structure masking flag.
            shrink_size: Compression factor for efficient decoders.
            spatial_bias: Enable relative distance distance injection.
            spatial_bias_scale: Multiplier for distance bias.
            decoder_type: Architectural style of the decoder.
        """
        self.encoder = component_factory.create_encoder(
            embed_dim=self.embed_dim,
            feed_forward_hidden=self.hidden_dim,  # the factory default (512) ignored hidden_dim
            n_layers=n_encode_layers,
            n_sublayers=n_encode_sublayers,
            norm_config=norm_config,
            activation_config=activation_config,
            dropout_rate=dropout_rate,
            aggregation=aggregation,
            hyper_expansion=hyper_expansion,
            connection_type=connection_type,
            n_heads=self.n_heads,
            mask_inner=mask_inner,
            mask_graph=mask_graph,
            spatial_bias=spatial_bias,
            spatial_bias_scale=spatial_bias_scale,
        )

        self.decoder = component_factory.create_decoder(
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            problem=self.problem,
            n_layers=n_decode_layers,
            n_heads=self.n_heads,
            step_context_dim=step_context_dim,
            predictor_layers=predictor_layers,
            norm_config=norm_config,
            activation_config=activation_config,
            dropout_rate=dropout_rate,
            aggregation=aggregation,
            hyper_expansion=hyper_expansion,
            connection_type=connection_type,
            tanh_clipping=tanh_clipping,
            mask_logits=mask_logits,
            shrink_size=shrink_size,
            decoder_type=decoder_type,
        )

    def _get_initial_embeddings(self, input: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Processes raw instance data into initial node embeddings.

        Args:
            input: Key-tensor mapping of instance data (e.g., node coordinates).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - embeddings (torch.Tensor): Initial Projected feature vectors.
                - init_context (Optional[torch.Tensor]): Global problem context if any.
        """
        return self.context_embedder(input), None

    def forward(
        self,
        input: Dict[str, torch.Tensor],
        env: Optional[Any] = None,
        strategy: Optional[str] = None,
        return_pi: bool = False,
        pad: bool = False,
        mask: Optional[torch.Tensor] = None,
        expert_pi: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Executes the complete constructive search pipeline.

        Performs: (1) Initial embedding, (2) Graph encoding via Attention Encoder,
        (3) Global context aggregation, and (4) Sequential construction via Decoder.

        Args:
            input: Map of instance feature tensors.
            env: Environment managing problem physics.
            strategy: Decoding strategy identifier (e.g., "sampling").
            return_pi: Whether to include actions in the output map.
            pad: Whether to pad sequences to a fixed length.
            mask: Optional tensor of invalid nodes.
            expert_pi: Optional ground-truth actions for imitation learning.
            kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Construction results containing:
                - cost (torch.Tensor): Path lengths of the routes.
                - reward (torch.Tensor): Normalized negative costs for RL.
                - log_likelihood (torch.Tensor): Cumulative log prob of solution.
                - actions (torch.Tensor, optional): The node sequences.
                - td (TensorDict, optional): Final environment state.
        """
        # Ensure embedding init if dynamic projection is needed
        if not hasattr(self, "project_node_embeddings"):
            pass

        # 1. Project raw features to latent space
        embeddings, init_context = self._get_initial_embeddings(input)

        # 2. Encode graph features sequentially or via checkpointing
        if self.checkpoint_encoder and self.training:
            outputs = cast(
                torch.Tensor,
                torch.utils.checkpoint.checkpoint(self.encoder, embeddings, mask, use_reentrant=False),
            )
        else:
            outputs = self.encoder(embeddings, mask)

        # 3. Pool node embeddings for global graph context
        graph_context = self._aggregate_graph_context(outputs)

        # 4. Construct solution step-by-step
        out_dec = self.decoder(
            input,
            outputs,
            graph_context,
            init_context,
            env,
            strategy=strategy or self.strategy,
            return_pi=return_pi,
            expert_pi=expert_pi,
        )

        # Result packaging (handling variant decoder return structures)
        if isinstance(out_dec, tuple):
            if len(out_dec) == 4:
                _log_p, pi, cost, final_td = out_dec
            elif len(out_dec) == 3:
                _log_p, pi, cost = out_dec
                final_td = None
            else:
                _log_p = out_dec[0]
                pi = out_dec[1] if len(out_dec) > 1 else None
                cost = out_dec[2] if len(out_dec) > 2 else None
                final_td = out_dec[3] if len(out_dec) > 3 else None
        else:
            _log_p = out_dec
            pi = None
            cost = None
            final_td = None

        reward = -cost if cost is not None else torch.tensor(0.0, device=outputs.device)
        out = {"cost": cost, "reward": reward, "td": final_td}

        if _log_p is not None:
            out["log_likelihood"] = _log_p
            out["log_p"] = _log_p

        if pi is not None:
            out["actions"] = pi
            if return_pi:
                out["pi"] = pi

        return out

    def _aggregate_graph_context(self, outputs: torch.Tensor) -> torch.Tensor:
        """Aggregates per-node embeddings into a summary graph-level representation.

        Args:
            outputs: Encoded node embeddings [batch, nodes, dim].

        Returns:
            torch.Tensor: Pooled graph context vector [batch, dim].
        """
        if self.aggregation_graph == "avg":
            return outputs.mean(dim=1)
        if self.aggregation_graph == "max":
            return outputs.max(dim=1)[0]
        if self.aggregation_graph == "sum":
            return outputs.sum(dim=1)
        return outputs.mean(dim=1)

    def precompute_fixed(self, input: Dict[str, torch.Tensor], edges: Optional[torch.Tensor] = None) -> Any:
        """Precomputes and caches the graph-level state for efficient search.

        Used primarily in beam search or local search, where the graph embeddings
        remain constant across many iterations.

        Args:
            input: Raw problem instance data.
            edges: Optional graph connectivity information.

        Returns:
            CachedLookup: A lookup object for the pre-encoded graph state.
        """
        from logic.src.utils.decoding import CachedLookup

        embeddings, init_context = self._get_initial_embeddings(input)
        # Apply encoder (GNN/GAT) to get contextual embeddings
        encoded_embeddings = self.encoder(embeddings)
        # Use encoded embeddings for decoder pre-computation
        out = self.decoder(input, encoded_embeddings, init_context, None, precompute_only=True)

        if isinstance(out, tuple):
            if len(out) >= 3:
                _log_p, _pi, _cost = out[:3]
            else:
                _log_p = out[0]
        else:
            _log_p = out

        return CachedLookup(embeddings=encoded_embeddings, context=init_context)

    def expand(self, t: Union[torch.Tensor, ITensorDictLike, None]) -> Any:
        """Expands instance features to support parallel POMO constructions.

        Repeats batch elements to create multiple starts per instance (e.g.,
        one starting at each possible node).

        Args:
            t: Tensor, TensorDict, or dictionary to expand.

        Returns:
            Any: The expanded container with [batch * pomo_size, ...] shape.
        """
        if t is None:
            return None
        if isinstance(t, ITensorDictLike):
            return t.__class__({k: self.expand(v) for k, v in t.items()})  # type: ignore[call-arg]

        # Expand (Batch, ...) -> (Batch * POMO, ...)
        bs = t.size(0)
        shape = (bs, self.pomo_size) + t.shape[1:]
        return t.unsqueeze(1).expand(shape).reshape(-1, *t.shape[1:])
