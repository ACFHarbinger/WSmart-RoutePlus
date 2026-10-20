"""Hierarchical Reinforcement Learning Manager for Waste Collection.

This module provides the `MandatoryManager`, a neural agent designed to coordinate
high-level decisions in multi-period routing problems. It integrates spatial
encoding (via GNN/Transformer) with temporal pattern recognition (via LSTM)
to make bipartite decisions: which nodes are mandatory for collection, and
whether to dispatch a fleet at all.

Attributes:
    MandatoryManager: Neural decision agent for hierarchical routing control.

Example:
    None
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.constants.models import (
    DEFAULT_TEMPORAL_HORIZON,
    STATIC_DIM,
)
from logic.src.constants.routing import DEFAULT_EVAL_BATCH_SIZE
from logic.src.constants.waste import CRITICAL_FILL_THRESHOLD
from logic.src.models.subnets.factories import AttentionComponentFactory, NeuralComponentFactory
from logic.src.tracking.logging.pylogger import get_pylogger

from .critic_head import CriticHead
from .gate_head import GateHead
from .mandatory_head import PackingHead
from .temporal_encoder import TemporalEncoder

logger = get_pylogger(__name__)


class MandatoryManager(nn.Module):
    """Hierarchical decision agent for Multi-Period VRP.

    The manager evaluates the current fill levels and historical waste rates
    to determine if a routing pass is necessary and which bins are critical.
    It serves as the upper level in a manager-worker DRL hierarchy.

    Attributes:
        device (str): target hardware.
        batch_size (int): preferred execution batch size.
        embed_dim (int): feature space dimensionality.
        ff_hidden (int): feed-forward intermediate width.
        input_dim_dynamic (int): length of waste history window.
        global_input_dim (int): count of environment-wide features.
        critical_threshold (float): waste level triggering mandatory collection.
        temporal_encoder (nn.Module): RNN/Transformer for waste history.
        feature_embedding (nn.Linear): projection for fused features.
        gat_encoder (nn.Module): spatial context encoder (e.g., GAT).
        mandatory_head (PackingHead): actor for bin categorization.
        gate_head (GateHead): actor for fleet dispatch gating.
        critic (CriticHead): value function for state evaluation.
        states_static (List[torch.Tensor]): history of static features.
        states_dynamic (List[torch.Tensor]): history of dynamic features.
        states_global (List[Optional[torch.Tensor]]): history of global features.
        actions_mandatory (List[torch.Tensor]): history of mandatory selections.
        actions_gate (List[torch.Tensor]): history of dispatch decisions.
        log_probs_mandatory (List[torch.Tensor]): likelihood history for selection.
        log_probs_gate (List[torch.Tensor]): likelihood history for gating.
        rewards (List[float]): PPO rollout rewards.
        values (List[torch.Tensor]): rollout value estimates.
        target_mandatory (List[torch.Tensor]): ground truth mandatory masks.
        dones (List[bool]): rollout termination flags.
    """

    def __init__(
        self,
        input_dim_static: int = STATIC_DIM,
        input_dim_dynamic: int = DEFAULT_TEMPORAL_HORIZON,
        global_input_dim: int = 2,
        critical_threshold: float = CRITICAL_FILL_THRESHOLD,
        batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
        hidden_dim: int = 128,
        embed_dim: Optional[int] = None,
        feed_forward_hidden: Optional[int] = None,
        lstm_hidden: int = 64,
        num_layers_gat: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
        device: str = "cuda",
        shared_encoder: Optional[nn.Module] = None,
        temporal_encoder_cls: Optional[Type[nn.Module]] = None,
        temporal_encoder_kwargs: Optional[Dict[str, Any]] = None,
        spatial_encoder_cls: Optional[Type[nn.Module]] = None,
        spatial_encoder_kwargs: Optional[Dict[str, Any]] = None,
        component_factory: Optional[NeuralComponentFactory] = None,
        temporal_encoder_type: str = "lstm",
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
    ) -> None:
        """Initializes the MandatoryManager.

        Args:
            input_dim_static: raw coordinate dimension (usually 2).
            input_dim_dynamic: window size for historical waste levels.
            global_input_dim: environment-level features (e.g., fleet status).
            critical_threshold: fill level beyond which collection is required.
            batch_size: data processing batch size.
            hidden_dim: legacy embedding dimension.
            embed_dim: explicit embedding dimension.
            feed_forward_hidden: explicit FF hidden width.
            lstm_hidden: feature width for temporal embeddings.
            num_layers_gat: depth of the spatial attention encoder.
            n_heads: attention heads for spatial context.
            dropout: regularization probability.
            device: computation device.
            shared_encoder: pre-trained encoder to reuse.
            temporal_encoder_cls: class for custom temporal processing.
            temporal_encoder_kwargs: params for custom temporal encoder.
            spatial_encoder_cls: class for custom spatial processing.
            spatial_encoder_kwargs: params for custom spatial encoder.
            component_factory: factory for building modular subnets.
            temporal_encoder_type: 'lstm' or 'gru' for default temporal encoder.
            norm_config: normalization layer specification.
            activation_config: non-linear function specification.
        """
        super().__init__()
        self.device = device
        self.batch_size = batch_size

        norm_config = norm_config or NormalizationConfig()
        activation_config = activation_config or ActivationConfig()

        self.embed_dim, self.ff_hidden = self._init_shared_encoder_dim(
            shared_encoder,
            embed_dim or hidden_dim,
            feed_forward_hidden or (embed_dim or hidden_dim) * 4,
        )
        self.input_dim_dynamic = input_dim_dynamic
        self.global_input_dim = global_input_dim
        self.critical_threshold = critical_threshold

        # 1. Temporal Encoder
        self.temporal_encoder = self._init_temporal_encoder(
            temporal_encoder_cls,
            temporal_encoder_kwargs,
            lstm_hidden,
            temporal_encoder_type,
        )

        # 2. Feature Fusion
        # Static (2) + Temporal Embedding (lstm_hidden) -> Hidden
        self.feature_embedding = nn.Linear(input_dim_static + lstm_hidden, self.embed_dim)

        self.gat_encoder = self._init_spatial_encoder(
            shared_encoder=shared_encoder,
            component_factory=component_factory,
            spatial_encoder_cls=spatial_encoder_cls,
            spatial_encoder_kwargs=spatial_encoder_kwargs,
            embed_dim=self.embed_dim,
            feed_forward_hidden=self.ff_hidden,
            num_layers_gat=num_layers_gat,
            n_heads=n_heads,
            dropout=dropout,
            norm_config=norm_config,
            activation_config=activation_config,
        )

        # 4. Heads
        self._init_heads(self.embed_dim, lstm_hidden, global_input_dim)

        self.to(device)

        # Buffers for PPO
        self.clear_memory()

        # Force the Manager to prefer Routing (1) over Skipping (0) at the start
        self._initialize_head_weights()

    def _init_shared_encoder_dim(
        self, shared_encoder: Optional[nn.Module], embed_dim: int, hidden_dim: int
    ) -> Tuple[int, int]:
        """Calculates dimensions based on an existing encoder instance.

        Args:
            shared_encoder: Instance of a pre-defined encoder.
            embed_dim: Value to use if encoder lacks explicit dimension attributes.
            hidden_dim: Value to use if encoder lacks FF dimension attributes.

        Returns:
            Tuple[int, int]: (Calculated embed_dim, Calculated ff_hidden).
        """
        if shared_encoder is not None:
            # Prefer shared encoder attributes if they exist
            e_dim = getattr(shared_encoder, "embed_dim", getattr(shared_encoder, "hidden_dim", embed_dim))
            h_dim = getattr(
                shared_encoder,
                "feed_forward_hidden",
                getattr(shared_encoder, "hidden_dim", hidden_dim),
            )
            return e_dim, h_dim
        return embed_dim, hidden_dim

    def _init_temporal_encoder(
        self,
        temporal_encoder_cls: Optional[Type[nn.Module]] = None,
        temporal_encoder_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden: int = 64,
        temporal_encoder_type: str = "lstm",
    ) -> nn.Module:
        """Constructs the temporal processing subnet.

        Args:
            temporal_encoder_cls: Custom module class.
            temporal_encoder_kwargs: Initializer params for custom class.
            lstm_hidden: dimensional size of the output temporal features.
            temporal_encoder_type: algorithm type ('lstm'/'gru').

        Returns:
            nn.Module: The temporal encoding module.
        """
        if temporal_encoder_cls is not None:
            kwargs = temporal_encoder_kwargs or {}
            return temporal_encoder_cls(**kwargs)

        return TemporalEncoder(hidden_dim=lstm_hidden, rnn_type=temporal_encoder_type)

    def _init_spatial_encoder(
        self,
        shared_encoder: Optional[nn.Module] = None,
        component_factory: Optional[NeuralComponentFactory] = None,
        spatial_encoder_cls: Optional[Type[nn.Module]] = None,
        spatial_encoder_kwargs: Optional[Dict[str, Any]] = None,
        embed_dim: int = 128,
        feed_forward_hidden: int = 512,
        num_layers_gat: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
    ) -> nn.Module:
        """Constructs the spatial context encoder.

        Args:
            shared_encoder: existing instance to assign.
            component_factory: factory for creating a standardized encoder.
            spatial_encoder_cls: custom encoder class.
            spatial_encoder_kwargs: custom encoder params.
            embed_dim: feature dimensionality.
            feed_forward_hidden: expansion layer width.
            num_layers_gat: block count.
            n_heads: attention head count.
            dropout: dropout probability.
            norm_config: normalization layer spec.
            activation_config: activation function spec.

        Returns:
            nn.Module: Spatial encoding module.
        """
        if shared_encoder is not None:
            return shared_encoder

        if component_factory is not None:
            return component_factory.create_encoder(
                norm_config=norm_config,
                activation_config=activation_config,
                embed_dim=embed_dim,
                feed_forward_hidden=feed_forward_hidden,
                n_layers=num_layers_gat,
                n_heads=n_heads,
                dropout_rate=dropout,
            )

        if spatial_encoder_cls is not None:
            kwargs = spatial_encoder_kwargs or {}
            return spatial_encoder_cls(**kwargs)

        factory = AttentionComponentFactory()
        return factory.create_encoder(
            norm_config=norm_config,
            activation_config=activation_config,
            embed_dim=embed_dim,
            feed_forward_hidden=feed_forward_hidden,
            n_layers=num_layers_gat,
            n_heads=n_heads,
            dropout_rate=dropout,
        )

    def _init_heads(self, hidden_dim: int, lstm_hidden: int, global_input_dim: int) -> None:
        """Initializes the three functional heads of the manager.

        Args:
            hidden_dim: width of the fused spatial features.
            lstm_hidden: width of temporal embeddings.
            global_input_dim: count of environmental global features.
        """
        # Packing Head (Actor 1)
        self.mandatory_head = PackingHead(input_dim=hidden_dim + lstm_hidden, hidden_dim=hidden_dim)

        # Route Gate Head (Actor 2)
        self.gate_head = GateHead(input_dim=hidden_dim + global_input_dim, hidden_dim=hidden_dim)

        # Critic (Value Function)
        self.critic = CriticHead(input_dim=hidden_dim + global_input_dim, hidden_dim=hidden_dim)

    def _initialize_head_weights(self) -> None:
        """Biases initial dispatch decisions to promote exploration of routing paths."""
        with torch.no_grad():
            self.gate_head.net[-1].bias.fill_(0)
            self.mandatory_head.net[-1].bias.fill_(0)

    def clear_memory(self) -> None:
        """Flushes PPO trajectory buffers after an update."""
        self.states_static: List[torch.Tensor] = []
        self.states_dynamic: List[torch.Tensor] = []
        self.states_global: List[Optional[torch.Tensor]] = []
        self.actions_mandatory: List[torch.Tensor] = []
        self.actions_gate: List[torch.Tensor] = []
        self.log_probs_mandatory: List[torch.Tensor] = []
        self.log_probs_gate: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.values: List[torch.Tensor] = []
        self.target_mandatory: List[torch.Tensor] = []
        self.dones: List[bool] = []

    def feature_processing(self, static: torch.Tensor, dynamic: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuses static coordinates with temporal waste embeddings.

        Args:
            static: Static features [Batch, N, 2].
            dynamic: Dynamic features [Batch, N, History].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - x: Combined features projected to hidden dimension [Batch, N, Hidden].
                - temporal_embed: Temporal embeddings from RNN [Batch, N, lstm_hidden].
        """
        temporal_embed = self.temporal_encoder(dynamic)
        combined = torch.cat([static, temporal_embed], dim=2)
        x = self.feature_embedding(combined)
        return x, temporal_embed

    def forward(
        self,
        static: torch.Tensor,
        dynamic: torch.Tensor,
        global_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass analyzing temporal-spatial state.

        Args:
            static: coordinates [B, N, 2].
            dynamic: waste history [B, N, H].
            global_features: environment status [B, G].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - mandatory_logits: Categorical logits for packing [B, N, 2].
                - gate_logits: Categorical logits for dispatch toggle [B, 2].
                - value: State value estimation [B, 1].
        """
        # Encode
        x, temporal_embed = self.feature_processing(static, dynamic)

        # Spatial Context
        h = self.gat_encoder(x)

        # SKIP CONNECTION: Concatenate temporal_embed (LSTM output) to h
        h_skip = torch.cat([h, temporal_embed], dim=-1)

        # Packing Logits
        mandatory_logits = self.mandatory_head(h_skip)

        # Global Pooling for Gate/Critic
        h_mean = torch.mean(h, dim=1)
        h_max = torch.max(h, dim=1)[0]
        h_global = h_mean + h_max

        # Concatenate Global Features
        h_combined = torch.cat([h_global, global_features], dim=1)

        # Gate Logits
        gate_logits = self.gate_head(h_combined)

        # Value
        value = self.critic(h_combined)

        return mandatory_logits, gate_logits, value

    def select_action(
        self,
        static: torch.Tensor,
        dynamic: torch.Tensor,
        global_features: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        threshold: float = 0.5,
        mandatory_threshold: float = 0.5,
        target_mandatory: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Determines management actions and buffers results for training.

        Args:
            static: static coordinates.
            dynamic: waste history.
            global_features: env-wide status.
            deterministic: if true, uses argmax selection.
            threshold: probability cutoff for route dispatch.
            mandatory_threshold: probability cutoff for packing.
            target_mandatory: explicit labels for supervised training components.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - mandatory_action: bool mask of required collections [B, N].
                - gate_action: int dispatch toggle [B].
                - value: critic evaluation [B, 1].
        """
        if global_features is None:
            # Fallback for missing global features
            global_features = torch.zeros(static.size(0), self.global_input_dim).to(static.device)

        mandatory_logits, gate_logits, value = self.forward(static, dynamic, global_features)

        # 1. Gate Decision (0=Skip routing, 1=Dispatch vehicles)
        gate_probs = F.softmax(gate_logits, dim=-1)
        gate_dist = torch.distributions.Categorical(gate_probs)

        if deterministic:
            if threshold < 0:
                gate_action = torch.ones_like(gate_probs[:, 1]).long()
            else:
                gate_action = (gate_probs[:, 1] > threshold).long()
        else:
            gate_action = gate_dist.sample()

        # 2. Packing (Which bins must be collected)
        mandatory_probs = F.softmax(mandatory_logits, dim=-1)
        mandatory_dist = torch.distributions.Categorical(mandatory_probs)

        if deterministic:
            mandatory_action = (mandatory_probs[:, :, 1] > mandatory_threshold).long()
        else:
            mandatory_action = mandatory_dist.sample()

        # Store for PPO
        if not deterministic:
            log_prob_mandatory = mandatory_dist.log_prob(mandatory_action).sum(dim=1)
            log_prob_gate = gate_dist.log_prob(gate_action)

            self.states_static.append(static.detach().cpu())
            self.states_dynamic.append(dynamic.detach().cpu())
            self.states_global.append(global_features.detach().cpu() if global_features is not None else None)
            self.actions_gate.append(gate_action.detach().cpu())
            self.actions_mandatory.append(mandatory_action.detach().cpu())
            self.log_probs_gate.append(log_prob_gate.detach().cpu())
            self.log_probs_mandatory.append(log_prob_mandatory.detach().cpu())
            self.values.append(value.detach().cpu())

            # Store target mandatory for auxiliary loss
            if target_mandatory is not None:
                self.target_mandatory.append(target_mandatory.detach().cpu())
            else:
                # Fallback: bins with fill > critical_threshold should be collected
                self.target_mandatory.append((dynamic[:, :, -1] > self.critical_threshold).float().detach().cpu())

        return mandatory_action, gate_action, value

    def get_mandatory_mask(
        self,
        static: torch.Tensor,
        dynamic: torch.Tensor,
        global_features: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Retrieves a binary mask for nodes requiring collection.

        Args:
            static: static coordinates.
            dynamic: waste history.
            global_features: context status.
            threshold: categorization probability cutoff.

        Returns:
            torch.Tensor: Boolean tensor where True marks mandatory nodes [B, N].
        """
        if global_features is None:
            global_features = torch.zeros(static.size(0), self.global_input_dim).to(static.device)

        mandatory_logits, _, _ = self.forward(static, dynamic, global_features)
        mandatory_probs = F.softmax(mandatory_logits, dim=-1)

        # mandatory is True where P(mandatory=1) > threshold
        mandatory = mandatory_probs[:, :, 1] > threshold

        # Depot (index 0) should never be a mandatory target
        mandatory[:, 0] = False

        return mandatory
