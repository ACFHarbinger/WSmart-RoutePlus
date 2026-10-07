"""Temporal Attention Model (TAM) core module.

This module provides the implementation of the Temporal Attention Model,
an extension of the base Attention Model designed for dynamic environments like
stochastic waste collection. It integrates a recurrent fill-level predictor
to incorporate historical trends into the routing construction process.

Attributes:
    TemporalAttentionModel: AM variant with proactive temporal features.

Example:
    >>> from logic.src.models.core.temporal_attention_model.model import TemporalAttentionModel
    >>> model = TemporalAttentionModel(embed_dim=128, predictor_type="gru")
    >>> out = model(td, env)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.models.core.attention_model import AttentionModel
from logic.src.models.subnets.factories import NeuralComponentFactory
from logic.src.models.subnets.modules import ActivationFunction
from logic.src.models.subnets.other.gru_fill_predictor import GatedRecurrentUnitFillPredictor
from logic.src.models.subnets.other.lstm_fill_predictor import LongShortTermMemoryFillPredictor


class TemporalAttentionModel(AttentionModel):
    """Attention Model with temporal bin feature prediction.

    TAM improves upon standard AM by leveraging a recurrent sub-network (GRU or LSTM)
    to predict future node states (e.g., bin fill levels) from history. These
    predictions are fused with spatial embeddings before the encoding stage.

    Attributes:
        fill_predictor (nn.Module): Recurrent network for time-series forecasting.
        temporal_horizon (int): Number of historical steps used for prediction.
        temporal_embed (nn.Module): Linear projection for predicted features.
        combine_embeddings (nn.Module): MLP for fusing spatial and temporal latents.
        predict_future (bool): Global toggle for enabling prediction logic.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        problem: Any,
        component_factory: NeuralComponentFactory,
        n_encode_layers: int = 2,
        n_encode_sublayers: Optional[int] = None,
        n_decode_layers: Optional[int] = None,
        dropout_rate: float = 0.1,
        aggregation: str = "sum",
        aggregation_graph: str = "mean",
        tanh_clipping: float = 10.0,
        mask_inner: bool = True,
        mask_logits: bool = True,
        mask_graph: bool = False,
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
        n_heads: int = 8,
        checkpoint_encoder: bool = False,
        shrink_size: Optional[int] = None,
        temporal_horizon: int = 5,
        predictor_layers: int = 2,
        pomo_size: int = 0,
        spatial_bias: bool = False,
        spatial_bias_scale: float = 1.0,
        entropy_weight: float = 0.0,
        connection_type: str = "residual",
        hyper_expansion: int = 4,
        decoder_type: str = "attention",
        predictor_type: str = "gru",
        **kwargs: Any,
    ) -> None:
        """Initializes the Temporal Attention Model.

        Args:
            embed_dim: Dimensionality of latent embeddings.
            hidden_dim: Dimensionality of hidden layers.
            problem: Environment or problem logic wrapper.
            component_factory: Factory for neural building blocks.
            n_encode_layers: Number of transformer encoder layers.
            n_encode_sublayers: Optional internal encoder depth.
            n_decode_layers: Optional decoder layers (if not attention-based).
            dropout_rate: Dropout probability for regularization.
            aggregation: Aggregation mode for local features ("sum", "mean").
            aggregation_graph: Aggregation mode for global features.
            tanh_clipping: Range for logit clipping.
            mask_inner: Whether to mask during attention computation.
            mask_logits: Whether to mask before final softmax.
            mask_graph: Whether to mask nodes in the global context.
            norm_config: Configuration for layer normalization.
            activation_config: Configuration for activation functions.
            n_heads: Number of attention heads.
            checkpoint_encoder: Whether to use gradient checkpointing.
            shrink_size: Optional dimension reduction.
            temporal_horizon: Window size for historical fill prediction.
            predictor_layers: Number of layers in the recurrent predictor.
            pomo_size: Size for POMO-style multi-start augmentation.
            spatial_bias: Whether to inject spatial distance bias.
            spatial_bias_scale: Magnitude of the spatial bias.
            entropy_weight: Weight for entropy regularization.
            connection_type: Type of skip connections ("residual", "dense").
            hyper_expansion: Expansion factor for feed-forward layers.
            decoder_type: Type of constructor ("attention", "pointer").
            predictor_type: Recurrent unit type ("gru", "lstm").
            kwargs: Additional keyword arguments.
        """
        super().__init__(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            problem=problem,
            component_factory=component_factory,
            n_encode_layers=n_encode_layers,
            n_encode_sublayers=n_encode_sublayers,
            n_decode_layers=n_decode_layers,
            dropout_rate=dropout_rate,
            aggregation=aggregation,
            aggregation_graph=aggregation_graph,
            tanh_clipping=tanh_clipping,
            mask_inner=mask_inner,
            mask_logits=mask_logits,
            mask_graph=mask_graph,
            norm_config=norm_config,
            activation_config=activation_config,
            n_heads=n_heads,
            checkpoint_encoder=checkpoint_encoder,
            shrink_size=shrink_size,
            pomo_size=pomo_size,
            temporal_horizon=0,  # Base model horizon is 0; TAM uses fill_predictor
            spatial_bias=spatial_bias,
            spatial_bias_scale=spatial_bias_scale,
            entropy_weight=entropy_weight,
            predictor_layers=predictor_layers,
            connection_type=connection_type,
            hyper_expansion=hyper_expansion,
            decoder_type=decoder_type,
            **kwargs,
        )

        activation_config = activation_config or ActivationConfig()
        self.temporal_horizon = temporal_horizon

        if predictor_type == "lstm":
            self.fill_predictor = LongShortTermMemoryFillPredictor(
                input_dim=1,
                hidden_dim=hidden_dim,
                num_layers=predictor_layers,
                dropout=dropout_rate,
            )
        else:
            assert predictor_type == "gru", f"Unknown predictor type: {predictor_type}"
            self.fill_predictor = GatedRecurrentUnitFillPredictor(  # type: ignore[assignment]
                input_dim=1,
                hidden_dim=hidden_dim,
                num_layers=predictor_layers,
                dropout=dropout_rate,
            )

        self.temporal_embed = nn.Linear(1, embed_dim)
        self.predict_future = self.is_vrpp

        self.combine_embeddings = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            ActivationFunction(
                activation_config.name,
                activation_config.param,
                activation_config.threshold,
                activation_config.replacement_value,
                activation_config.n_params,
                activation_config.range,  # type: ignore[arg-type]
            ),
            nn.Linear(embed_dim, embed_dim),
        )

    def _get_initial_embeddings(self, input: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[Any]]:
        """Processes static and temporal features into fused initial embeddings.

        Identifies 'fill_history' in the input, predicts next-day fill levels,
        and combines these latents with the spatial projections.

        Args:
            input: Key-tensor mapping of instance data including node history.

        Returns:
            Tuple[torch.Tensor, Optional[Any]]:
                - embeddings (torch.Tensor): Fused feature vectors [batch, nodes, dim].
                - init_context (Optional[Any]): Initial state context (None here).
        """
        # 1. Base spatial/constant feature projection
        base_embeddings = self.context_embedder.init_node_embeddings(input)

        if "fill_history" not in input or not self.predict_future:
            return base_embeddings, None

        # 2. Extract and reshape history: [batch, graph, horizon, 1]
        fill_history = input["fill_history"]
        batch_size, graph_size, _ = fill_history.size()
        reshaped_history = fill_history.view(batch_size * graph_size, self.temporal_horizon, 1)

        # 3. Predict next step fill Level
        predicted_fills = self.fill_predictor(reshaped_history)
        predicted_fills = predicted_fills.view(batch_size, graph_size, 1)

        # 4. Handle depot separately (no fill prediction)
        if self.is_vrpp and predicted_fills.size(1) < base_embeddings.size(1):
            depot_fill = torch.zeros((batch_size, 1, 1), device=predicted_fills.device)
            predicted_fills = torch.cat((depot_fill, predicted_fills), dim=1)

        # 5. Fuse spatial and temporal latents
        fill_embeddings = self.temporal_embed(predicted_fills)
        combined_embeddings = self.combine_embeddings(torch.cat((base_embeddings, fill_embeddings), dim=-1))
        return combined_embeddings, None

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
        """Constructs solutions while proactively anticipating future state.

        Args:
            input: Dictionary containing problem instance data.
            env: Environment managing problem physics.
            strategy: Decoding strategy identifier (e.g., "greedy").
            return_pi: Whether to return action probabilities.
            pad: Whether to pad sequences to uniform length.
            mask: Optional node mask for specific constraints.
            expert_pi: Optional teacher distribution for distillation.
            kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Result container (see AttentionModel.forward).
        """
        # Ensure fill_history exists for temporal problems
        if "fill_history" not in input and self.predict_future:
            locs_key = "locs" if "locs" in input else "loc"
            batch_size = input[locs_key].size(0)
            graph_size = input[locs_key].size(1)

            fill_history = torch.zeros(
                (batch_size, graph_size, self.temporal_horizon),
                device=input[locs_key].device,
            )
            input["fill_history"] = fill_history

        # cost_weights legacy mapping
        return super().forward(
            input,
            env=env,
            strategy=strategy,
            return_pi=return_pi,
            pad=pad,
            mask=mask,
            expert_pi=expert_pi,
            **kwargs,
        )

    def update_fill_history(self, fill_history: torch.Tensor, new_fills: torch.Tensor) -> torch.Tensor:
        """Appends new observations to the rolling historical window.

        Args:
            fill_history: Current window [batch, graph, horizon].
            new_fills: Latest observed fill levels [batch, graph].

        Returns:
            torch.Tensor: Updated rolling window.
        """
        updated_history = fill_history.clone()
        updated_history[:, :, :-1] = fill_history[:, :, 1:]
        updated_history[:, :, -1] = new_fills
        return updated_history

    def compute_simulator_day(self, input: Dict[str, Any], graph: Any) -> Dict[str, Any]:
        """Executes one simulation day and updates bin memory.

        Args:
            input: Day-specific simulation data.
            graph: The spatial infrastructure graph.

        Returns:
            Dict[str, Any]: Daily routing accomplishments.
        """
        if "fill_history" in input and "current_fill" in input:
            input["fill_history"] = self.update_fill_history(input["fill_history"], input["current_fill"])
        return super().compute_simulator_day(input, graph)  # type: ignore[misc]
