"""Temporal Attention Model (TAM) Policy.

This module implements the policy for the Temporal Attention Model, which
integrates recurrent fill-level estimation into the constructive routing
framework for RL4CO environments.

Attributes:
    TemporalAMPolicy: Policy wrapper with temporal prediction and fusion capabilities.

Example:
    >>> from logic.src.models.core.temporal_attention_model.policy import TemporalAMPolicy
    >>> policy = TemporalAMPolicy(env_name="ptp", temporal_horizon=5)
    >>> out = policy(td, env)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.core.attention_model.policy import AttentionModelPolicy
from logic.src.models.subnets.modules.activation_function import ActivationFunction
from logic.src.models.subnets.other.gru_fill_predictor import GatedRecurrentUnitFillPredictor
from logic.src.models.subnets.other.lstm_fill_predictor import LongShortTermMemoryFillPredictor
from logic.src.utils.data.td_state_wrapper import TensorDictStateWrapper


class TemporalAMPolicy(AttentionModelPolicy):
    """Routing policy with integrated temporal forecasting.

    Extends canonical attention policies by adding a time-series sub-network
    to predict node fill levels, fusing these predictions with spatial features
    prior to graph encoding.

    Attributes:
        fill_predictor (Union[LSTM, GRU]): History-aware state estimator.
        temporal_embed (nn.Linear): Linear mapping for scalar predictions.
        combine_embeddings (nn.Sequential): MLP for feature fusion.
        temporal_horizon (int): Input history window size.
    """

    fill_predictor: Union[LongShortTermMemoryFillPredictor, GatedRecurrentUnitFillPredictor]
    temporal_embed: nn.Linear
    combine_embeddings: nn.Sequential

    def __init__(
        self,
        env_name: str,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        temporal_horizon: int = 5,
        predictor_layers: int = 2,
        predictor_type: str = "gru",
        **kwargs: Any,
    ) -> None:
        """Initializes the TemporalAMPolicy.

        Args:
            env_name: Name of the environment identifier.
            embed_dim: Dimensionality of latent embeddings.
            hidden_dim: Dimensionality of hidden layers.
            temporal_horizon: Window size for historical fill prediction.
            predictor_layers: Number of layers in the recurrent predictor.
            predictor_type: Recurrent unit type ("gru", "lstm").
            kwargs: Additional keyword arguments for AttentionModelPolicy.
        """
        super().__init__(
            env_name=env_name,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            **kwargs,
        )
        self.temporal_horizon = temporal_horizon

        # Predictive component instantiation
        dropout_rate = kwargs.get("dropout_rate", 0.1)
        if predictor_type == "lstm":
            self.fill_predictor = LongShortTermMemoryFillPredictor(
                input_dim=1,
                hidden_dim=hidden_dim,
                num_layers=predictor_layers,
                dropout=dropout_rate,
            )
        else:
            self.fill_predictor = GatedRecurrentUnitFillPredictor(  # type: ignore[assignment]
                input_dim=1,
                hidden_dim=hidden_dim,
                num_layers=predictor_layers,
                dropout=dropout_rate,
            )

        self.temporal_embed = nn.Linear(1, embed_dim)

        activation_fn = kwargs.get("activation_function", "gelu")
        self.combine_embeddings = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            ActivationFunction(activation_fn),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        strategy: str = "sampling",
        num_starts: int = 1,
        actions: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Performs constructive decoding with proactive state estimation.

        Args:
            td: TensorDict containing problem instance data.
            env: Environment managing problem physics.
            strategy: Decoding strategy identifier (e.g., "sampling").
            num_starts: Number of parallel construction starts.
            actions: Optional pre-defined actions for teacher forcing.
            kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: result dictionary with rewards and constructed actions.

        Raises:
            AssertionError: If encoder/decoder are uninitialized.
        """
        # 1. Handle missing temporal features (cold start)
        if "fill_history" not in td.keys():
            batch_size = td.batch_size[0]
            num_nodes = td["locs"].shape[1]
            td["fill_history"] = torch.zeros(
                (batch_size, num_nodes, self.temporal_horizon, 1),
                device=td.device,
            )

        # 2. Predict next-state features from history
        fill_history = td["fill_history"]  # [batch, nodes, horizon, 1]
        batch_size, num_nodes, horizon, _ = fill_history.size()

        h_flat = fill_history.view(batch_size * num_nodes, horizon, 1)
        predicted_fills = self.fill_predictor(h_flat)
        predicted_fills = predicted_fills.view(batch_size, num_nodes, 1)

        # 3. Form fused initial latent representation
        init_embeds = self.init_embedding(td)
        fill_embeds = self.temporal_embed(predicted_fills)
        combined_embeds = self.combine_embeddings(torch.cat([init_embeds, fill_embeds], dim=-1))

        # 4. Encoding
        edges = td.get("edges")
        assert self.encoder is not None, "Encoder is not initialized"
        embeddings = self.encoder(combined_embeds, edges)

        # 5. Precompute stationary decoder features
        assert self.decoder is not None, "Decoder is not initialized"
        fixed = self.decoder._precompute(embeddings)

        # 6. Sequential constructive loop
        log_likelihood: torch.Tensor | float = 0.0
        entropy: torch.Tensor | float = 0.0
        output_actions = []
        step_idx = 0

        while not td["done"].all():
            assert self.env_name is not None, "env_name must be set"
            state_wrapper = TensorDictStateWrapper(td, self.env_name)
            logits, mask = self.decoder._get_log_p(fixed, state_wrapper)

            # Flatten multi-head logits if applicable
            if logits.dim() == 3:
                logits = logits[:, 0, :]

            # Clean mask for action selection
            if mask.dim() == 3:
                mask = mask.squeeze(1)
            valid_mask = ~mask

            if actions is not None:
                # Teacher forcing / Evaluation
                action = actions[:, step_idx]
                probs = torch.softmax(logits.masked_fill(~valid_mask, float("-inf")), dim=-1)
                log_p = torch.log(probs.gather(1, action.unsqueeze(-1)) + 1e-10).squeeze(-1)
            else:
                # Active strategy selection (sampling/greedy)
                action, log_p, entropy_step = self._select_action(logits, valid_mask, strategy)
                if isinstance(entropy_step, torch.Tensor):
                    entropy = entropy + entropy_step

            # Step environment
            td["action"] = action
            td = env.step(td)["next"].clone()

            log_likelihood = log_likelihood + log_p
            output_actions.append(action)
            step_idx += 1

        # 7. Finalize result
        constructed_actions = torch.stack(output_actions, dim=1)
        reward = env.get_reward(td, constructed_actions)

        return {
            "reward": reward,
            "log_likelihood": log_likelihood,
            "actions": constructed_actions,
            "entropy": entropy,
            "td": td,
        }
