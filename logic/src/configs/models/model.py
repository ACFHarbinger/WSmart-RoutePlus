"""
Model Config module.

Attributes:
    ModelConfig: Model architecture configuration.

Example:
    >>> from logic.src.configs.models import ModelConfig
    >>> config = ModelConfig()
    >>> print(config)
    ModelConfig(name='am', encoder=<EncoderConfig(type='gat', embed_dim=128, hidden_dim=512, n_layers=3, n_heads=8, n_sublayers=None, normalization=<NormalizationConfig(embed_dim=128, use_layer_norm=True)>, activation=<ActivationConfig(type='gelu')>, dropout=0.1, mask_inner=True, mask_graph=False, spatial_bias=False, connection_type='residual', aggregation_graph='avg', aggregation_node='sum', spatial_bias_scale=1.0, hyper_expansion=4)>, decoder=<DecoderConfig(type='attention', embed_dim=128, hidden_dim=512, n_layers=3, n_heads=8, normalization=<NormalizationConfig(embed_dim=128, use_layer_norm=True)>, activation=<ActivationConfig(type='gelu')>, decoding=<DecodingConfig(strategy='greedy', beam_width=1, temperature=1.0, top_k=None, top_p=None, tanh_clipping=0.0, mask_logits=True, multistart=False, num_starts=1, select_best=False)>, dropout=0.1, mask_logits=True, connection_type='residual', n_predictor_layers=None, tanh_clipping=10.0, hyper_expansion=4)>, reward=<ObjectiveConfig(type='ptp')>, temporal_horizon=0, policy_config=None, load_path=None)
"""

from dataclasses import dataclass, field
from typing import Optional

from logic.src.configs.models.decoder import DecoderConfig
from logic.src.configs.models.encoder import EncoderConfig


@dataclass
class ModelConfig:
    """Model architecture configuration.

    Attributes:
        name: Name of the model architecture (e.g., 'am', 'deep_decoder').
        encoder: Encoder configuration.
        decoder: Decoder configuration.
        temporal_horizon: Temporal horizon for the model.
        policy_config: Policy configuration string.
    """

    name: str = "am"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    temporal_horizon: int = 0
    policy_config: Optional[str] = None
    load_path: Optional[str] = None
