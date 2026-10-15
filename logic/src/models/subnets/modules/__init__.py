"""Neural network modules and building blocks.

This package contains reusable layers and sub-networks used to build complex
VRP models, including attention mechanisms, graph convolutions, and custom
normalization layers.

Attributes:
    ActivationFunction: Configurable activation factory.
    MultiHeadAttention: Foundation for neural routing encoders.
    MultiHeadCrossAttention: Cross-modal attention mechanism.
    MultiHeadFlashAttention: Memory-efficient attention implementation.
    GraphConvolution: Standard spatial aggregation layers.
    GatedGraphConvolution: Gated edge-weighted aggregation.
    EfficientGraphConvolution: Optimized multi-head graph layer.
    FeedForward: Standard positional-wise transformation.
    MPNNEncoder: Message Passing Neural Network sequence encoder.
    Normalization: Unified wrapper for Layer/Instance/Batch normalization.
    SkipConnection: Residual and hyper-network wrappers.
    DynamicHyperConnection: Adaptive state-dependent skip mechanism.
    StaticHyperConnection: Fixed weight-sharing skip mechanism.

Example:
    >>> from logic.src.models.subnets.modules import MultiHeadAttention
    >>> mha = MultiHeadAttention(embed_dim=128, n_heads=8)
"""

from typing import Any

# Lazy re-exports (P-25): importing this package does not pull every encoder;
# each name loads its module on first access.
_LAZY = {
    "ActivationFunction": "logic.src.models.subnets.modules.activation_function:ActivationFunction",
    "FeedForward": "logic.src.models.subnets.modules.feed_forward:FeedForward",
    "GatedGraphConvolution": "logic.src.models.subnets.modules.gated_graph_convolution:GatedGraphConvolution",
    "GraphConvolution": "logic.src.models.subnets.modules.graph_convolution:GraphConvolution",
    "MPNNEncoder": "logic.src.models.subnets.modules.mpnn_encoder:MPNNEncoder",
    "MessagePassingLayer": "logic.src.models.subnets.modules.mpnn_layer:MessagePassingLayer",
    "MixedScoreMHA": "logic.src.models.subnets.modules.matnet_attention:MixedScoreMHA",
    "MultiHeadAttention": "logic.src.models.subnets.modules.multi_head_attention:MultiHeadAttention",
    "Normalization": "logic.src.models.subnets.modules.normalization:Normalization",
    "PointerAttnMoE": "logic.src.models.subnets.modules.pointer_attn_moe:PointerAttnMoE",
    "PolyNetAttention": "logic.src.models.subnets.modules.polynet_attention:PolyNetAttention",
    "SkipConnection": "logic.src.models.subnets.modules.skip_connection:SkipConnection",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        module_name, _, attr = _LAZY[name].partition(":")
        module = importlib.import_module(module_name)
        obj = getattr(module, attr)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ActivationFunction",
    "MultiHeadAttention",
    "MultiHeadCrossAttention",
    "MultiHeadFlashAttention",
    "GraphConvolution",
    "GatedGraphConvolution",
    "EfficientGraphConvolution",
    "FeedForward",
    "MPNNEncoder",
    "Normalization",
    "SkipConnection",
    "DynamicHyperConnection",
    "StaticHyperConnection",
    "MixedScoreMHA",
    "MultiHeadAttentionMDAM",
    "MessagePassingLayer",
    "PointerAttnMoE",
    "PolyNetAttention",
]
