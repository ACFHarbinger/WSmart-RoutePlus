"""Encoder Subnetworks.

Attributes:
    TransformerEncoderBase: Abstract base class for transformer-style graph encoders.
    EncoderFeedForwardSubLayer: Reusable Feed-Forward Sub-Layer.
    MultiHeadAttentionLayerBase: Base class for multi-head attention layers.
    DeepACOEncoder: DeepACO-style graph encoder.
    GraphAttConvEncoder: Graph Attention Convolution Encoder.
    GraphAttentionEncoder: Standard Graph Attention Encoder.
    GraphConvolutionEncoder: Gated Graph Convolution Encoder.
    GFACSEncoder: GFlowNet-ACO Heatmap Predictor Encoder.
    GatedGraphAttConvEncoder: Gated Graph Attention Convolution Encoder.
    MatNetEncoder: Matrix Encoding Network Encoder.
    MDAMGraphAttentionEncoder: Multi-Decoder Attention Model Encoder.
    MLPEncoder: Multilayer Perceptron Encoder.
    MoEGraphAttentionEncoder: Mixture-of-Experts Graph Attention Encoder.
    NARGNNEncoder: Non-Autoregressive Graph Neural Network Encoder.
    NARGNNNodeEncoder: Node-only NARGNN Encoder.
    PointerEncoder: Pointer Network Encoder.
    TransGraphConvEncoder: Transformer-based Graph Convolution Encoder.

Example:
    >>> from logic.src.models.subnets.encoders import MLPEncoder
    >>> encoder = MLPEncoder(embed_dim=128, n_layers=3)
"""

from typing import Any

# Lazy re-exports (P-25): importing this package does not pull every encoder;
# each name loads its module on first access.
_LAZY = {
    "DeepACOEncoder": "logic.src.models.subnets.encoders.deepaco:DeepACOEncoder",
    "GFACSEncoder": "logic.src.models.subnets.encoders.gfacs:GFACSEncoder",
    "GatedGraphAttConvEncoder": "logic.src.models.subnets.encoders.ggac:GatedGraphAttConvEncoder",
    "GraphAttConvEncoder": "logic.src.models.subnets.encoders.gac:GraphAttConvEncoder",
    "GraphAttentionEncoder": "logic.src.models.subnets.encoders.gat:GraphAttentionEncoder",
    "GraphConvolutionEncoder": "logic.src.models.subnets.encoders.gcn:GraphConvolutionEncoder",
    "MDAMGraphAttentionEncoder": "logic.src.models.subnets.encoders.mdam:MDAMGraphAttentionEncoder",
    "MLPEncoder": "logic.src.models.subnets.encoders.mlp:MLPEncoder",
    "MatNetEncoder": "logic.src.models.subnets.encoders.matnet:MatNetEncoder",
    "MoEGraphAttentionEncoder": "logic.src.models.subnets.encoders.moe:MoEGraphAttentionEncoder",
    "NARGNNEncoder": "logic.src.models.subnets.encoders.nargnn:NARGNNEncoder",
    "NARGNNNodeEncoder": "logic.src.models.subnets.encoders.nargnn:NARGNNNodeEncoder",
    "PointerEncoder": "logic.src.models.subnets.encoders.ptr:PointerEncoder",
    "TransGraphConvEncoder": "logic.src.models.subnets.encoders.tgc:TransGraphConvEncoder",
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
    # Base classes and common components
    "TransformerEncoderBase",
    "EncoderFeedForwardSubLayer",
    "MultiHeadAttentionLayerBase",
    # Encoder implementations
    "DeepACOEncoder",
    "GraphAttConvEncoder",
    "GraphAttentionEncoder",
    "GraphConvolutionEncoder",
    "GFACSEncoder",
    "GatedGraphAttConvEncoder",
    "MatNetEncoder",
    "MDAMGraphAttentionEncoder",
    "MLPEncoder",
    "MoEGraphAttentionEncoder",
    "NARGNNEncoder",
    "NARGNNNodeEncoder",
    "PointerEncoder",
    "TransGraphConvEncoder",
]
