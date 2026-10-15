"""Decoders package.

This package provides various constructive and non-autoregressive decoder
architectures for combinatorial optimization problems.

Attributes:
    GraphAttentionDecoder: GAT-based constructive decoder.
    GlimpseDecoder: Decoder using multiple glimpse steps.
    MatNetDecoder: Matrix-based attention decoder.
    MDAMDecoder: Parallel multi-path decoder.
    SimpleNARDecoder: Heatmap-based non-autoregressive decoder.
    PolyNetDecoder: Polynomial attention decoder.
    PointerDecoder: Classical recurrent pointer network decoder.
    ACODecoder: Ant Colony Optimization based decoder.

Example:
    >>> from logic.src.models.subnets.decoders import MDAMDecoder
    >>> decoder = MDAMDecoder(...)
"""

from typing import Any

# Lazy re-exports (P-25): importing this package does not pull every encoder;
# each name loads its module on first access.
_LAZY = {
    "ACODecoder": "logic.src.models.subnets.decoders.deepaco:ACODecoder",
    "DeepGATDecoder": "logic.src.models.subnets.decoders.gat:DeepGATDecoder",
    "FeedForwardSubLayer": "logic.src.models.subnets.decoders.common:FeedForwardSubLayer",
    "GlimpseDecoder": "logic.src.models.subnets.decoders.glimpse.decoder:GlimpseDecoder",
    "GraphAttentionDecoder": "logic.src.models.subnets.decoders.gat:GraphAttentionDecoder",
    "MDAMDecoder": "logic.src.models.subnets.decoders.mdam:MDAMDecoder",
    "MatNetDecoder": "logic.src.models.subnets.decoders.matnet:MatNetDecoder",
    "PointerDecoder": "logic.src.models.subnets.decoders.ptr:PointerDecoder",
    "PolyNetDecoder": "logic.src.models.subnets.decoders.polynet:PolyNetDecoder",
    "SimpleNARDecoder": "logic.src.models.subnets.decoders.nar:SimpleNARDecoder",
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


__all__: list[str] = [
    "FeedForwardSubLayer",
    "ACODecoder",
    "DeepGATDecoder",
    "GraphAttentionDecoder",
    "GlimpseDecoder",
    "MatNetDecoder",
    "MDAMDecoder",
    "SimpleNARDecoder",
    "PolyNetDecoder",
    "PointerDecoder",
]
