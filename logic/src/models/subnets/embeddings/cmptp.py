"""CMPTP Embedding module.

This module provides the CMPTPInitEmbedding layer, which encodes locations
and waste levels for Capacitated Vehicle Routing Problems with Profits.

Attributes:
    CMPTPInitEmbedding: Initial feature encoder for CMPTP instances.

Example:
    >>> from logic.src.models.subnets.embeddings.cmptp import CMPTPInitEmbedding
    >>> embed = CMPTPInitEmbedding(embed_dim=128)
    >>> h = embed(td)
"""

from __future__ import annotations

from logic.src.models.subnets.embeddings.ptp import PTPInitEmbedding


class CMPTPInitEmbedding(PTPInitEmbedding):
    """Initial embedding for CMPTP (Capacitated Multi-vehicle Profitable Tour Problem).

    Inherits customer and depot feature projections from PTPInitEmbedding
    while preserving bitwise parameter parity and state_dict key structure.
    """


__all__ = ["CMPTPInitEmbedding"]
