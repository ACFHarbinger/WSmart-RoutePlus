"""CPTP Embedding module.

This module provides the CPTPInitEmbedding layer, which encodes locations
and waste levels for Capacitated Vehicle Routing Problems with Profits.

Attributes:
    CPTPInitEmbedding: Initial feature encoder for CPTP instances.

Example:
    >>> from logic.src.models.subnets.embeddings.cptp import CPTPInitEmbedding
    >>> embed = CPTPInitEmbedding(embed_dim=128)
    >>> h = embed(td)
"""

from __future__ import annotations

from logic.src.models.subnets.embeddings.ptp import PTPInitEmbedding


class CPTPInitEmbedding(PTPInitEmbedding):
    """Initial embedding for CPTP (Capacitated Profitable Tour Problem).

    Inherits customer and depot feature projections from PTPInitEmbedding
    while preserving bitwise parameter parity and state_dict key structure.
    """


__all__ = ["CPTPInitEmbedding"]
