"""CPTP specific context embedding module.

This module provides the CPTPState component, which builds on PTPState
to provide context for Capacitated VRPs with Profits.

Attributes:
    CPTPState: State encoder for CPTPs.

Example:
    >>> from logic.src.models.subnets.embeddings.state.cptp import CPTPState
    >>> state_embedder = CPTPState(embed_dim=128)
    >>> context = state_embedder(embeddings, td)
"""

from __future__ import annotations

from .ptp import PTPState


class CPTPState(PTPState):
    """Context embedding for CPTP.

    Inherits from PTPState as the fundamental state features (capacity/remaining
    length) are shared between these problem types.

    Attributes:
        embed_dim (int): Dimensionality of the projected state context.
    """

    pass
