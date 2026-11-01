"""CMPTP specific context embedding module.

This module provides the CMPTPState component, which builds on PTPState
to provide context for Capacitated VRPs with Profits.

Attributes:
    CMPTPState: State encoder for CMPTPs.

Example:
    >>> from logic.src.models.subnets.embeddings.state.cmptp import CMPTPState
    >>> state_embedder = CMPTPState(embed_dim=128)
    >>> context = state_embedder(embeddings, td)
"""

from __future__ import annotations

from .ptp import PTPState


class CMPTPState(PTPState):
    """Context embedding for CMPTP.

    Inherits from PTPState as the fundamental state features (capacity/remaining
    length) are shared between these problem types.

    Attributes:
        embed_dim (int): Dimensionality of the projected state context.
    """

    pass
