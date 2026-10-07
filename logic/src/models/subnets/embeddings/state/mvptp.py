"""MVPTP specific context embedding module.

This module provides the MVPTPState component, which builds on PTPState
to provide context for Capacitated VRPs with Profits.

Attributes:
    MVPTPState: State encoder for MVPTPs.

Example:
    >>> from logic.src.models.subnets.embeddings.state.mvptp import MVPTPState
    >>> state_embedder = MVPTPState(embed_dim=128)
    >>> context = state_embedder(embeddings, td)
"""

from __future__ import annotations

from .ptp import PTPState


class MVPTPState(PTPState):
    """Context embedding for MVPTP.

    Inherits from PTPState as the fundamental state features (capacity/remaining
    length) are shared between these problem types.

    Attributes:
        embed_dim (int): Dimensionality of the projected state context.
    """

    pass
