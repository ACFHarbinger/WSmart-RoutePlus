"""
CPTP problem generator.

Attributes:
    CPTPGenerator: Capacitated Profitable Tour Problem instances.

Example:
    >>> from logic.src.envs.generators import CPTPGenerator
    >>> td = CPTPGenerator(num_loc=20, capacity=2.0)(batch_size=4)
    >>> td["capacity"]
    tensor([2., 2., 2., 2.])
"""

from __future__ import annotations

from .ptp import PTPGenerator


class CPTPGenerator(PTPGenerator):
    """Generator for Capacitated Profitable Tour Problem (CPTP) instances.

    The instance layout is PTP's; the capacity it carries (``capacity``) binds per trip
    in ``CPTPEnv``.
    """


__all__ = ["CPTPGenerator"]
