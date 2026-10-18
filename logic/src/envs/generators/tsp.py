"""
Traveling Salesman Problem (TSP) instance generator.

Attributes:
    TSPGenerator: TSPGenerator class.

Example:
    >>> from logic.src.envs.generators import TSPGenerator
    >>> td = TSPGenerator(num_loc=20)(batch_size=2)
    >>> td["locs"].shape, td["depot"].shape
    (torch.Size([2, 20, 2]), torch.Size([2, 2]))
"""

from __future__ import annotations

from .base import RoutingGenerator


class TSPGenerator(RoutingGenerator):
    """
    Generator for Traveling Salesman Problem (TSP) instances: customer locations and a depot.

    The tour starts and ends at the depot, which ``TSPEnv`` prepends as node 0.
    """


__all__ = ["TSPGenerator"]
