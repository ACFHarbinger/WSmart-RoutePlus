"""Sequential Routing Algorithm (SRC) package.

This package implements a meta-constructor that executes a fixed chain
of routing heuristics in sequence, threading the state between them.

Attributes:
    SRCConfig: Parameters for the Sequential Routing Algorithm.
    SequentialRoutingAlgorithm: Sequential Routing Algorithm (SRC).

Example:
    >>> from logic.src.policies.routing.other_algorithms.sequential_routing_algorithm import SequentialRoutingAlgorithm, SRCConfig
    >>> params = SRCConfig()
    >>> src = SequentialRoutingAlgorithm(config=params)
    >>> src.execute(mandatory=[1, 2, 3], bins=bins_object, distance_matrix=distance_matrix)
"""

from logic.src.configs.policies.src import SRCConfig as SRCConfig
from .policy_src import SequentialRoutingAlgorithm as SequentialRoutingAlgorithm

__all__ = ["SequentialRoutingAlgorithm", "SRCConfig"]
