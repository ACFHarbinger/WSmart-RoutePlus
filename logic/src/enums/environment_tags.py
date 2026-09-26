"""
Environment enum for WSmart-Route.

Attributes:
    EnvironmentTag: Enum for environment tags

Example:
    >>> from logic.src.enums import EnvironmentTag
    >>> EnvironmentTag.EUCLIDEAN
    <EnvironmentTag.EUCLIDEAN: 1>
"""

from enum import Enum, auto


class EnvironmentTag(Enum):
    """
    Environment tags for WSmart-Route.

    Attributes:
        EUCLIDEAN: Nodes have (x,y) coordinates
        GRAPH: Fully defined by a distance matrix (non-Euclidean possible)
        CAPACITATED: Vehicle load limits
        TIME_WINDOWS: Strict [early, late] arrival constraints
        MULTI_DEPOT: Multiple start/end locations
        MULTI_PERIOD: Decisions span across a time horizon (days/weeks)
        TIME_BUDGET: Total tour/trip time budget (orienteering-style resource
            constraint), distinct from TIME_WINDOWS' per-node arrival window
        DETERMINISTIC: All info known at t=0
        STOCHASTIC: Node demands/presence are random variables
        DYNAMIC: Nodes appear over time during execution
        MIN_DISTANCE: Canonical VRP
        MAX_PROFIT: Orienteering / Prize-collecting
    """

    # Spatial Topology
    EUCLIDEAN = auto()  # Nodes have (x,y) coordinates
    GRAPH = auto()  # Fully defined by a distance matrix (non-Euclidean possible)

    # Constraints
    CAPACITATED = auto()  # Vehicle load limits
    TIME_WINDOWS = auto()  # Strict [early, late] arrival constraints
    TIME_BUDGET = auto()  # Total tour/trip time budget (orienteering-style)
    MULTI_DEPOT = auto()  # Multiple start/end locations
    MULTI_PERIOD = auto()  # Decisions span across a time horizon (days/weeks)

    # Dynamics & Uncertainty
    DETERMINISTIC = auto()  # All info known at t=0
    STOCHASTIC = auto()  # Node demands/presence are random variables
    DYNAMIC = auto()  # Nodes appear over time during execution

    # Objective
    MIN_DISTANCE = auto()  # Canonical VRP
    MAX_PROFIT = auto()  # Orienteering / Prize-collecting
