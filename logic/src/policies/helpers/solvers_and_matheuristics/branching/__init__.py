"""Branching package for VRPP Branch-and-Price-and-Cut.

Provides several constraint types and the supporting B&B tree infrastructure.

Attributes:
    BranchAndBoundTree: Orchestrator for the B&B search process.
    EdgeBranchingConstraint: Constraint forcing/forbidding specific arcs.
    RyanFosterBranchingConstraint: Constraint forcing/forbidding node pairs together.
    FleetSizeBranchingConstraint: Constraint on total number of vehicles.
    MultiEdgePartitionBranching: Spatial fleet-partitioning heuristic.

Example:
    >>> tree = BranchAndBoundTree(root_node, strategy="ryan_foster")
    >>> tree.solve()

Theoretical Framework:
    Ryan-Foster branching is utilized for VRPP because it appropriately modifies
    the Resource-Constrained Shortest Path Problem (RCSPP) used for pricing by
    enforcing or forbidding node pairs. In a Set Partitioning Problem (SPP),
    node-pair branching is mathematically exact and avoids the "symmetry" issues
    inherent in arc-based branching for problems with multiple identical vehicles.
"""

from __future__ import annotations

from .constraints import (
    AnyBranchingConstraint,
    EdgeBranchingConstraint,
    FleetSizeBranchingConstraint,
    NodeVisitationBranchingConstraint,
    RyanFosterBranchingConstraint,
)
from .strategies import (
    EdgeBranching,
    FleetSizeBranching,
    MultiEdgePartitionBranching,
    NodeVisitationBranching,
    RyanFosterBranching,
)
from .tree import BranchAndBoundTree

__all__ = [
    "EdgeBranchingConstraint",
    "RyanFosterBranchingConstraint",
    "FleetSizeBranchingConstraint",
    "NodeVisitationBranchingConstraint",
    "AnyBranchingConstraint",
    "EdgeBranching",
    "RyanFosterBranching",
    "FleetSizeBranching",
    "NodeVisitationBranching",
    "MultiEdgePartitionBranching",
    "BranchAndBoundTree",
]
