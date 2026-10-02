"""Consolidated Exact Solvers for VRPP.

Provides high-performance, mathematically rigorous implementations of
Branch-and-Price-and-Cut components.

Attributes:
    VRPPMasterProblem: Column generation master problem implementation.
    RCSPPSolver: Resource Constrained Shortest Path Problem solver.
    SeparationEngine: Automated cut generation and separation.
    BranchAndBoundTree: Orchestrator for the B&B search.
    run_subgradient: Lagrangian subgradient optimization entry point.

Example:
    >>> master = VRPPMasterProblem(nodes, edges, capacity)
    >>> pricing = RCSPPSolver(dist_matrix, duals)
    >>> routes = pricing.solve()
"""

from logic.src.policies.helpers.solvers_and_matheuristics.branching import (
    AnyBranchingConstraint,
    BranchAndBoundTree,
    BranchingConstraint,
    EdgeBranching,
    EdgeBranchingConstraint,
    FleetSizeBranchingConstraint,
    MultiEdgePartitionBranching,
    NodeVisitationBranchingConstraint,
    RyanFosterBranching,
    RyanFosterBranchingConstraint,
)
from logic.src.policies.helpers.solvers_and_matheuristics.common import BranchNode, Route
from logic.src.policies.helpers.solvers_and_matheuristics.master_problem import GlobalCutPool, VRPPMasterProblem
from logic.src.policies.helpers.solvers_and_matheuristics.pricing import Label, PricingStatus, RCSPPSolver
from logic.src.policies.helpers.solvers_and_matheuristics.search import (
    BasicFleetCoverEngine,
    BestFirstSearch,
    CompositeCuttingPlaneEngine,
    CuttingPlaneEngine,
    DepthFirstSearch,
    EdgeCliqueCutEngine,
    HybridSearchStrategy,
    KnapsackCoverEngine,
    NodeSelectionStrategy,
    PhysicalCapacityLCIEngine,
    RoundedCapacityCutEngine,
    RoundedMultistarCutEngine,
    SaturatedArcLCIEngine,
    SubsetRowCutEngine,
    create_cutting_plane_engine,
    create_search_strategy,
)
from logic.src.policies.helpers.solvers_and_matheuristics.separation import (
    CapacityCut,
    CombInequality,
    PCSubtourEliminationCut,
    SeparationEngine,
)

from .lagrangian_relaxation.subgradient_optimization import _nearest_neighbour_tour_cost, run_subgradient
from .lagrangian_relaxation.uncapacitated_orienteering_problem import solve_uncapacitated_op

__all__ = [
    "AnyBranchingConstraint",
    "BranchingConstraint",
    "EdgeBranchingConstraint",
    "FleetSizeBranchingConstraint",
    "NodeVisitationBranchingConstraint",
    "RyanFosterBranchingConstraint",
    "BranchAndBoundTree",
    "BranchNode",
    "EdgeBranching",
    "RyanFosterBranching",
    "MultiEdgePartitionBranching",
    "GlobalCutPool",
    "Route",
    "VRPPMasterProblem",
    "PricingStatus",
    "RCSPPSolver",
    "Label",
    "CapacityCut",
    "CombInequality",
    "PCSubtourEliminationCut",
    "SeparationEngine",
    "run_subgradient",
    "_nearest_neighbour_tour_cost",
    "solve_uncapacitated_op",
    "BasicFleetCoverEngine",
    "CompositeCuttingPlaneEngine",
    "CuttingPlaneEngine",
    "EdgeCliqueCutEngine",
    "KnapsackCoverEngine",
    "PhysicalCapacityLCIEngine",
    "RoundedCapacityCutEngine",
    "RoundedMultistarCutEngine",
    "SaturatedArcLCIEngine",
    "SubsetRowCutEngine",
    "create_cutting_plane_engine",
    "BestFirstSearch",
    "DepthFirstSearch",
    "HybridSearchStrategy",
    "NodeSelectionStrategy",
    "create_search_strategy",
]
