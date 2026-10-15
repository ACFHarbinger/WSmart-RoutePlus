"""Search and cutting plane package for Branch-and-Price-and-Cut solvers.

This package provides components for tree exploration strategies and
valid inequality separation (cutting planes) used in exact and
decomposition-based solvers.

Attributes:
    CuttingPlaneEngine (class): Abstract base class for cut separation.
    RoundedCapacityCutEngine (class): Standard VRP capacity cuts.
    SubsetRowCutEngine (class): Three-row subset row cuts for Set Partitioning.
    EdgeCliqueCutEngine (class): Clique-based edge cuts.
    KnapsackCoverEngine (class): Knapsack-derived cover inequalities.
    BasicFleetCoverEngine (class): Fleet size bounding cuts.
    PhysicalCapacityLCIEngine (class): Lifted Cover Inequalities for capacity.
    SaturatedArcLCIEngine (class): Lifted Cover Inequalities for arcs.
    RoundedMultistarCutEngine (class): Multi-star inequalities for VRP.
    CompositeCuttingPlaneEngine (class): Orchestrator for multiple engines.
    create_cutting_plane_engine (function): Factory for cutting plane engines.

    NodeSelectionStrategy (class): Abstract base class for B&B tree search.
    BestFirstSearch (class): BFS strategy.
    DepthFirstSearch (class): DFS strategy.
    HybridSearchStrategy (class): Combined DFS/BFS strategy.
    create_search_strategy (function): Factory for search strategies.

Example:
    >>> from logic.src.policies.helpers.solvers_and_matheuristics.search import (
    >>>     BestFirstSearch,
    >>>     create_search_strategy,
    >>> )
    >>> search = create_search_strategy("best_first")
    >>> search = BestFirstSearch()
"""

from .bpc_tree import BPCSearchOptions, search_bpc_tree, total_travel_cost
from .cutting_planes import (
    BasicFleetCoverEngine,
    CompositeCuttingPlaneEngine,
    CuttingPlaneEngine,
    EdgeCliqueCutEngine,
    KnapsackCoverEngine,
    PhysicalCapacityLCIEngine,
    RoundedCapacityCutEngine,
    RoundedMultistarCutEngine,
    SaturatedArcLCIEngine,
    SubsetRowCutEngine,
    create_cutting_plane_engine,
)
from .search_strategy import (
    BestFirstSearch,
    DepthFirstSearch,
    HybridSearchStrategy,
    NodeSelectionStrategy,
    create_search_strategy,
)

__all__ = [
    "CuttingPlaneEngine",
    "RoundedCapacityCutEngine",
    "SubsetRowCutEngine",
    "EdgeCliqueCutEngine",
    "KnapsackCoverEngine",
    "BasicFleetCoverEngine",
    "PhysicalCapacityLCIEngine",
    "SaturatedArcLCIEngine",
    "RoundedMultistarCutEngine",
    "CompositeCuttingPlaneEngine",
    "create_cutting_plane_engine",
    "NodeSelectionStrategy",
    "BestFirstSearch",
    "DepthFirstSearch",
    "HybridSearchStrategy",
    "create_search_strategy",
    "BPCSearchOptions",
    "search_bpc_tree",
    "total_travel_cost",
]
