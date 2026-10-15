"""Search strategies for Branch-and-Bound tree node selection in BPC algorithms.

Provides abstraction for different tree exploration strategies:
- Best-First Search (BFS): Explores nodes with best LP bounds first
- Depth-First Search (DFS): Explores deepest nodes first

Attributes:
    NodeSelectionStrategy (class): Abstract base class for strategies.
    BestFirstSearch (class): BFS implementation.
    DepthFirstSearch (class): DFS implementation.
    HybridSearchStrategy (class): Combined dive-and-bound implementation.

Example:
    >>> strategy = create_search_strategy("depth_first")
    >>> next_node = strategy.select_node(open_nodes)
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from logic.src.policies.helpers.solvers_and_matheuristics import BranchAndBoundTree, BranchNode
else:
    # Use Any for runtime or environments where TYPE_CHECKING is False
    BranchAndBoundTree = Any
    BranchNode = Any

logger = logging.getLogger(__name__)


class NodeSelectionStrategy(ABC):
    """Abstract base class for B&B tree node selection strategies.

    Different strategies prioritize nodes differently, affecting both the
    search performance and the ability to leverage LP basis information.

    Attributes:
        None (abstract).
    """

    @abstractmethod
    def select_node(self, open_nodes: List[BranchNode]) -> Optional[BranchNode]:
        """Select and remove the next node to explore from the open node list.

        Args:
            open_nodes: List of unexplored branch nodes.

        Returns:
            Optional[BranchNode]: Selected node, or None if list is empty.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the strategy name for logging and debugging.

        Returns:
            str: Name of the strategy.
        """
        pass


class BestFirstSearch(NodeSelectionStrategy):
    """Best-First Search strategy: selects node with highest LP objective bound.

    While standard for various optimization problems, in the context of BPC
    for VRPP, it can be less efficient than Depth-First Search because it
    frequently jumps across different tree branches, preventing effective
    LP basis reuse.

    Attributes:
        None.
    """

    def select_node(self, open_nodes: List[BranchNode]) -> Optional[BranchNode]:
        """Select node with highest LP bound (maximization problem).

        Args:
            open_nodes: List of unexplored branch nodes.

        Returns:
            Optional[BranchNode]: Node with highest LP bound, or None if list is empty.
        """
        if not open_nodes:
            return None

        # Fix 8: Use O(n) scan instead of O(n log n) sort
        def get_best_first_key(i: int) -> float:
            """Key function for best-first selection based on LP bound.

            Args:
                i: Index in the open nodes list.

            Returns:
                float: The LP bound of the node, or -inf if None.
            """
            bound = open_nodes[i].lp_bound
            return bound if bound is not None else float("-inf")

        best_idx = max(range(len(open_nodes)), key=get_best_first_key)

        return open_nodes.pop(best_idx)

    def get_name(self) -> str:
        """Returns the identifier for this strategy.

        Returns:
            str: The strategy name 'best_first'.
        """
        return "best_first"


class DepthFirstSearch(NodeSelectionStrategy):
    """Depth-First Search strategy: selects deepest node in the tree.

    This is the preferred default strategy for the BPC solver as it:
    - Maximizes LP basis reuse by staying within the same branch.
    - Achieves significant speedups (often 3-5x) by resolving child LPs
      from the parent's basis using dual simplex.

    Attributes:
        None.
    """

    def select_node(self, open_nodes: List[BranchNode]) -> Optional[BranchNode]:
        """
        Select the next node to explore using DFS (LIFO).

        Args:
            open_nodes: List of unexplored branch nodes (treated as a stack).

        Returns:
            The last node appended, or None if the list is empty.
        """
        if not open_nodes:
            return None
        # open_nodes is used as a LIFO stack: the last element appended is
        # the next to explore. In run_custom_bpc, add the preferred branch
        # (shorter path / higher LP bound) LAST so it is popped first.
        return open_nodes.pop()

    def get_name(self) -> str:
        """Returns the identifier for this strategy.

        Returns:
            str: The strategy name 'depth_first'.
        """
        return "depth_first"


class HybridSearchStrategy(NodeSelectionStrategy):
    """Hybrid Dive-and-Best-Bound (D&BB) strategy.

    1. If no integer incumbent exists: Behaves like DFS.
    2. Once an incumbent exists: Behaves like BestFS.

    Attributes:
        bb_tree (BranchAndBoundTree): Reference to tree state.
    """

    def __init__(self, bb_tree: "BranchAndBoundTree"):
        """
        Store reference to tree state to monitor incumbents.

        Args:
            bb_tree: The B&B tree structure.
        """
        self.bb_tree = bb_tree

    def select_node(self, open_nodes: List[BranchNode]) -> Optional[BranchNode]:
        """Select node based on search phase.

        Args:
            open_nodes: List of unexplored branch nodes.

        Returns:
            Optional[BranchNode]: Selected node.
        """
        if not open_nodes:
            return None

        # Dive Phase: DFS (LIFO)
        if self.bb_tree.best_integer_solution is None:
            return open_nodes.pop()

        # Bound Phase: Best-First Search (O(n) scan)
        def get_best_first_key(i: int) -> float:
            """Key function for best-first selection based on LP bound.

            Args:
                i: Index in the open nodes list.

            Returns:
                float: The LP bound of the node, or -inf if None.
            """
            bound = open_nodes[i].lp_bound
            return bound if bound is not None else float("-inf")

        best_idx = max(range(len(open_nodes)), key=get_best_first_key)
        return open_nodes.pop(best_idx)

    def get_name(self) -> str:
        """Returns the identifier for this strategy.

        Returns:
            str: The strategy name 'hybrid'.
        """
        return "hybrid"


def create_search_strategy(strategy_name: str, bb_tree: Optional[Any] = None) -> NodeSelectionStrategy:
    """Factory function to create node selection strategies.

    Args:
        strategy_name: Name of the strategy ("best_first", "depth_first", "hybrid")
        bb_tree: Optional reference to the tree for hybrid strategy.

    Returns:
        NodeSelectionStrategy: Instance of the requested strategy.

    Raises:
        ValueError: If strategy_name is not recognized.
    """
    if strategy_name == "best_first":
        return BestFirstSearch()
    elif strategy_name == "depth_first":
        return DepthFirstSearch()
    elif strategy_name == "hybrid":
        if bb_tree is None:
            raise ValueError("Hybrid search strategy requires a BranchAndBoundTree reference.")
        return HybridSearchStrategy(bb_tree)
    else:
        raise ValueError(
            f"Unknown search strategy '{strategy_name}'. Valid options are: ['best_first', 'depth_first', 'hybrid']"
        )
