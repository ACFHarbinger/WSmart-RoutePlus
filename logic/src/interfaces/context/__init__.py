"""
Context interfaces for WSmart-Route logic components.

Attributes:
    JointSelectionConstructionContext: Joint packing and routing context
    MultiDayContext: Multi-day routing context
    ProblemContext: Problem definition context
    SolutionContext: Solution context
    SearchContext: Search context.
    AcceptanceMetrics: Acceptance metrics.
    ConstructionMetrics: Construction metrics.
    ImprovementMetrics: Improvement metrics.
    SelectionMetrics: Selection metrics.
    SearchPhase: Search phase.
    PackingContext: Data container for selection context and state.
    merge_context: Merge contexts.

Example:
    >>> from logic.src.interfaces.context import (
        AcceptanceMetrics,
        ConstructionMetrics,
        ImprovementMetrics,
        SearchContext,
        SearchPhase,
        SelectionMetrics,
        PackingContext,
        merge_context,
        MultiDayContext,
        ProblemContext,
        SolutionContext,
        JointSelectionConstructionContext,
    )
"""

from .joint_context import JointSelectionConstructionContext
from .multi_day_context import MultiDayContext
from .packing_context import PackingContext
from .problem_context import ProblemContext
from .search_context import (
    AcceptanceMetrics,
    ConstructionMetrics,
    ImprovementMetrics,
    SearchContext,
    SearchPhase,
    SelectionMetrics,
    merge_context,
)
from .solution_context import SolutionContext

__all__ = [
    "SearchContext",
    "SearchPhase",
    "SelectionMetrics",
    "ConstructionMetrics",
    "AcceptanceMetrics",
    "ImprovementMetrics",
    "merge_context",
    "MultiDayContext",
    "ProblemContext",
    "SolutionContext",
    "JointSelectionConstructionContext",
    "PackingContext",
]
