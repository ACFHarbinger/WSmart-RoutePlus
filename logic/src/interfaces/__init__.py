"""Interfaces for WSmart-Route logic components. Names load on first access."""

from typing import Any

from logic.src.utils.registry import lazy_getattr

_LAZY = {
    "IAcceptanceCriterion": "logic.src.interfaces.acceptance_criterion:IAcceptanceCriterion",
    "IBinContainer": "logic.src.interfaces.bin_container:IBinContainer",
    "IEnv": "logic.src.interfaces.env:IEnv",
    "IPackingAlgorithm": "logic.src.interfaces.packing:IPackingAlgorithm",
    "IModel": "logic.src.interfaces.model:IModel",
    "IPolicy": "logic.src.interfaces.policy:IPolicy",
    "IRoutingAlgorithm": "logic.src.interfaces.routing_algorithm:IRoutingAlgorithm",
    "IRouteImprovement": "logic.src.interfaces.route_improvement:IRouteImprovement",
    "ITensorDictLike": "logic.src.interfaces.tensor_dict_like:ITensorDictLike",
    "ITraversable": "logic.src.interfaces.traversable:ITraversable",
    "SearchContext": "logic.src.interfaces.context.search_context:SearchContext",
    "SearchPhase": "logic.src.interfaces.context.search_context:SearchPhase",
    "SelectionMetrics": "logic.src.interfaces.context.search_context:SelectionMetrics",
    "ConstructionMetrics": "logic.src.interfaces.context.search_context:ConstructionMetrics",
    "AcceptanceMetrics": "logic.src.interfaces.context.search_context:AcceptanceMetrics",
    "ImprovementMetrics": "logic.src.interfaces.context.search_context:ImprovementMetrics",
    "merge_context": "logic.src.interfaces.context.search_context:merge_context",
    "SelectionContext": "logic.src.interfaces.context.selection_context:SelectionContext",
    "JointSelectionConstructionContext": "logic.src.interfaces.context.joint_context:JointSelectionConstructionContext",
    "MultiDayContext": "logic.src.interfaces.context.multi_day_context:MultiDayContext",
    "ProblemContext": "logic.src.interfaces.context.problem_context:ProblemContext",
    "SolutionContext": "logic.src.interfaces.context.solution_context:SolutionContext",
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> Any:
    return lazy_getattr(globals(), _LAZY, name)
