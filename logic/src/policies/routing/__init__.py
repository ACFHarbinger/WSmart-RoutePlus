"""Routing algorithms. Family packages load on first access."""

from typing import Any

from logic.src.utils.registry import lazy_getattr

_LAZY = {
    "exact_and_decomposition_solvers": "logic.src.policies.routing.exact_and_decomposition_solvers",
    "hyper_heuristics": "logic.src.policies.routing.hyper_heuristics",
    "learning_algorithms": "logic.src.policies.routing.learning_algorithms",
    "learning_heuristic_algorithms": "logic.src.policies.routing.learning_heuristic_algorithms",
    "learning_matheuristic_algorithms": "logic.src.policies.routing.learning_matheuristic_algorithms",
    "matheuristics": "logic.src.policies.routing.matheuristics",
    "meta_heuristics": "logic.src.policies.routing.meta_heuristics",
    "other_algorithms": "logic.src.policies.routing.other_algorithms",
}

__all__ = [
    "exact_and_decomposition_solvers",
    "hyper_heuristics",
    "learning_algorithms",
    "learning_heuristic_algorithms",
    "learning_matheuristic_algorithms",
    "matheuristics",
    "meta_heuristics",
    "other_algorithms",
]


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target)
    globals()[name] = module
    return module
