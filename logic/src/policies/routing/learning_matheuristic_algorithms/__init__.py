"""Learning-matheuristic routing policies. Policy modules load on first access."""

from typing import Any

from logic.src.utils.registry import lazy_getattr

_LAZY = {
    "policy_abpc_hg": "logic.src.policies.routing.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg",
    "policy_calm": "logic.src.policies.routing.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm",
}

__all__ = list(_LAZY)  # pyrefly: ignore[unresolvable-dunder-all]  # lazy re-exports resolved through __getattr__


def __getattr__(name: str) -> Any:
    return lazy_getattr(globals(), _LAZY, name)
