"""Learning-heuristic routing policies. Policy modules load on first access."""

from typing import Any

from logic.src.utils.registry import lazy_getattr

_LAZY = {
    "policy_rl_alns": "logic.src.policies.routing.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.policy_rl_alns",
    "policy_rl_hvpl": "logic.src.policies.routing.learning_heuristic_algorithms.reinforcement_learning_hybrid_volleyball_premier_league.policy_rl_hvpl",
    "policy_rl_gd_hh": "logic.src.policies.routing.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.policy_rl_gd_hh",
}

__all__ = list(_LAZY)  # pyrefly: ignore[unresolvable-dunder-all]  # lazy re-exports resolved through __getattr__


def __getattr__(name: str) -> Any:
    return lazy_getattr(globals(), _LAZY, name)
