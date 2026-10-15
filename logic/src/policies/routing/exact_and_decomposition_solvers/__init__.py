"""Exact and decomposition solvers. Subpackages load on first access."""

from typing import Any

from logic.src.utils.registry import lazy_getattr

_LAZY = {
    "branch_and_bound": "logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_bound",
    "branch_and_cut": "logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_cut",
    "branch_and_price": "logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_price",
    "branch_and_price_and_cut": "logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_price_and_cut",
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> Any:
    return lazy_getattr(globals(), _LAZY, name)
