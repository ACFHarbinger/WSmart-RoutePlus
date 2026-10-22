"""Vectorized packing adapters. Public names load on first access."""

from typing import Any

from logic.src.utils.registry import lazy_getattr

_LAZY = {
    "VectorizedSelector": "logic.src.policies.vectorized.packing.base:VectorizedSelector",
    "LastMinuteSelector": "logic.src.policies.vectorized.packing.last_minute:LastMinuteSelector",
    "RegularSelector": "logic.src.policies.vectorized.packing.regular:RegularSelector",
    "LookaheadSelector": "logic.src.policies.vectorized.packing.lookahead:LookaheadSelector",
    "RevenueSelector": "logic.src.policies.vectorized.packing.revenue:RevenueSelector",
    "ServiceLevelSelector": "logic.src.policies.vectorized.packing.service_level:ServiceLevelSelector",
    "CombinedSelector": "logic.src.policies.vectorized.packing.combined:CombinedSelector",
    "get_vectorized_selector": "logic.src.policies.vectorized.packing.factory:get_vectorized_selector",
    "create_selector_from_config": "logic.src.policies.vectorized.packing.factory:create_selector_from_config",
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> Any:
    return lazy_getattr(globals(), _LAZY, name)
