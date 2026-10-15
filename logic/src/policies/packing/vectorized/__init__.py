"""Vectorized packing adapters. Public names load on first access."""

from typing import Any

from logic.src.utils.registry import lazy_getattr

_LAZY = {
    "VectorizedSelector": "logic.src.policies.packing.vectorized.base:VectorizedSelector",
    "LastMinuteSelector": "logic.src.policies.packing.vectorized.last_minute:LastMinuteSelector",
    "RegularSelector": "logic.src.policies.packing.vectorized.regular:RegularSelector",
    "LookaheadSelector": "logic.src.policies.packing.vectorized.lookahead:LookaheadSelector",
    "RevenueSelector": "logic.src.policies.packing.vectorized.revenue:RevenueSelector",
    "ServiceLevelSelector": "logic.src.policies.packing.vectorized.service_level:ServiceLevelSelector",
    "CombinedSelector": "logic.src.policies.packing.vectorized.combined:CombinedSelector",
    "get_vectorized_selector": "logic.src.policies.packing.vectorized.factory:get_vectorized_selector",
    "create_selector_from_config": "logic.src.policies.packing.vectorized.factory:create_selector_from_config",
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> Any:
    return lazy_getattr(globals(), _LAZY, name)
