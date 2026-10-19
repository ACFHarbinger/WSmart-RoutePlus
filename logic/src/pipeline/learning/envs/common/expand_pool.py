"""Problem-derived optional-visit flag for routing solvers.

The day loop sets the flag from ``ProblemSpec.expand_pool`` before a solve.
Solvers read ``current_expand_pool()`` instead of a per-policy ``vrpp`` field.
"""

from contextvars import ContextVar
from typing import Optional

_EXPAND_POOL: ContextVar[Optional[bool]] = ContextVar("expand_pool", default=None)


def set_expand_pool(value: Optional[bool]):
    """Bind the flag for the current solve. Returns the contextvar token."""
    return _EXPAND_POOL.set(value)


def reset_expand_pool(token) -> None:
    """Restore the previous flag."""
    _EXPAND_POOL.reset(token)


def current_expand_pool(default: bool = True) -> bool:
    """Return the active flag, or ``default`` when the day loop has not set one."""
    value = _EXPAND_POOL.get()
    return default if value is None else bool(value)
