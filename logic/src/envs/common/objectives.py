"""Pure objective arithmetic shared by scalar, NumPy and Torch callers.

Inputs must already use compatible physical units. These functions neither replay
routes nor normalize fills, close depot legs, apply penalties, or convert arrays
to Python floats. Consequently tensor shapes, devices and gradients are retained.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Economics:
    """Injected economic rates in EUR/kg and EUR/km, respectively."""

    revenue_kg: float
    cost_km: float


def profit_objective(
    collected_kg: Any,
    distance_km: Any,
    overflows: Any = 0.0,
    revenue_kg: Any = 1.0,
    cost_km: Any = 1.0,
) -> Any:
    """Return net profit; overflow counts are reported separately, without a penalty.

    The unit-rate defaults preserve the existing task API. Physical callers must
    inject their area's rates. Normalized RL inputs must supply correspondingly
    scaled rates; conversion belongs at the input boundary, never inside a solver.
    """
    return collected_kg * revenue_kg - distance_km * cost_km


def distance_objective(distance_km: Any, **kwargs: Any) -> Any:
    """Return negative distance for maximization; CVRP cost is its negation."""
    return -distance_km
