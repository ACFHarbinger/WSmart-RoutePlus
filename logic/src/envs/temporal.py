"""Temporal-resource defaults shared by time-constrained routing environments."""

from __future__ import annotations

from typing import Tuple


def get_default_temporal_params() -> Tuple[float, float, float]:
    """Return the default per-trip shift, travel-speed, and service-time values.

    These values mirror ``SimulationRepository.get_temporal_params()`` but
    live outside the simulation package so routing environments remain
    importable without initializing simulation-data repositories.
    """
    return (7.0, 35.0, 1.5 / 60.0)
