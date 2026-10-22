"""Make the final route vehicle-feasible after construction and improvement (CPTPTC: also shift-feasible)."""

import logging
from typing import Any, Dict

import numpy as np

from logic.src.utils.routing.tours import get_multi_tour

from .base import SimulationAction, bind_context

logger = logging.getLogger(__name__)


class TimeConstraintAction(SimulationAction):
    """Insert depot returns so every trip fits the vehicle (and, for CPTPTC, the shift).

    CPTPTC: capacity and per-trip time, on the observed fill, as before.
    Other problems (owner decision 2026-09-27): capacity only, on the true fill that the
    vehicle actually loads. A trip that would overflow gets a depot return at the point
    where the next bin no longer fits; the order of visits is kept, and the extra
    kilometres are counted because collection recomputes km from the final tour.
    """

    timed = False
    reads = ("tour", "bins", "problem_spec", "vehicle_capacity", "distance_matrix")
    writes = ("tour", "capacity_splits")
    finalizes = ()
    requires_final = ()

    def execute(self, context: Dict[str, Any]) -> None:
        """Repair the final route before collection mutates any bin contents."""

        context = bind_context(context)
        from logic.src.pipeline.simulations.day_setup import per_trip_shift

        if not per_trip_shift(context):
            self._split_over_capacity_trips(context)
            return
        tour = getattr(context, "tour", None)
        context.tour = get_multi_tour(
            list(tour) if tour is not None else [],
            context.bins.c,
            context.vehicle_capacity,
            context.distance_matrix,
            shift_hours=getattr(context, "shift_hours", 7.0),
            avg_speed_kmh=getattr(context, "avg_speed_kmh", 35.0),
            service_time_h=getattr(context, "service_time_h", 0.025),
            time_matrix=getattr(context, "time_matrix", None),
        )

    @staticmethod
    def _split_over_capacity_trips(context: Dict[str, Any]) -> None:
        """Split every trip whose true load exceeds the vehicle capacity (non-CPTPTC problems)."""
        tour = getattr(context, "tour", None)
        capacity = getattr(context, "vehicle_capacity", None)
        bins = getattr(context, "bins", None)
        context.capacity_splits = 0
        if not tour or capacity is None or bins is None:
            return
        try:
            capacity_ok = np.isfinite(float(capacity))
        except (TypeError, ValueError):
            return
        if not capacity_ok:
            return
        tour = [int(n) for n in tour]
        if not any(n != 0 for n in tour):
            return
        framed = tour if tour[0] == 0 else [0] + tour
        framed = framed if framed[-1] == 0 else framed + [0]
        try:
            load = np.asarray(bins.real_c, dtype=float)
            split = get_multi_tour(framed, load, float(capacity), context.distance_matrix)
        except (AttributeError, IndexError, TypeError, ValueError) as exc:  # never lose the day's route
            logger.warning("capacity split skipped: %s", exc)
            return
        added = split.count(0) - framed.count(0)
        if added > 0:
            context.tour = split
            context.capacity_splits = added

