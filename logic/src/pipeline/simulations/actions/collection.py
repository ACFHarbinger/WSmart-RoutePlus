"""
Action for waste collection execution.

Attributes:
    CollectAction: Command to execute waste collection.

Example:
    >>> # action = CollectAction()
    >>> # action.execute(context)
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from .base import SimulationAction, bind_context

logger = logging.getLogger(__name__)


def trip_loads(tour: List[int], fill_pct: np.ndarray) -> List[float]:
    """Load of every depot-to-depot trip in ``tour``, in percent-of-one-bin units.

    This is the unit the routing algorithms plan in: ``load_area_and_waste_type_params``
    returns the vehicle capacity in percent of one bin too (3500 kg at Rio Maior plastic
    is 7368.4 %), so each value compares directly with it.

    Args:
        tour: Depot-framed tour ``[0, ..., 0, ..., 0]``; internal zeros separate trips.
        fill_pct: Fill level of every bin in percent, 0-indexed by bin (node - 1).

    Returns:
        List[float]: One load per non-empty trip, in order.
    """
    loads: List[float] = []
    current, visited = 0.0, False
    for node in tour[1:]:
        if node == 0:
            if visited:
                loads.append(current)
            current, visited = 0.0, False
        else:
            current += float(fill_pct[node - 1])
            visited = True
    if visited:
        loads.append(current)
    return loads


def capacity_report(tour: List[int], bins: Any, capacity_pct: float) -> Tuple[List[float], List[float], int]:
    """Per-trip loads (percent and kg) of the true fill, and how many trips exceed the vehicle capacity.

    Args:
        tour: Depot-framed tour.
        bins: Bins object (``real_c`` in percent, ``volume``, ``density``).
        capacity_pct: Vehicle capacity in percent-of-one-bin units.

    Returns:
        Tuple of (loads in percent, loads in kg, number of over-capacity trips).
    """
    loads_pct = trip_loads(tour, np.asarray(bins.real_c, dtype=float))
    kg_per_pct = float(bins.volume) * float(bins.density) / 100.0
    loads_kg = [load * kg_per_pct for load in loads_pct]
    violations = sum(1 for load in loads_pct if load > capacity_pct + 1e-6)
    return loads_pct, loads_kg, violations


def _record_thompson_reward(context: Any, profit: float) -> None:
    """Update the Thompson posterior with this day's profit.

    ``pack`` only samples. Nothing else called ``record_reward``, so every
    later day redrew the prior. The winner name is stored on the packing
    search context.
    """
    search = getattr(context, "search_context", None)
    metrics = getattr(search, "selection_metrics", None) or {}
    winner = metrics.get("winner")
    if not isinstance(winner, str) or not winner:
        return
    from logic.src.policies.packing.dispatcher_thompson import ThompsonDispatcher

    ThompsonDispatcher.record_reward(
        winner,
        reward=float(profit),
        success_prob=1.0 if profit > 0 else 0.0,
        state_path=metrics.get("dispatcher_state_path"),
    )


class CollectAction(SimulationAction):
    """
    Processes waste collection from bins visited in the tour.

    Attributes:
        None
    """

    timed = False
    reads = ("tour", "bins", "distance_matrix")
    writes = (
        "tour",
        "cost",
        "collected",
        "total_collected",
        "ncol",
        "profit",
        "time_spent",
        "metrics_phase",
    )
    finalizes = ("profit", "cost")
    requires_final = ()

    def execute(self, context: Dict[str, Any]) -> None:
        """
        Execute waste collection based on the generated tour.

        Args:
            context: Shared dictionary containing simulation state.
        """

        context = bind_context(context)
        from logic.src.utils.routing.tours import (
            get_route_cost,
        )

        bins = context.bins
        tour = list(context.tour)
        if not tour or tour[0] != 0:
            tour.insert(0, 0)
        if tour[-1] != 0 or len(tour) == 1:
            tour.append(0)
        context.tour = tour

        # 1. METRIC CONSISTENCY: Always re-calculate KM from the final tour
        # This combines packing, construction, and any route improvements.
        # We use the raw distance matrix from the context (guaranteed to be KM).
        dist_matrix = context.distance_matrix
        raw_km = get_route_cost(dist_matrix, tour)

        # 3. Calculate operational time spent (travel time + per-bin service time)
        avg_speed_kmh = float(getattr(context, "avg_speed_kmh", 35.0))
        service_time_h = float(getattr(context, "service_time_h", 1.5 / 60.0))
        shift_hours = float(getattr(context, "shift_hours", 7.0))
        time_matrix = getattr(context, "time_matrix", None)
        if time_matrix is None:
            if not np.isfinite(avg_speed_kmh) or avg_speed_kmh <= 0:
                raise ValueError("avg_speed_kmh must be finite and positive")
            time_matrix = np.asarray(dist_matrix) / avg_speed_kmh

        driving_time_h = get_route_cost(time_matrix, tour)
        service_time_total_h = sum(node != 0 for node in tour) * service_time_h
        time_spent_h = driving_time_h + service_time_total_h

        # TCPTP validates each trip against the shift. Other problems keep the tour.
        from logic.src.pipeline.simulations.day_setup import per_trip_shift

        shift_checked = per_trip_shift(context)
        if shift_checked and tour and len(tour) > 2:
            cur_trip_time = 0.0
            cur_load = 0.0
            cur_trip_bins = 0
            prev_node = tour[0]
            for node in tour[1:]:
                cur_trip_time += float(time_matrix[prev_node, node])
                if node == 0:
                    trip_time = cur_trip_time + (cur_trip_bins * service_time_h)
                    if trip_time > shift_hours + 1e-6:
                        raise AssertionError(
                            f"TCPTP violation: trip duration {trip_time:.4f}h exceeds shift budget {shift_hours:.4f}h"
                        )
                    if cur_load > getattr(context, "vehicle_capacity", float("inf")) + 1e-6:
                        raise AssertionError("TCPTP violation: trip exceeds vehicle capacity")
                    cur_trip_time = 0.0
                    cur_load = 0.0
                    cur_trip_bins = 0
                else:
                    cur_trip_bins += 1
                    if getattr(context, "vehicle_capacity", None) is not None:
                        # Feasibility is physical: validate against the TRUE fill
                        # (real_c), not the noisy observation (c). Using c could
                        # pass a physically over-capacity trip (silent kg/profit
                        # corruption) or fail a feasible one (killing the sample).
                        cur_load += float(bins.real_c[node - 1])
                prev_node = node

        # Per-trip payload telemetry, for every problem type. TCPTP rejects violations above;
        # for the PTP variants the simulator still executes the tour, so record and warn.
        _vehicle_capacity = getattr(context, "vehicle_capacity", None)
        capacity_pct = float(_vehicle_capacity) if _vehicle_capacity is not None else float("inf")
        try:
            loads_pct, loads_kg, violations = capacity_report(tour, bins, capacity_pct)
        except (AttributeError, IndexError, TypeError, ValueError) as exc:  # telemetry must never stop a collection
            logger.debug("trip-load telemetry unavailable: %s", exc)
            loads_pct, loads_kg, violations = [], [], 0
        context.trip_loads_pct = loads_pct
        context.trip_loads_kg = loads_kg
        context.capacity_violations = violations
        if violations and not shift_checked:
            logger.warning(
                "%d trip(s) exceed the vehicle capacity (%.1f %% of a bin): loads %s",
                violations,
                capacity_pct,
                [round(x, 1) for x in loads_pct],
            )

        # Only mutate bins after every trip has passed validation.
        collected, total_collected, ncol, profit = bins.collect(tour, raw_km)

        # 5. Update context with definitive source-of-truth metrics for LogAction
        context.cost = raw_km
        context.collected = collected
        context.total_collected = total_collected
        context.ncol = ncol
        context.profit = profit
        context.time_spent = time_spent_h
        context.metrics_phase = "final"
        _record_thompson_reward(context, profit)
