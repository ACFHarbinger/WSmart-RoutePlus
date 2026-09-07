"""
Action for waste collection execution.

Attributes:
    CollectAction: Command to execute waste collection.

Example:
    >>> # action = CollectAction()
    >>> # action.execute(context)
"""

from typing import Any, Dict

from .base import SimulationAction


class CollectAction(SimulationAction):
    """
    Processes waste collection from bins visited in the tour.

    Attributes:
        None
    """

    def execute(self, context: Dict[str, Any]) -> None:
        """
        Execute waste collection based on the generated tour.

        Args:
            context: Shared dictionary containing simulation state.
        """
        from logic.src.policies.route_construction.other_algorithms.travelling_salesman_problem.tsp import (
            get_route_cost,
        )

        bins = context["bins"]
        tour = context["tour"]

        # 1. METRIC CONSISTENCY: Always re-calculate KM from the final tour
        # This combines mandatory selection, construction, and any route improvements.
        # We use the raw distance matrix from the context (guaranteed to be KM).
        dist_matrix = context["distance_matrix"]
        raw_km = get_route_cost(dist_matrix, tour)

        # 2. Perform collection using strictly KM-based cost
        # Bins.collect internally handles normalized revenue (collected mass * €/kg)
        # and expenses (raw_km * €/km).
        collected, total_collected, ncol, profit = bins.collect(tour, raw_km)

        # 3. Calculate operational time spent (travel time + per-bin service time)
        avg_speed_kmh = float(context.get("avg_speed_kmh", 35.0) or 35.0)
        service_time_h = float(context.get("service_time_h", 1.5 / 60.0) or (1.5 / 60.0))
        shift_hours = float(context.get("shift_hours", 7.0) or 7.0)

        driving_time_h = raw_km / avg_speed_kmh if avg_speed_kmh > 0 else 0.0
        service_time_total_h = ncol * service_time_h
        time_spent_h = driving_time_h + service_time_total_h

        # 4. If problem is CTOP, validate per-trip constraints (capacity + shift duration)
        problem = str(context.get("problem", "vrpp") or "vrpp").lower()
        if problem == "ctop" and tour and len(tour) > 2:
            cur_trip_dist = 0.0
            cur_trip_bins = 0
            prev_node = tour[0]
            for node in tour[1:]:
                cur_trip_dist += float(dist_matrix[prev_node, node])
                if node == 0:
                    trip_time = (cur_trip_dist / avg_speed_kmh) + (cur_trip_bins * service_time_h)
                    assert trip_time <= shift_hours + 1e-5, (
                        f"CTOP violation: trip duration {trip_time:.4f}h exceeds shift budget {shift_hours:.4f}h"
                    )
                    cur_trip_dist = 0.0
                    cur_trip_bins = 0
                else:
                    cur_trip_bins += 1
                prev_node = node

        # 5. Update context with definitive source-of-truth metrics for LogAction
        context["cost"] = raw_km
        context["collected"] = collected
        context["total_collected"] = total_collected
        context["ncol"] = ncol
        context["profit"] = profit
        context["time_spent"] = time_spent_h
