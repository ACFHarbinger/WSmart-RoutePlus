"""
Action for waste collection execution.

Attributes:
    CollectAction: Command to execute waste collection.

Example:
    >>> # action = CollectAction()
    >>> # action.execute(context)
"""

from typing import Any, Dict

import numpy as np

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
        tour = list(context["tour"])
        if not tour or tour[0] != 0:
            tour.insert(0, 0)
        if tour[-1] != 0 or len(tour) == 1:
            tour.append(0)
        context["tour"] = tour

        # 1. METRIC CONSISTENCY: Always re-calculate KM from the final tour
        # This combines mandatory selection, construction, and any route improvements.
        # We use the raw distance matrix from the context (guaranteed to be KM).
        dist_matrix = context["distance_matrix"]
        raw_km = get_route_cost(dist_matrix, tour)

        # 3. Calculate operational time spent (travel time + per-bin service time)
        avg_speed_kmh = float(context.get("avg_speed_kmh", 35.0))
        service_time_h = float(context.get("service_time_h", 1.5 / 60.0))
        shift_hours = float(context.get("shift_hours", 7.0))
        time_matrix = context.get("time_matrix")
        if time_matrix is None:
            if not np.isfinite(avg_speed_kmh) or avg_speed_kmh <= 0:
                raise ValueError("avg_speed_kmh must be finite and positive")
            time_matrix = np.asarray(dist_matrix) / avg_speed_kmh

        driving_time_h = get_route_cost(time_matrix, tour)
        service_time_total_h = sum(node != 0 for node in tour) * service_time_h
        time_spent_h = driving_time_h + service_time_total_h

        # 4. If problem is CTOP, validate per-trip constraints (capacity + shift duration)
        problem = str(context.get("problem", "vrpp") or "vrpp").lower()
        if problem == "ctop" and tour and len(tour) > 2:
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
                            f"CTOP violation: trip duration {trip_time:.4f}h exceeds shift budget {shift_hours:.4f}h"
                        )
                    if cur_load > context.get("vehicle_capacity", float("inf")) + 1e-6:
                        raise AssertionError("CTOP violation: trip exceeds vehicle capacity")
                    cur_trip_time = 0.0
                    cur_load = 0.0
                    cur_trip_bins = 0
                else:
                    cur_trip_bins += 1
                    if "vehicle_capacity" in context:
                        cur_load += float(bins.c[node - 1])
                prev_node = node

        # Only mutate bins after every trip has passed validation.
        collected, total_collected, ncol, profit = bins.collect(tour, raw_km)

        # 5. Update context with definitive source-of-truth metrics for LogAction
        context["cost"] = raw_km
        context["collected"] = collected
        context["total_collected"] = total_collected
        context["ncol"] = ncol
        context["profit"] = profit
        context["time_spent"] = time_spent_h
