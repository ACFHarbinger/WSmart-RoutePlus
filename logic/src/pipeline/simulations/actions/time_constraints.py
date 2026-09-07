"""Apply CTOP constraints after all route construction and improvement."""

from typing import Any, Dict

from logic.src.policies.route_construction.other_algorithms.travelling_salesman_problem.tsp import get_multi_tour

from .base import SimulationAction


class TimeConstraintAction(SimulationAction):
    """Insert depot returns so every CTOP trip fits capacity and time."""

    def execute(self, context: Dict[str, Any]) -> None:
        """Repair the final route before collection mutates any bin contents."""
        if str(context.get("problem", "")).lower() != "ctop":
            return
        tour = context.get("tour")
        context["tour"] = get_multi_tour(
            list(tour) if tour is not None else [],
            context["bins"].c,
            context["vehicle_capacity"],
            context["distance_matrix"],
            shift_hours=context.get("shift_hours", 7.0),
            avg_speed_kmh=context.get("avg_speed_kmh", 35.0),
            service_time_h=context.get("service_time_h", 0.025),
            time_matrix=context.get("time_matrix"),
        )
