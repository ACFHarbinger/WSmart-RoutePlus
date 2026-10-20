"""
Multi-Phase Route Improver.

Attributes:
    MultiPhaseRouteImprover: Main class for sequential multi-phase improvement.

Example:
    >>> from logic.src.policies.route_improvement.multi_phase import MultiPhaseRouteImprover
    >>> improver = MultiPhaseRouteImprover(config=cfg)
    >>> tour, metrics = improver.process([0, 1, 2, 0], phases=["fast_tsp", "lk"])
"""

from typing import Any, List, Tuple

from logic.src.enums import PolicyTag
from logic.src.interfaces.context.search_context import ImprovementMetrics
from logic.src.interfaces.route_improvement import IRouteImprovement
from logic.src.utils.registry import ALL

from .base import RouteImproverRegistry


@ALL.register(
    PolicyTag.IMPROVEMENT,
    PolicyTag.ORCHESTRATOR,
)
@RouteImproverRegistry.register("multi_phase")
class MultiPhaseRouteImprover(IRouteImprovement):
    """Composition route improver running multiple strategies in sequence.

    Facilitates a pipeline of improvement moves, such as an augmentation phase
    followed by an intensification phase and a final polishing phase.

    Attributes:
        config (Dict[str, Any]): Internal configuration state.

    Example:
        >>> improver = MultiPhaseRouteImprover(config=cfg)
        >>> refined_tour, metrics = improver.process(tour, phases=["cheapest_insertion", "lkh"])
    """

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:
        """Run multi-phase refinement on the tour.

        Args:
            tour: Initial tour sequence.
            kwargs: Context containing:
                phases: List of improver names to run in order.
                distance_matrix: Distance lookup.
                wastes: Bin waste mass mapping.
                capacity: Maximum vehicle capacity.

        Returns:
            Tuple[List[int], ImprovementMetrics]: Refined tour and multi-phase metrics.

        Raises:
            ValueError: If fewer than 2 phases are provided or a phase name is unknown.
        """
        phase_names = (
            kwargs.get("phases", [])
            if kwargs.get("phases") is not None
            else self.config.get("phases", ["cheapest_insertion", "lkh"])
        )

        if len(phase_names) < 2:
            raise ValueError(
                f"MultiPhaseRouteImprover requires at least 2 phases, but got {len(phase_names)}. "
                f"Phases provided: {phase_names}"
            )

        metrics: ImprovementMetrics = {
            "algorithm": "MultiPhaseRouteImprover",
            "phases": [],
        }

        current_tour = tour
        for i, phase_name in enumerate(phase_names):
            p_cls = RouteImproverRegistry.get_route_improver_class(phase_name)
            if p_cls is None:
                raise ValueError(f"Unknown phase strategy at index {i}: {phase_name!r}")

            # Construct sub-processor with reference to full config dict
            p = p_cls(config=self.config)
            current_tour, p_metrics = p.process(current_tour, **kwargs)

            # Store phase-specific metrics
            metrics["phases"].append({"index": i + 1, "strategy": phase_name, "metrics": p_metrics})

        return current_tour, metrics
