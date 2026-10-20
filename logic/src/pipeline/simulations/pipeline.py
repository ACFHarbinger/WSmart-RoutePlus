"""Validated day pipeline.

The default order is fill, packing, routing, improvement, repair, collection,
then logging. A configured order may drop a step. It may not run a step before
the step that produces its inputs, and it may not treat a preliminary route
metric as final. Validation fails before any action runs.
"""

from typing import Dict, List, Optional, Sequence, Set, Tuple, Type

from logic.src.pipeline.simulations.actions.base import SimulationAction


class PipelineError(ValueError):
    """The requested day pipeline is not a valid dependency order."""


# Names accepted in ``sim.day_pipeline``.
ACTION_NAMES: Tuple[str, ...] = (
    "fill",
    "packing",
    "routing",
    "route_improvement",
    "time_constraints",
    "collection",
    "logging",
)

DEFAULT_PIPELINE: Tuple[str, ...] = ACTION_NAMES

# Earlier name must appear before later name when both are present.
_ORDER: Tuple[Tuple[str, str], ...] = (
    ("fill", "packing"),
    ("fill", "routing"),
    ("fill", "collection"),
    ("packing", "routing"),
    ("routing", "route_improvement"),
    ("route_improvement", "time_constraints"),
    ("route_improvement", "collection"),
    ("route_improvement", "logging"),
    ("routing", "time_constraints"),
    ("routing", "collection"),
    ("time_constraints", "collection"),
    ("collection", "logging"),
)

# Fields the day context already holds before the first action.
_PROVIDED = frozenset(
    {
        "bins",
        "day",
        "coords",
        "distance_matrix",
        "config",
        "problem",
        "problem_spec",
        "vehicle_capacity",
        "shift_hours",
        "avg_speed_kmh",
        "service_time_h",
        "time_matrix",
        "mandatory",
        "policy_name",
        "full_policy",
        "sample_id",
        "seed",
        "n_vehicles",
        "graph_size",
        "time_windows",
        "routing_allowed",
    }
)


def _registry() -> Dict[str, Type[SimulationAction]]:
    """Import action classes lazily so a pipeline check does not import solvers."""
    from logic.src.pipeline.simulations.actions import (
        CollectAction,
        FillAction,
        LogAction,
        PackingAction,
        RouteImprovementAction,
        RoutingAction,
    )
    from logic.src.pipeline.simulations.actions.time_constraints import TimeConstraintAction

    return {
        "fill": FillAction,
        "packing": PackingAction,
        "routing": RoutingAction,
        "route_improvement": RouteImprovementAction,
        "time_constraints": TimeConstraintAction,
        "collection": CollectAction,
        "logging": LogAction,
    }


def _declared(action: Type[SimulationAction], name: str) -> Tuple[str, ...]:
    """Read a class tuple. Mocks and missing attributes count as empty."""
    value = getattr(action, name, ())
    if isinstance(value, (tuple, list)) and all(isinstance(item, str) for item in value):
        return tuple(value)
    return ()


def resolve_pipeline(names: Optional[Sequence[str]] = None) -> List[Type[SimulationAction]]:
    """Return action classes in a validated order.

    ``names`` None selects the default pipeline. An unknown name, a repeated
    name, or a dependency cycle against the declared reads/writes fails here.
    """
    selected = list(DEFAULT_PIPELINE if names is None else names)
    registry = _registry()
    unknown = [name for name in selected if name not in registry]
    if unknown:
        raise PipelineError(f"Unknown day-pipeline step(s): {unknown}. Known: {list(ACTION_NAMES)}")
    if len(selected) != len(set(selected)):
        raise PipelineError(f"Day pipeline repeats a step: {selected}")

    index = {name: pos for pos, name in enumerate(selected)}
    for earlier, later in _ORDER:
        if earlier in index and later in index and index[earlier] > index[later]:
            raise PipelineError(f"{later} cannot run before {earlier}")

    written = set(_PROVIDED)
    finalized: Set[str] = set()
    saw_opaque = False
    for name in selected:
        action = registry[name]
        reads = _declared(action, "reads")
        writes = _declared(action, "writes")
        requires_final = _declared(action, "requires_final")
        finalizes = _declared(action, "finalizes")
        # A test double with no contract may stand in for a writer. Real steps
        # still fail when every preceding step declared its outputs.
        has_contract = bool(reads or writes or requires_final or finalizes)
        if has_contract and not saw_opaque:
            missing = [key for key in reads if key not in written]
            if missing:
                raise PipelineError(f"{name} reads {missing} before any step writes them")
            premature = [key for key in requires_final if key not in finalized]
            if premature:
                raise PipelineError(f"{name} requires final {premature} before collection publishes them")
        if not has_contract:
            saw_opaque = True
        written.update(writes)
        # A later mutation invalidates a previously finalized field.
        finalized.difference_update(writes)
        finalized.update(finalizes)
    return [registry[name] for name in selected]
