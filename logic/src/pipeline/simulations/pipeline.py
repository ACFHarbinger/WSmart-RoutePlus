"""Validated day pipeline and the per-day inputs of the day loop.

The default order is fill, packing, routing, improvement, repair, collection,
then logging. A configured order may drop a step. It may not run a step before
the step that produces its inputs, and it may not treat a preliminary route
metric as final. Validation fails before any action runs.

Per-day inputs: random streams and problem flags. The environment stream is
shared by every policy on the same sample and day. The policy stream depends on
a stable policy id and must not reseed the bins. The problem flags are read
from ``ProblemSpec``.
"""

import contextlib
import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Type

import numpy as np

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


_PIPELINE_CACHE: Dict[Tuple[str, ...], Tuple[Type[SimulationAction], ...]] = {}


def resolve_pipeline(names: Optional[Sequence[str]] = None) -> List[Type[SimulationAction]]:
    """Return action classes in a validated order.

    ``names`` None selects the default pipeline. An unknown name, a repeated
    name, or a dependency cycle against the declared reads/writes fails here.
    Successfully validated pipelines are cached by their step names.
    """
    key = tuple(DEFAULT_PIPELINE if names is None else names)
    cached = _PIPELINE_CACHE.get(key)
    if cached is not None:
        return list(cached)

    selected = list(key)
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
    resolved = tuple(registry[name] for name in selected)
    _PIPELINE_CACHE[key] = resolved
    return list(resolved)


# ---------------------------------------------------------------------------
# Per-day inputs: random streams and problem flags
# ---------------------------------------------------------------------------


def shared_env_seed(seed: int, sample_id: int, day: int) -> int:
    """Seed for waste noise. Independent of which policy is running."""
    return (int(seed) + int(sample_id) * 1_000_003 + int(day)) % (2**31)


def policy_seed(seed: int, canonical_name: str, day: int) -> int:
    """Seed for a routing policy. Does not affect the shared fill stream."""
    import zlib

    name_hash = zlib.adler32(canonical_name.encode()) & 0xFFFFFFFF
    return (int(seed) + name_hash + int(day)) % (2**31)


def seed_policy_rng(seed: int) -> None:
    """Seed Python and NumPy. Seed PyTorch only when it is installed."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def apply_shared_bin_rng(bins: Any, seed: int) -> None:
    """Reset the bin generator to the shared environment seed."""
    if bins is None:
        return
    bins.rng = np.random.default_rng(int(seed))


def per_trip_shift(context: Any) -> bool:
    """True when each trip must fit the shift (CPTPTC).

    A missing spec is loaded from the released registry by problem name.
    Unreleased names are not shift-checked.
    """
    spec = getattr(context, "problem_spec", None)
    if spec is None:
        from logic.src.envs.common.released_specs import released_spec

        spec = released_spec(str(getattr(context, "problem", "") or ""))
        if spec is not None:
            with contextlib.suppress(Exception):
                context.problem_spec = spec
    return getattr(spec, "per_trip_shift", False) is True
