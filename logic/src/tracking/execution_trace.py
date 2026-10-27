"""Versioned execution trace for the Studio algorithm visualizer.

A trace records the inputs that enter a policy, the stages and operators that
touch an intermediate solution or the incumbent, and the final routes. Writing
a trace is opt-in. Callers that never construct an :class:`ExecutionTrace` do
not allocate one.

The document is JSON. ``schema_version`` is ``1``.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

SCHEMA_VERSION = 1
EVENT_KINDS = ("input", "stage", "operator", "incumbent", "output")
PathLike = Union[str, Path]


def routes_hash(routes: Sequence[Sequence[int]]) -> str:
    """Stable short hash of a route set. The same tours always hash the same."""
    payload = json.dumps([list(route) for route in routes], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def solution_record(
    routes: Sequence[Sequence[int]],
    objective: Optional[float] = None,
    load: Optional[float] = None,
    feasible: Optional[bool] = None,
) -> Dict[str, Any]:
    """Solution snapshot: routes, objective, load, feasibility, and a hash.

    The hash covers the routes only, so an objective change with the same
    tours stays visible without looking like a different tour.
    """
    copied = [list(route) for route in routes]
    return {
        "routes": copied,
        "objective": None if objective is None else float(objective),
        "load": None if load is None else float(load),
        "feasible": feasible,
        "unchanged_hash": routes_hash(copied),
    }


class ExecutionTrace:
    """In-memory trace. ``add`` appends one event and returns its id."""

    def __init__(
        self,
        problem: str,
        policies: Dict[str, Optional[str]],
        seed: Optional[int],
        instance: Dict[str, Any],
    ) -> None:
        self.problem = problem
        self.policies = dict(policies)
        self.seed = seed
        self.instance = dict(instance)
        self.events: List[Dict[str, Any]] = []
        self.final: Optional[Dict[str, Any]] = None

    def add(
        self,
        name: str,
        kind: str,
        parent_id: Optional[str] = None,
        changed_solution: bool = False,
        arguments: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        after: Optional[Dict[str, Any]] = None,
        interior: Optional[str] = None,
    ) -> str:
        """Append an event. ``parent_id`` must already exist on this trace."""
        if kind not in EVENT_KINDS:
            raise ValueError(f"unknown trace event kind {kind!r}")
        if parent_id is not None and parent_id not in {event["id"] for event in self.events}:
            raise ValueError(f"parent {parent_id!r} is not an earlier event")
        event_id = f"e{len(self.events)}"
        event: Dict[str, Any] = {
            "id": event_id,
            "parent_id": parent_id,
            "name": name,
            "kind": kind,
            "changed_solution": bool(changed_solution),
            "arguments": {} if arguments is None else dict(arguments),
            "before": before,
            "after": after,
        }
        if interior is not None:
            event["interior"] = interior
        self.events.append(event)
        return event_id

    def mark_uninstrumented(self, name: str, parent_id: Optional[str] = None) -> str:
        """Record a stage whose interior operators are not traced yet."""
        return self.add(
            name=name,
            kind="stage",
            parent_id=parent_id,
            changed_solution=False,
            interior="not_instrumented",
        )

    def set_final(self, routes: Sequence[Sequence[int]], objective: Optional[float] = None) -> Dict[str, Any]:
        """Store the routes the solver returned. This does not add an event."""
        self.final = solution_record(routes, objective=objective)
        return self.final

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready document."""
        return {
            "schema_version": SCHEMA_VERSION,
            "problem": self.problem,
            "policies": self.policies,
            "seed": self.seed,
            "instance": self.instance,
            "events": self.events,
            "final": self.final,
        }

    def write(self, path: PathLike) -> None:
        """Write the document as UTF-8 JSON."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_trace(path: PathLike) -> Dict[str, Any]:
    """Read a trace document from disk."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("trace document must be a JSON object")
    return document


def validate_trace(document: Dict[str, Any]) -> None:
    """Raise ``ValueError`` when the document breaks schema version 1.

    Parents must point at an earlier event. The final solution is required.
    """
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    events = document.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("trace has no events")
    seen: List[str] = []
    for event in events:
        event_id = event.get("id")
        if not isinstance(event_id, str):
            raise ValueError("event id must be a string")
        parent_id = event.get("parent_id")
        if parent_id is not None and parent_id not in seen:
            raise ValueError(f"parent {parent_id!r} is not an earlier event")
        if event.get("kind") not in EVENT_KINDS:
            raise ValueError(f"bad event kind on {event_id}")
        seen.append(event_id)
    final = document.get("final")
    if not isinstance(final, dict) or "routes" not in final:
        raise ValueError("trace final solution is missing")
