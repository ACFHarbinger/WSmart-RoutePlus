"""Schema checks for the execution-trace document."""

import pytest
from logic.src.tracking.execution_trace import (
    SCHEMA_VERSION,
    ExecutionTrace,
    load_trace,
    routes_hash,
    solution_record,
    validate_trace,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_parent_must_be_an_earlier_event():
    trace = ExecutionTrace("ptp", {"routing": "alns"}, seed=1, instance={"n_nodes": 3})
    with pytest.raises(ValueError, match="parent"):
        trace.add("later", "stage", parent_id="e9")


def test_round_trip_keeps_order_and_parents(tmp_path):
    trace = ExecutionTrace(
        "ptp",
        {"packing": "regular", "routing": "alns", "route_improvement": None, "acceptance": "oi"},
        seed=1,
        instance={"n_nodes": 3, "depot": [0.0, 0.0], "capacity": 10.0},
    )
    root = trace.add("instance", "input")
    middle = trace.mark_uninstrumented("alns", parent_id=root)
    routes = [[0, 1, 0], [0, 2, 0]]
    trace.add("routes", "output", parent_id=root, changed_solution=True, after=solution_record(routes, 1.5))
    trace.set_final(routes, objective=1.5)
    path = tmp_path / "trace.json"
    trace.write(path)

    document = load_trace(path)
    validate_trace(document)
    assert document["schema_version"] == SCHEMA_VERSION
    assert [event["id"] for event in document["events"]] == ["e0", "e1", "e2"]
    assert document["events"][1]["parent_id"] == "e0"
    assert document["events"][1]["interior"] == "not_instrumented"
    assert document["events"][2]["parent_id"] == "e0"
    assert middle == "e1"
    assert document["final"]["unchanged_hash"] == routes_hash(routes)


def test_validate_rejects_a_forward_parent():
    document = {
        "schema_version": SCHEMA_VERSION,
        "events": [
            {"id": "e0", "parent_id": "e1", "kind": "input", "name": "in"},
            {"id": "e1", "parent_id": None, "kind": "output", "name": "out"},
        ],
        "final": {"routes": [[0, 0]]},
    }
    with pytest.raises(ValueError, match="parent"):
        validate_trace(document)
