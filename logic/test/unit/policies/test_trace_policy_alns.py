"""A traced ALNS solve matches an untraced solve on the same seed."""

import pytest
from logic.src.policies.trace_runner import build_alns_trace, make_instance, routes_equal, solve_alns
from logic.src.tracking.execution_trace import validate_trace

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_traced_alns_matches_untraced_solve():
    graph_size = 4
    seed = 2
    acceptance = "oi"
    iterations = 8
    dist, wastes, _coords = make_instance(graph_size, seed)
    plain = solve_alns(dist, wastes, seed, acceptance, iterations)
    trace, traced = build_alns_trace(
        "ptp",
        graph_size,
        seed,
        acceptance=acceptance,
        packing="regular",
        max_iterations=iterations,
    )
    document = trace.to_dict()
    validate_trace(document)
    assert routes_equal(plain.routes, traced.routes)
    assert plain.profit == pytest.approx(traced.profit)
    assert document["final"]["routes"] == [list(route) for route in traced.routes]
    assert document["events"][1]["name"] == "alns"
    assert document["events"][1]["kind"] == "stage"
    assert "interior" not in document["events"][1]
    assert "oi" in {event["name"] for event in document["events"]}
    assert document["policies"]["acceptance"] == "oi"
    assert document["policies"]["packing"] == "regular"
