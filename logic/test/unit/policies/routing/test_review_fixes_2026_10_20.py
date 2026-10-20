"""Surviving routing-base, registry and interface defects from the 2026-10-20 Cursor review."""

from types import SimpleNamespace

import numpy as np
import pytest

from logic.src.interfaces.context.problem_context import ProblemContext

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _kwargs(**extra):
    n = 3
    dm = np.ones((n + 1, n + 1))
    np.fill_diagonal(dm, 0.0)
    bins = SimpleNamespace(
        c=np.array([10.0, 20.0, 30.0]),
        means=np.array([5.0, 6.0, 7.0]),
        std=np.array([1.0, 1.5, 2.0]),
        max_fill=100.0,
    )
    base = {"distance_matrix": dm, "bins": bins}
    base.update(extra)
    return base


def test_problem_context_reads_the_fill_rates_bins_actually_store() -> None:
    problem = ProblemContext.from_kwargs(_kwargs(), 100.0, 1.0, 1.0)
    np.testing.assert_allclose(problem.fill_rate_means, [5.0, 6.0, 7.0])
    np.testing.assert_allclose(problem.fill_rate_stds, [1.0, 1.5, 2.0])


@pytest.mark.parametrize(
    "extra, expected",
    [
        ({}, 7),
        ({"horizon": 3}, 3),
        ({"config": {"horizon": 2}}, 2),
        ({"config": {"num_days": 5}}, 5),
        ({"config": SimpleNamespace(horizon=4)}, 4),
    ],
)
def test_problem_context_horizon_follows_the_policy(extra, expected) -> None:
    assert ProblemContext.from_kwargs(_kwargs(**extra), 100.0, 1.0, 1.0).horizon == expected


@pytest.mark.parametrize(
    "flat, expected",
    [
        ({}, 7),
        ({"horizon": 3}, 3),
        ({"num_days": 4}, 4),
        ({"policy.horizon": 5}, 5),
    ],
)
def test_policy_horizon_reads_the_flattened_keys(flat, expected) -> None:
    from logic.src.pipeline.simulations.actions.policy_config import policy_horizon

    assert policy_horizon(flat) == expected


def test_multi_period_base_passes_its_configured_horizon(monkeypatch) -> None:
    from logic.src.policies.routing.base import base_multi_period_policy as mp

    seen = {}

    def capture(cls, kwargs, capacity, revenue, cost_unit):
        seen["horizon"] = kwargs.get("horizon")
        raise RuntimeError("stop")

    monkeypatch.setattr(ProblemContext, "from_kwargs", classmethod(capture))

    class _Probe(mp.BaseMultiPeriodRoutingPolicy):
        def _run_multi_period_solver(self, *args, **kwargs):  # pragma: no cover - never reached
            raise AssertionError

        def _load_area_params(self, *args, **kwargs):
            return 100.0, 1.0, 1.0, {}

    probe = _Probe.__new__(_Probe)
    probe._config = SimpleNamespace(horizon=2)
    probe.horizon = 2
    with pytest.raises(RuntimeError, match="stop"):
        probe.execute(**_kwargs())
    assert seen["horizon"] == 2


@pytest.mark.parametrize("key", ["jgo", "jsa", "nds_brkga"])
def test_joint_policies_are_in_the_routing_manifest(key: str) -> None:
    from logic.src.policies.routing._manifest import ENTRIES

    assert key in ENTRIES


def test_joint_base_unpacks_the_area_parameter_tuple(monkeypatch) -> None:
    from logic.src.policies.packing_and_routing.base import base_joint_policy as bj
    from logic.src.pipeline.simulations import repository

    monkeypatch.setattr(
        repository, "load_area_and_waste_type_params", lambda area, waste: (80.0, 0.5, 20.0, 1.2, 2.5)
    )
    seen = {}

    class _Probe(bj.BaseJointPolicy):
        def solve_joint(self, context):
            seen["ctx"] = context
            return [], [], 0.0, 0.0

    probe = _Probe.__new__(_Probe)
    out = probe.execute(**_kwargs())
    ctx = seen["ctx"]
    assert ctx.capacity == pytest.approx(80.0 / 100.0 * 2.5 * 20.0)
    assert ctx.revenue_kg == pytest.approx(0.5)
    assert ctx.cost_per_km == pytest.approx(1.2)
    assert out[0] == [0, 0]
