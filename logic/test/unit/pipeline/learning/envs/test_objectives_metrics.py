"""Economic, aggregation and optional-dependency contracts for P-21/P-42."""

import json
import math
import subprocess
import sys

import numpy as np
import pytest
from logic.src.envs.common.metrics import (
    DEFAULT_METRICS,
    METRICS,
    DayOutcome,
    Metric,
    MetricRegistry,
    SampleResult,
    summarize_samples,
)
from logic.src.envs.common.objectives import Economics, distance_objective, profit_objective


def test_shared_modules_import_without_neural_dependencies():
    code = """
import importlib.abc
import sys
class NoNeural(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchrl', 'tensordict', 'torch_geometric'}:
            raise ImportError('neural dependency requested: ' + fullname)
sys.meta_path.insert(0, NoNeural())
from logic.src.envs.common.metrics import METRICS
from logic.src.envs.common.objectives import profit_objective
from logic.src.envs.spec import PROBLEMS
assert 'cptptc' in PROBLEMS
assert profit_objective(10, 2, revenue_kg=3, cost_km=4) == 22
assert 'reward' not in METRICS
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_economics_uses_injected_physical_rates_and_keeps_counts_separate():
    outcome = DayOutcome(
        100,
        12,
        overflows=3,
        collection_count=7,
        policy_time_s=2,
        day_wall_time_s=9,
        time_spent_h=4,
        collection_day=True,
    )
    day = outcome.metrics(Economics(0.2, 0.5))
    assert day["profit"] == 14
    assert day["overflows"] == 3
    assert day["ncol"] == 7
    assert (day["time"], day["day_wall_time"], day["time_spent"]) == (2, 9, 4)
    assert day["days_simulated"] == day["collection_days"] == 1
    assert "reward" not in day and "days" not in day
    json.dumps(day, allow_nan=False)


def test_numpy_objective_preserves_batch_and_distance_objective_sign():
    kg = np.array([10.0, 15.0])
    km = np.array([2.0, 4.0])
    np.testing.assert_array_equal(profit_objective(kg, km, revenue_kg=2, cost_km=3), [14, 18])
    np.testing.assert_array_equal(distance_objective(km), [-2, -4])


def test_tensor_objective_preserves_gradients_dtype_and_shape():
    import torch

    kg = torch.tensor([10.0, 15.0], dtype=torch.float64, requires_grad=True)
    km = torch.tensor([2.0, 4.0], dtype=torch.float64)
    reward = profit_objective(kg, km, revenue_kg=2, cost_km=3)
    assert reward.shape == kg.shape and reward.dtype == kg.dtype and reward.device == kg.device
    assert torch.equal(reward, torch.tensor([14.0, 18.0], dtype=torch.float64))
    reward.sum().backward()
    assert torch.equal(kg.grad, torch.full_like(kg, 2))


def test_day_ratio_then_equal_weight_sample_statistics_not_pooled_ratio():
    first = METRICS.aggregate_days([{"kg": 10, "km": 1}, {"kg": 90, "km": 99}], ("kg", "km", "kg/km"))
    second = METRICS.aggregate_days([{"kg": 9, "km": 1}], ("kg", "km", "kg/km"))
    assert first["kg/km"] == 1  # not mean([10, 90/99])
    result = summarize_samples([SampleResult("a", first), SampleResult("b", second)], ("kg/km",))
    assert result.mean["kg/km"] == 5  # not pooled 109/101
    assert result.std["kg/km"] == pytest.approx(math.sqrt(32))
    assert result.n_ratio_defined == {"kg/km": 2}


def test_failures_are_excluded_and_zero_distance_keeps_compatibility():
    partial = SampleResult("broken", {"kg": 999}, "failed", failed_day=4, exception_type="ValueError")
    result = summarize_samples([partial, SampleResult("zero", {"kg": 0, "km": 0})], ("kg/km",))
    assert (result.n_success, result.n_failed) == (1, 1)
    assert result.mean["kg/km"] == result.std["kg/km"] == 0
    assert result.n_ratio_defined["kg/km"] == 0
    assert partial.metrics == {"kg": 999} and partial.failed_day == 4
    empty = summarize_samples([partial], ("kg/km",))
    assert empty.mean["kg/km"] is empty.std["kg/km"] is None


def test_unknown_missing_nonfinite_and_duplicate_records_fail_loudly():
    with pytest.raises(KeyError):
        METRICS.aggregate_days([{"kg": 1}], ("kg/km",))
    with pytest.raises(ValueError, match="Non-finite"):
        METRICS.aggregate_days([{"kg": float("nan")}], ("kg",))
    sample = SampleResult("same", {"kg": 1})
    with pytest.raises(ValueError, match="Duplicate sample"):
        summarize_samples([sample, sample], ("kg",))
    with pytest.raises(ValueError, match="Duplicate metric"):
        METRICS.register(METRICS["kg"])
    with pytest.raises(ValueError, match="dependency"):
        MetricRegistry([Metric("ratio", "x", "max", "Ratio", "ratio_of_sums", "x", "y")])
    assert "time_spent" not in DEFAULT_METRICS
    assert "time_spent" not in DayOutcome(0, 0).metrics(Economics(1, 1))


def test_asymmetric_multitrip_tour_evaluation_counts_every_depot_leg():
    import torch
    from logic.src.envs.core.ptp import evaluate_profitable_tour

    dm = torch.tensor([[0.0, 1.0, 2.0], [3.0, 0.0, 4.0], [6.0, 5.0, 0.0]], dtype=torch.float64)
    data = {
        "dm": dm,
        "waste": torch.tensor([[0.0, 10.0, 20.0]], dtype=torch.float64),
        "revenue_kg": 2.0,
        "cost_km": 3.0,
    }
    out = evaluate_profitable_tour(data, torch.tensor([[0, 1, 0, 2, 0]]))
    assert out["length"].item() == 12
    assert out["reward"].item() == 24
    assert out["cost"].item() == -24
    assert out["reward"].dtype == torch.float64


@pytest.mark.parametrize(
    "actions,expected", [([0, 1, 0, 2, 0], 12), ([1, 2], 11), ([0, 1, 0, 0], 4), ([1], 4), ([0], 0), ([], 0)]
)
def test_shared_road_length_handles_open_padded_and_empty_paths(actions, expected):
    import torch
    from logic.src.envs.core.ptp import closed_tour_length
    from logic.src.envs.cvrp import evaluate_cvrp_tour

    dm = torch.tensor([[0.0, 1.0, 2.0], [3.0, 0.0, 4.0], [6.0, 5.0, 0.0]])
    pi = torch.tensor([actions], dtype=torch.long)
    assert closed_tour_length(dm, pi).item() == expected
    data = {"dm": dm, "demand": torch.tensor([[0.0, 0.1, 0.2]]), "capacity": 1.0}
    result = evaluate_cvrp_tour(data, pi)
    assert result["length"].item() == expected
    assert result["reward"].item() == -expected


def test_problem_specs_keep_tensor_objectives_and_canonical_kpis():
    import torch
    from logic.src.envs.spec import PROBLEMS, ProblemRegistry

    assert not ProblemRegistry().keys()
    for name in ("ptp", "cptp", "cptptc", "cvrp"):
        spec = PROBLEMS[name]
        assert set(spec.metrics) <= set(METRICS)
        assert "reward" not in spec.metrics
        length = torch.tensor([1.0, 2.0])
        reward = (
            spec.objective(distance_km=length) if name == "cvrp" else spec.objective(torch.tensor([3.0, 4.0]), length)
        )
        assert reward.shape == length.shape
    assert PROBLEMS["thop"].name == "thop"


def test_overflowed_aggregate_is_rejected_instead_of_serializing_infinity():
    with pytest.raises(ValueError, match="Non-finite aggregate"):
        METRICS.aggregate_days([{"kg": 1e308}, {"kg": 1e308}], ("kg",))


def test_new_metric_registration_defines_metadata_and_daily_extraction():
    registry = MetricRegistry(
        [
            Metric("co2", "kg CO2", "min", "Emissions", per_day=lambda day: day["km"] * 0.2),
            Metric("fill", "%", "none", "Mean fill", aggregate="mean"),
        ]
    )
    result = registry.aggregate_days([{"km": 10, "fill": 40}, {"km": 20, "fill": 60}], ("co2", "fill"))
    assert result == {"co2": 6.0, "fill": 50.0}


def test_ratio_only_selection_retains_dependencies_for_sample_summary():
    totals = METRICS.aggregate_days([{"kg": 12, "km": 3}], ("kg/km",))
    summary = summarize_samples([SampleResult("one", totals)], ("kg/km",))
    assert summary.mean["kg/km"] == 4
    assert summary.n_ratio_defined["kg/km"] == 1
