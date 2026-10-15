"""WP-08 day-loop contract: shared fills, KPI keys, pipeline order, torch-free specs."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from logic.src.pipeline.simulations.pipeline import PipelineError, resolve_pipeline

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_default_pipeline_order():
    """Fill runs before packing and logging is last."""
    classes = resolve_pipeline(None)
    names = [cls.__name__ for cls in classes]
    assert names[0] == "FillAction"
    assert names.index("PackingAction") < names.index("RoutingAction")
    assert names.index("RoutingAction") < names.index("RouteImprovementAction")
    assert names[-1] == "LogAction"


def test_logging_before_collection_is_rejected():
    """Validation fails before any action would run."""
    with pytest.raises(PipelineError):
        resolve_pipeline(["logging", "collection"])


def test_two_policies_share_the_fill_rng(monkeypatch):
    """Policy identity does not reseed the bin generator."""
    from logic.src.pipeline.simulations.day_context import SimulationDayContext, run_day

    seen = {}

    class RecordFill:
        timed = False
        writes = ()

        def execute(self, context):
            seen[context.policy_name] = float(context.bins.rng.random())

    monkeypatch.setattr(
        "logic.src.pipeline.simulations.pipeline.resolve_pipeline",
        lambda names=None: [RecordFill],
    )
    for policy in ("alns", "hgs"):
        context = SimulationDayContext(
            policy_name=policy,
            seed=1234,
            sample_id=2,
            day=4,
            problem="ptp",
            bins=SimpleNamespace(),
        )
        run_day(context)
    assert seen["alns"] == seen["hgs"]


def test_daily_record_drops_reward_and_splits_day_counts():
    """Executed days report profit, days_simulated, and collection_days."""
    import pandas as pd
    from logic.src.pipeline.simulations.day_context import get_daily_results

    coords = pd.DataFrame({"ID": [0, 11, 12]})
    full = get_daily_results(10, 2, 4, [0, 1, 2, 0], 3, 1, 0.5, coords, 6, 0.2)
    assert "reward" not in full
    assert "days" not in full
    assert full["profit"] == 6
    assert full["days_simulated"] == 1
    assert full["collection_days"] == 1
    empty = get_daily_results(10, 2, 4, [0], 3, 1, 0.5, coords, 6, 0.2)
    assert empty["profit"] == 0
    assert empty["collection_days"] == 0
    assert empty["kg"] == 0


def test_converter_maps_days_and_drops_reward(tmp_path):
    """The archive tool writes a new tree and leaves the source unchanged."""
    import json

    from tools.convert_sim_outputs import convert_tree

    source = tmp_path / "in"
    dest = tmp_path / "out"
    source.mkdir()
    (source / "sample.json").write_text(json.dumps({"reward": 3, "days": 9, "profit": 1, "daily": [{"reward": 1}]}))
    written = convert_tree(str(source), str(dest), days_simulated=31)
    assert written == 1
    original = json.loads((source / "sample.json").read_text())
    assert original["reward"] == 3
    converted = json.loads((dest / "sample.json").read_text())
    assert "reward" not in converted
    assert converted["collection_days"] == 9
    assert converted["days_simulated"] == 31
    assert "reward" not in converted["daily"][0]


def test_released_specs_import_without_torch():
    """ptp/cptp/tcptp/cvrp resolve without importing PyTorch."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "from logic.src.envs.tasks.spec import PROBLEMS\n"
        "ptp = PROBLEMS.get('ptp')\n"
        "cvrp = PROBLEMS.get('cvrp')\n"
        "tcptp = PROBLEMS.get('tcptp')\n"
        "assert ptp.expand_pool is True\n"
        "assert cvrp.expand_pool is False\n"
        "assert tcptp.per_trip_shift is True\n"
        "assert 'torch' not in sys.modules or sys.modules['torch'] is None\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.check_call([sys.executable, "-c", code], cwd=root, env=env)


@pytest.mark.parametrize(
    "steps",
    [
        ["fill", "packing", "routing", "time_constraints", "route_improvement", "collection", "logging"],
        ["fill", "packing", "routing", "time_constraints", "collection", "route_improvement", "logging"],
    ],
)
def test_improvement_cannot_follow_repair_or_collection(steps):
    with pytest.raises(PipelineError):
        resolve_pipeline(steps)


def test_invalid_pipeline_does_not_change_rng_or_expand_pool(monkeypatch):
    from logic.src.envs.tasks.common.expand_pool import current_expand_pool
    from logic.src.pipeline.simulations.day_context import SimulationDayContext, run_day

    bins = SimpleNamespace(rng="unchanged")
    ctx = SimulationDayContext(problem="cvrp", bins=bins, day_pipeline=["logging", "collection"])
    before = current_expand_pool()
    with pytest.raises(PipelineError):
        run_day(ctx)
    assert bins.rng == "unchanged"
    assert current_expand_pool() == before


def test_neural_import_does_not_replace_classical_spec():
    from logic.src.envs.tasks.profitable_tour.tcptp import TCPTP_SPEC
    from logic.src.envs.tasks.spec import PROBLEMS

    assert PROBLEMS.get("tcptp") is TCPTP_SPEC
    assert TCPTP_SPEC.per_trip_shift is True


def test_converter_daily_horizons_and_std_are_not_whole_run_horizons():
    from tools.convert_sim_outputs import _convert_payload

    payload = {
        "samples": {"0": {"days": 2, "kg": 10}},
        "daily": {"0": {"day": [1, 2], "kg": [4, 6], "tour": [[0, 1, 0], [0]]}},
        "std": {"days": 0, "kg": 3},
    }
    converted = _convert_payload(payload, 31)
    assert converted["samples"]["0"]["days_simulated"] == 31
    assert converted["daily"]["0"]["days_simulated"] == [1, 1]
    assert converted["daily"]["0"]["collection_days"] == [1, 0]
    assert converted["std"]["days_simulated"] == 0
    row = _convert_payload({"daily": [{"kg": 4, "tour": [0, 1, 0]}]}, 31)
    assert row["daily"][0]["days_simulated"] == 1


def test_converter_refuses_overlapping_trees(tmp_path):
    from tools.convert_sim_outputs import convert_tree

    with pytest.raises(ValueError, match="overlap"):
        convert_tree(str(tmp_path), str(tmp_path / "converted"), 31)
