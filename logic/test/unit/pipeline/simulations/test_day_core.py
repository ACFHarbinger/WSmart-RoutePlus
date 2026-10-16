import pytest
from logic.src.pipeline.simulations.day_context import run_day

pytestmark = [pytest.mark.unit, pytest.mark.fast]


class TestDay:
    """Class for daily execution tests."""

    @pytest.mark.unit
    def test_run_day(self, make_day_context, mock_run_day_deps):
        """Test the run_day orchestrator."""
        context = make_day_context()
        # Ensure we have some actions to run
        context.new_data = mock_run_day_deps["new_data"]
        context.coords = mock_run_day_deps["coords"]
        context.bins = mock_run_day_deps["bins"]

        # Mocking actions to avoid complex logic checks here
        updated_context = run_day(context)
        assert updated_context.day == context.day


def test_active_tracker_receives_canonical_executed_metrics(make_day_context, mock_run_day_deps, monkeypatch):
    from unittest.mock import MagicMock

    tracker = MagicMock()
    monkeypatch.setattr("logic.src.pipeline.simulations.actions.logging.get_sim_tracker", lambda *args: tracker)
    context = make_day_context()
    context.new_data = mock_run_day_deps["new_data"]
    context.coords = mock_run_day_deps["coords"]
    context.bins = mock_run_day_deps["bins"]
    result = run_day(context)
    metrics = tracker.log_day.call_args.args[1]
    assert "reward" not in metrics
    assert metrics == {key: result.daily_log[key] for key in metrics}
    assert "profit" in metrics and "km" in metrics
