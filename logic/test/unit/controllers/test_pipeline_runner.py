"""Regression tests for pipeline-runner tracking outcomes."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from logic.controllers.jobs.pipeline_runner import run_data_generation

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        tracking=SimpleNamespace(verbose=False),
        experiment_name=None,
        data=SimpleNamespace(problem="ptp"),
    )


def test_data_generation_marks_failed_run_on_generator_error():
    """A failing generator must not leave a completed tracking record."""
    run = MagicMock()
    with patch("logic.src.tracking.init"), patch("logic.src.tracking.get_active_run", return_value=run), patch(
        "logic.src.data.generators.generate_datasets", side_effect=RuntimeError("bad source data")
    ), pytest.raises(RuntimeError, match="bad source data"):
        run_data_generation(_cfg())

    run.set_tag.assert_called_once_with("status", "failed")
    run.flush.assert_called_once()
