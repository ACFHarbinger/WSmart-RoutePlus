"""Regression tests for CLI aliases at the Hydra dispatch boundary."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import main as main_module
import pytest
from logic.controllers.hydra_dispatch import _run_task

pytestmark = [pytest.mark.unit, pytest.mark.fast]


@pytest.mark.parametrize(("alias", "canonical"), [("evaluation", "eval"), ("sim_hpo", "hpo_sim")])
def test_main_normalizes_hydra_task_aliases(alias: str, canonical: str):
    """Documented aliases compose an existing Hydra task group."""
    captured_argv: list[str] = []
    with patch.object(sys, "argv", ["main.py", alias]), patch.object(main_module, "hydra_entry_point") as dispatch:
        dispatch.side_effect = lambda: captured_argv.extend(sys.argv)
        main_module.main()

    dispatch.assert_called_once()
    assert captured_argv == ["main.py", f"tasks={canonical}"]


def test_hydra_dispatch_accepts_evaluation_alias() -> None:
    """``tasks=evaluation`` must reach eval even when main.py did not rewrite it."""
    cfg = SimpleNamespace(task="evaluation")
    with patch("logic.controllers.jobs.pipeline_runner.run_evaluation", return_value=1.5) as mocked:
        result = _run_task(cfg)

    mocked.assert_called_once_with(cfg)
    assert result == 1.5
