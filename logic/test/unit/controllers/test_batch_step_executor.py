"""Regression tests for batch post/pre-step helpers."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from logic.controllers.manager import batch_step_executor as bse
from logic.controllers.manager.batch_job import BatchJob

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_gen_dist_matrix_step_invokes_an_existing_script():
    """The step must call the live generator script (it used to point at the removed logic/scripts/)."""
    job = BatchJob(name="j")
    args = {"area": "riomaior", "method": "osm", "check_exists": False}
    with patch.object(bse.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
        bse._step_gen_dist_matrix(args, job)

    cmd = run.call_args.args[0]
    script = Path(cmd[1])
    assert script.is_file(), f"batch step points at a missing script: {script}"
    assert script.name == "gen_dist_matrix.py"
    assert cmd[cmd.index("--dm-filepath") + 1] == "osm_distmat_plastic[riomaior].csv"
