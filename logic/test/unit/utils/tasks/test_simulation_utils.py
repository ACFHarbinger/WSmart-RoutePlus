"""Parallel task arguments must respect the per-policy sample lists (resume)."""

import pytest
from logic.src.utils.tasks.simulation_utils import prepare_parallel_task_args

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_single_sample_resume_skips_finished_policies():
    """With one sample, a policy whose sample 0 is already logged must not be scheduled again."""
    args = prepare_parallel_task_args(["a", "b", "c"], 1, [None], [[], [0], []])
    assert args == [(None, 0, 1)]


def test_single_sample_fresh_run_schedules_every_policy():
    args = prepare_parallel_task_args(["a", "b"], 1, ["idx"], [[0], [0]])
    assert args == [("idx", 0, 0), ("idx", 0, 1)]


def test_multi_sample_schedules_only_listed_samples():
    args = prepare_parallel_task_args(["a", "b"], 3, ["i0", "i1", "i2"], [[1, 2], []])
    assert args == [("i1", 1, 0), ("i2", 2, 0)]
