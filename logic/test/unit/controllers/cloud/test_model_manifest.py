"""Job model, state machine and manifest parsing of the cloud runner."""

import json

import pytest
from logic.controllers.cloud.manifest import cell_from_command, job_from_command, load_manifest, parse_command
from logic.controllers.cloud.model import (
    CellSpec,
    GurobiMode,
    JobRecord,
    JobSpec,
    JobState,
    TaskRecord,
    TaskState,
    TransitionError,
    check_transition,
    job_from_dict,
    job_state,
    task_from_dict,
    to_dict,
)


@pytest.mark.unit
def test_transitions_follow_the_state_machine():
    check_transition(TaskState.QUEUED, TaskState.PLACING)
    check_transition(TaskState.RUNNING, TaskState.PREEMPTED)
    check_transition(TaskState.FAILED, TaskState.RETRYING)
    for old, new in [
        (TaskState.QUEUED, TaskState.RUNNING),
        (TaskState.SUCCEEDED, TaskState.RETRYING),
        (TaskState.STARTING, TaskState.QUEUED),
    ]:
        with pytest.raises(TransitionError):
            check_transition(old, new)


@pytest.mark.unit
@pytest.mark.parametrize(
    "states, cancel, expected",
    [
        ([TaskState.QUEUED, TaskState.QUEUED], False, JobState.QUEUED),
        ([TaskState.SUCCEEDED, TaskState.RUNNING], False, JobState.RUNNING),
        ([TaskState.SUCCEEDED, TaskState.SUCCEEDED], False, JobState.SUCCEEDED),
        ([TaskState.SUCCEEDED, TaskState.FAILED], False, JobState.FAILED),
        ([TaskState.SUCCEEDED, TaskState.CANCELLED], True, JobState.CANCELLED),
        ([TaskState.RETRYING, TaskState.QUEUED], False, JobState.RUNNING),
    ],
)
def test_job_state_is_derived_from_tasks(states, cancel, expected):
    assert job_state(states, cancel) == expected


@pytest.mark.unit
def test_spec_validation():
    cell = CellSpec(cell_id="a", argv=["test_sim"])
    JobSpec(name="ok", cells=[cell]).validate()
    with pytest.raises(ValueError, match="at least one cell"):
        JobSpec(name="x", cells=[]).validate()
    with pytest.raises(ValueError, match="unique"):
        JobSpec(name="x", cells=[cell, cell]).validate()
    with pytest.raises(ValueError, match="test_sim"):
        JobSpec(name="x", cells=[CellSpec(cell_id="b", argv=["train"])]).validate()
    with pytest.raises(ValueError, match="Gurobi"):
        JobSpec(name="x", cells=[CellSpec(cell_id="g", argv=["test_sim"], needs_gurobi=True)]).validate()
    with pytest.raises(ValueError, match="gurobi_secret"):
        JobSpec(name="x", cells=[cell], gurobi=GurobiMode.WLS_SECRET).validate()
    JobSpec(name="x", cells=[cell], gurobi=GurobiMode.BYO, gurobi_secret="my-license").validate()


@pytest.mark.unit
def test_records_round_trip_through_json():
    spec = JobSpec(name="j", cells=[CellSpec(cell_id="a", argv=["test_sim", "x=1"], env={"K": "v"})])
    job = JobRecord(job_id="job-1", spec=spec)
    assert job_from_dict(json.loads(json.dumps(to_dict(job)))) == job
    task = TaskRecord(
        task_id="task-1", job_id="job-1", cell=spec.cells[0], state=TaskState.RUNNING, progress={"day": 2}
    )
    assert task_from_dict(json.loads(json.dumps(to_dict(task)))) == task


RERUN_COMMAND = (
    "WSR_POLICY_CONFIG_DIR=$WSR_POLICY_YAML_ROOT/aco_hh__lookahead__cls python main.py test_sim "
    "sim.graph.area=figueiradafoz sim.graph.num_loc=350 sim.cpu_cores=2 sim.resume=false "
    "'sim.graph.dm_filepath=\"osm_distmat_plastic[figueiradafoz].csv\"' sim.run_name=la_cls 'sim.policies=[aco_hh]'"
)


@pytest.mark.unit
def test_rerun_commands_drop_runner_owned_settings(monkeypatch):
    monkeypatch.setenv("WSR_POLICY_YAML_ROOT", "/yaml")
    argv, policy_dir, env = parse_command(RERUN_COMMAND)
    assert argv[0] == "test_sim"
    assert "sim.cpu_cores=2" not in argv and "sim.resume=false" not in argv
    assert 'sim.graph.dm_filepath="osm_distmat_plastic[figueiradafoz].csv"' in argv
    assert policy_dir == "$WSR_POLICY_YAML_ROOT/aco_hh__lookahead__cls" and env == {}
    cell = cell_from_command(RERUN_COMMAND, cell_id="c")
    assert cell.policy_dir == "/yaml/aco_hh__lookahead__cls"
    with pytest.raises(ValueError, match="not a test_sim"):
        parse_command("python main.py train")


@pytest.mark.unit
def test_single_command_becomes_a_one_cell_job():
    spec = job_from_command("test_sim sim.run_name=lm_cls 'sim.policies=[bpc]'", max_parallel=2)
    assert len(spec.cells) == 1 and spec.cells[0].cell_id == "lm_cls/bpc" and spec.max_parallel == 2


@pytest.mark.unit
def test_manifests_native_and_rerun(tmp_path):
    rerun = {
        "cells": [
            {"cell_id": "a/aco", "policy": "aco_hh", "command": RERUN_COMMAND, "expected_wall_seconds": 60.0},
            {"cell_id": "a/bpc", "policy": "bpc", "command": RERUN_COMMAND, "needs_gurobi": True},
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(rerun))
    spec = load_manifest(path, policy="aco_hh")
    assert [c.cell_id for c in spec.cells] == ["a/aco"] and spec.cells[0].expected_wall_seconds == 60.0
    with pytest.raises(ValueError, match="Gurobi"):
        load_manifest(path)
    assert len(load_manifest(path, gurobi=GurobiMode.WLS_SECRET, gurobi_secret="wls").cells) == 2

    native = tmp_path / "job.yaml"
    native.write_text(
        "name: native\nmax_parallel: 3\nresources: {memory_gb: 6, spot: true}\n"
        "cells:\n  - {cell_id: one, argv: [test_sim, sim.seed=1]}\n  - {cell_id: two, command: 'test_sim sim.seed=2'}\n"
    )
    spec = load_manifest(native)
    assert spec.name == "native" and spec.max_parallel == 3 and spec.resources.spot
    assert [c.argv for c in spec.cells] == [["test_sim", "sim.seed=1"], ["test_sim", "sim.seed=2"]]
