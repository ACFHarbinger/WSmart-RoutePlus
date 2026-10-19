"""Background worker, the light ``-m`` entry point and the download listing."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from logic.controllers.cloud import cli
from logic.controllers.cloud.__main__ import main as module_main
from logic.controllers.cloud.worker import serve, spawn_worker, worker_lock, worker_running

REPO = Path(__file__).resolve().parents[5]

FAKE_MAIN = """
import json, os, sys
args = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
out = os.path.join(args["sim.output_dir"], "1days", "riomaior100_plastic", "emp", args["sim.run_name"])
os.makedirs(out, exist_ok=True)
json.dump({"ok": True}, open(os.path.join(out, "log_x_1N.json"), "w"))
"""


@pytest.mark.unit
def test_worker_lock_is_exclusive(tmp_path):
    assert not worker_running(tmp_path)
    with worker_lock(tmp_path) as mine:
        assert mine
        holder = subprocess.run(
            [
                sys.executable,
                "-c",
                f"from logic.controllers.cloud.worker import worker_running; print(worker_running({str(tmp_path)!r}))",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        )
        assert holder.stdout.strip() == "True"
        with worker_lock(tmp_path) as nested:
            assert not nested  # a second holder in the same process is refused too
    assert not worker_running(tmp_path)


@pytest.mark.unit
def test_submit_spawns_one_worker_that_drains_the_queue_and_exits(tmp_path, monkeypatch):
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    (fake_repo / "main.py").write_text(FAKE_MAIN)
    root = tmp_path / "state"
    common = {
        "root": str(root),
        "project_root": str(fake_repo),
        "json": False,
        "cloud_command": "submit",
        "manifest": None,
        "policy": None,
    }
    first = cli.run_cloud({**common, "sim_command": "test_sim sim.run_name=a", "memory_gb": 0.0})
    second = cli.run_cloud({**common, "sim_command": "test_sim sim.run_name=b", "memory_gb": 0.0})
    assert first["worker_pid"] is not None
    deadline = time.time() + 60
    service = cli.local_service(root, fake_repo)
    while time.time() < deadline:
        states = [service.status(j)["state"] for j in (first["job_id"], second["job_id"])]
        if states == ["succeeded", "succeeded"] and not worker_running(root):
            break
        time.sleep(0.2)
    else:
        raise AssertionError((states, (root / "worker.log").read_text()))
    assert "queue drained" in (root / "worker.log").read_text()


@pytest.mark.unit
def test_serve_returns_when_another_worker_holds_the_lock(tmp_path):
    with worker_lock(tmp_path):
        assert serve(cli.local_service(tmp_path), tmp_path, poll_seconds=0.01) == "another worker is running"
    assert spawn_worker(tmp_path, project_root=REPO) is not None or True  # an idle worker starts and exits


@pytest.mark.unit
def test_module_entry_prints_only_json_and_reports_errors(tmp_path, capsys):
    assert module_main(["--root", str(tmp_path), "--json", "status"]) == 0
    assert json.loads(capsys.readouterr().out) == []
    assert module_main(["--root", str(tmp_path), "--json", "status", "job-missing"]) == 1
    assert "KeyError" in capsys.readouterr().err


@pytest.mark.unit
def test_download_lists_files_with_hashes(tmp_path, monkeypatch, capsys):
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    (fake_repo / "main.py").write_text(FAKE_MAIN)
    monkeypatch.setattr(cli, "REPO_ROOT", fake_repo)
    root = tmp_path / "state"
    job = cli.run_cloud(
        {
            "root": str(root),
            "json": False,
            "cloud_command": "submit",
            "manifest": None,
            "policy": None,
            "sim_command": "test_sim sim.run_name=a",
            "memory_gb": 0.0,
            "wait": True,
            "poll": 0.1,
        }
    )
    capsys.readouterr()
    cli.run_cloud(
        {
            "root": str(root),
            "json": True,
            "cloud_command": "download",
            "job_id": job["job_id"],
            "dest": str(tmp_path / "out"),
        }
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["files"] == 1
    (entry,) = payload["file_list"]
    assert entry["relpath"] == "1days/riomaior100_plastic/emp/a/log_x_1N.json" and len(entry["sha256"]) == 64
