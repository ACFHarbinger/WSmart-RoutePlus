"""Smoke test of the cloud image against a local control plane and store (CR-03, #123).

Stages the inputs of one one-day Rio Maior 100 cell into a filesystem object
store, records the task in a SQLite control plane, then runs the image the way
a cloud backend would, with only the state directory mounted:

    python infra/cloud/image/smoke.py wsmart-route:<commit> --state /tmp/wsr-smoke

It passes when the container exits 0 and the task's results index lists the
simulation logs. No cloud account is involved.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from logic.controllers.cloud.local import FilesystemObjectStore, SQLiteControlPlane  # noqa: E402
from logic.controllers.cloud.model import CellSpec, ComputeKind, JobRecord, JobSpec, Provider, TaskRecord  # noqa: E402
from logic.controllers.cloud.staging import stage_job_inputs  # noqa: E402

CELL = [
    "test_sim",
    "sim.graph.area=riomaior",
    "sim.graph.num_loc=100",
    "sim.graph.waste_type=plastic",
    "sim.data_distribution=emp",
    "sim.graph.n_days=1",
    "sim.graph.start_day=0",
    "sim.graph.n_samples=1",
    "sim.n_samples=1",
    "sim.n_vehicles=0",
    "sim.seed=42",
    "sim.no_cuda=true",
    "sim.graph.load_dataset=data/simulator/datasets/riomaior100_emp_wsr30_N1_seed42.npz",
    "sim.graph.distance_method=gmaps",
    "sim.run_name=smoke",
    "sim.policies=[aco_hh]",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image")
    parser.add_argument("--state", type=Path, required=True, help="Host directory for the control plane and store.")
    parser.add_argument("--data-root", type=Path, default=REPO, help="Repository whose data/simulator is staged.")
    args = parser.parse_args()

    state = args.state.resolve()
    state.mkdir(parents=True, exist_ok=True)
    cp = SQLiteControlPlane(state / "cp.db")
    store = FilesystemObjectStore(state / "store")
    spec = JobSpec(
        name="image-smoke",
        cells=[CellSpec(cell_id="smoke", argv=CELL)],
        provider=Provider.GCP,
        compute=ComputeKind.BATCH,
        image=args.image,
    )
    job_id, task_id = "job-smoke", "task-smoke"
    spec.inputs_manifest = stage_job_inputs(store, spec, args.data_root, job_id)
    cp.create_job(JobRecord(job_id=job_id, spec=spec), [TaskRecord(task_id=task_id, job_id=job_id, cell=spec.cells[0])])
    os.chmod(state, 0o777)  # the container runs as <uid>:0 and writes the database and store
    for path in state.rglob("*"):
        path.chmod(0o777 if path.is_dir() else 0o666)

    command = [
        "docker", "run", "--rm", "--user", f"{os.getuid()}:0", "--memory=3g",
        "-v", f"{state}:/state",
        "-e", "WSR_CLOUD_CONTROL_PLANE=sqlite:////state/cp.db",
        "-e", "WSR_CLOUD_STORE=file:///state/store",
        "-e", f"WSR_CLOUD_JOB_ID={job_id}", "-e", f"WSR_CLOUD_TASK_ID={task_id}",
        "-e", "WSR_CLOUD_SYNC_SECONDS=5",
        args.image,
    ]  # fmt: skip
    code = subprocess.run(command).returncode
    results = [f.relpath for f in cp.list_results(task_id)]
    logs = [r for r in results if r.endswith("_1N.json") and "/log_" in r]
    print(
        json.dumps({"exit_code": code, "results": len(results), "logs": logs, "message": cp.get_task(task_id).message})
    )
    return 0 if code == 0 and logs else 1


if __name__ == "__main__":
    sys.exit(main())
