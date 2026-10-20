"""Launch specs of :class:`.service.CloudService` (moved from ``service.py``).

Attributes:
    ServiceLaunchMixin: ``_launch`` (local processes) and ``_remote_launch`` (the container task runtime).
"""

from __future__ import annotations

from pathlib import Path

from logic.controllers.cloud.interfaces import LaunchSpec
from logic.controllers.cloud.model import GurobiMode, JobRecord, Provider, TaskRecord
from logic.controllers.cloud.runtime import entry_script


class ServiceLaunchMixin:
    """What a backend is asked to start; needs the service's ``python``, ``remote``, ``_backend`` and ``task_dir``."""

    def _launch(self, job: JobRecord, task: TaskRecord) -> LaunchSpec:
        if Provider(job.spec.provider) != Provider.LOCAL:
            return self._remote_launch(job, task)
        spec = job.spec
        workdir = self.task_dir(task)
        outputs = workdir / "outputs"
        argv = [
            self.python,
            entry_script(getattr(self._backend(spec), "cwd", Path("."))),
            *task.cell.argv,
            "sim.cpu_cores=1",
            "sim.resume=true",
            "sim.checkpoint_days=1",
            f"sim.output_dir={outputs}",
        ]
        env = {var: "1" for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
        env.update(task.cell.env)
        env["WSR_CLOUD_JOB_ID"], env["WSR_CLOUD_TASK_ID"] = job.job_id, task.task_id
        env["WSR_CLOUD_ATTEMPT"] = str(task.attempt)
        if task.cell.policy_dir:
            env["WSR_POLICY_CONFIG_DIR"] = task.cell.policy_dir
        secret_env = {}
        if GurobiMode(spec.gurobi) != GurobiMode.NONE and spec.gurobi_secret:
            # The task runtime (CR-02) turns this into a license file before the run.
            secret_env["WSR_GUROBI_LICENSE"] = spec.gurobi_secret
        return LaunchSpec(
            task=task,
            resources=spec.resources,
            argv=argv,
            env=env,
            secret_env=secret_env,
            workdir=str(workdir),
            image=spec.image,
        )

    def _remote_launch(self, job: JobRecord, task: TaskRecord) -> LaunchSpec:
        """A container attempt: the image runs the task runtime, which reads the rest from the control plane."""
        assert self.remote is not None
        env = {
            "WSR_CLOUD_CONTROL_PLANE": self.remote.control_plane_uri,
            "WSR_CLOUD_STORE": self.remote.store_uri,
            "WSR_CLOUD_JOB_ID": job.job_id,
            "WSR_CLOUD_TASK_ID": task.task_id,
            "WSR_CLOUD_ATTEMPT": str(task.attempt),
        }
        secret_env = {}
        if GurobiMode(job.spec.gurobi) != GurobiMode.NONE and job.spec.gurobi_secret:
            secret_env["WSR_GUROBI_LICENSE"] = job.spec.gurobi_secret
        return LaunchSpec(
            task=task,
            resources=job.spec.resources,
            argv=[self.remote.python, "-m", "logic.controllers.cloud.runtime"],
            env=env,
            secret_env=secret_env,
            image=job.spec.image,
        )
