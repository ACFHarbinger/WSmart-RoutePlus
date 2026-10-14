"""
BatchManager — top-level orchestrator for multi-run experiment batches.

Usage
-----
The ``BatchManager`` is the primary public interface of the ``manager`` package.
It loads a batch YAML configuration, expands any ``expand`` blocks into concrete
jobs, runs setup steps, dispatches jobs through ``BatchExecutor``, and finally
executes teardown steps before printing a summary report.

.. code-block:: python

    from logic.controllers.manager import BatchManager

    mgr = BatchManager.from_yaml("logic/configs/my_experiment.yaml")
    mgr.run()

Or from the CLI via the dedicated justfile recipe::

    just batch-run batch_cfg=logic/configs/my_experiment.yaml

Batch YAML structure
--------------------
See ``logic/configs/batch.yaml`` for a comprehensive annotated example.

Key top-level keys:

``name`` (str)
    Human-readable name for the batch (used in reports).

``fail_fast`` (bool, default False)
    Abort the batch on the first job failure.

``dry_run`` (bool, default False)
    Print commands without executing them.

``setup`` (list of steps)
    Steps executed once before ALL jobs begin.

``teardown`` (list of steps)
    Steps executed once after ALL jobs finish.  Each step's ``condition`` field
    controls whether it runs (e.g. ``all_succeeded``).

``runs`` (list)
    Ordered list of run entries.  Each entry is either:

    * A **concrete run** with explicit ``overrides`` (one job per entry), or
    * An **expand block** with an ``expand`` key (one job per Cartesian
      combination).

    Both entry types support ``pre_steps`` and ``post_steps``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

from .batch_executor import BatchExecutor
from .batch_expander import BatchExpander, _parse_steps
from .batch_job import BatchJob, JobResult
from .batch_step import BatchStep
from .batch_step_executor import run_step

__all__ = ["BatchManager"]

_BRIGHT = "\033[1m"
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_BLUE = "\033[94m"
_RESET = "\033[0m"


_MS_VARIANT_COUNT: Dict[str, int] = {
    "lookahead": 1,
    "last_minute": 2,
    "service_level": 2,
}
_RI_VARIANT_COUNT: Dict[str, int] = {
    "cls": 1,
    "ftsp": 1,
}


def _parse_override_context(overrides: List[str]) -> Dict[str, str]:
    """Flatten Hydra override strings into a template-variable dict.

    ``"sim.graph.area=riomaior"`` → ``{"sim_graph_area": "riomaior"}``.
    List values like ``[bpc,hgs]`` have their brackets stripped.
    """
    ctx: Dict[str, str] = {}
    for ov in overrides:
        if "=" not in ov:
            continue
        key, val = ov.split("=", 1)
        ctx[key.replace(".", "_")] = val.strip("[]")
    return ctx


def _estimate_cores(task: str, overrides: List[str], pre_steps: Any) -> int:
    """Estimate the number of parallel worker slots a job will occupy.

    test_sim / hpo_sim: ``n_policies × n_ms_variants × n_ri_variants``.
    All other tasks: 1.
    """
    task_lower = task.lower()
    if not any(kw in task_lower for kw in ("test_sim", "sim_hpo", "hpo_sim")):
        return 1

    n_policies = 1
    for ov in overrides:
        if "sim.policies=" in ov:
            val = ov.split("sim.policies=", 1)[1].strip("[]")
            n_policies = max(1, len([p.strip() for p in val.split(",") if p.strip()]))
            break

    n_ms = 1
    n_ri = 1
    for step in pre_steps:
        if getattr(step, "type", None) == "patch_policy_yaml":
            n_ms = _MS_VARIANT_COUNT.get(step.args.get("packing", ""), 1)
            n_ri = _RI_VARIANT_COUNT.get(step.args.get("route_improvement", ""), 1)
            break

    return n_policies * n_ms * n_ri


def _box(text: str, width: int = 72) -> None:
    print(f"\n{_BLUE}╔{'═' * (width - 2)}╗{_RESET}")
    print(f"{_BLUE}║{_BRIGHT} {text:<{width - 3}}{_RESET}{_BLUE}║{_RESET}")
    print(f"{_BLUE}╚{'═' * (width - 2)}╝{_RESET}\n")


def _section(text: str) -> None:
    print(f"\n{_CYAN}══ {text} {'═' * max(0, 60 - len(text))}{_RESET}")


class BatchManager:
    """Orchestrate a multi-run batch of WSmart-Route experiments.

    Parameters
    ----------
    config:
        Parsed batch configuration dictionary.  Typically loaded via
        ``BatchManager.from_yaml()``.
    project_root:
        Override for the repository root path (defaults to auto-detection).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        project_root: Optional[Path] = None,
    ) -> None:
        self._cfg = config
        if project_root is None:
            project_root = Path(__file__).resolve().parents[3]
        self._root = project_root

    # -----------------------------------------------------------------------
    # Routing Algorithms
    # -----------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path, project_root: Optional[Path] = None) -> "BatchManager":
        """Load a ``BatchManager`` from a YAML file.

        Args:
            path: Path to the batch YAML file (absolute or relative to CWD).
            project_root: Optional override for the repository root.

        Returns:
            Configured ``BatchManager`` instance.
        """
        cfg_path = Path(path)
        if not cfg_path.is_absolute():
            cfg_path = Path.cwd() / cfg_path
        raw = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        return cls(raw, project_root=project_root)  # type: ignore[arg-type]

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(self, resume: bool = False) -> int:
        """Execute the entire batch and return an exit code.

        Args:
            resume: When True, check git history for each job's commit message
                and skip any job that already has a matching commit.  Setup
                steps are also suppressed so existing output is not deleted.

        Returns:
            0 if all jobs (and setup/teardown) succeeded, 1 otherwise.
        """
        cfg = self._cfg
        name: str = cfg.get("name", "batch")
        fail_fast: bool = bool(cfg.get("fail_fast", False))
        dry_run: bool = bool(cfg.get("dry_run", False))

        _box(f"WSmart-Route Batch: {name}")

        # --- Parse jobs -------------------------------------------------------
        all_jobs = self._parse_jobs(cfg.get("runs") or [])
        jobs = self._filter_incomplete_jobs(all_jobs) if resume else all_jobs
        _section(f"Jobs: {len(jobs)} to run" + (f" ({len(all_jobs) - len(jobs)} already completed)" if resume else " total"))
        for j in jobs:
            print(f"  [{j.index + 1}] {j.task} — {j.name}")
        print()

        if not jobs:
            print(f"{_GREEN}All jobs already completed — nothing to do.{_RESET}\n")
            return 0

        start = time.monotonic()
        overall_ok = True

        # --- Setup steps (suppressed in resume mode) --------------------------
        if resume:
            _section("Setup")
            print(f"  {_YELLOW}[resume] Setup steps suppressed — preserving existing output.{_RESET}")
        else:
            setup_steps = _parse_steps(cfg.get("setup") or [])
            if setup_steps:
                _section("Setup")
                for step in setup_steps:
                    ok = self._run_global_step(step, [], None, dry_run)
                    if not ok and fail_fast:
                        print(f"{_RED}Setup step failed — aborting batch.{_RESET}")
                        return 1

        # --- Execute jobs -----------------------------------------------------
        max_cores: int = int(cfg.get("max_cores", 0))
        executor = BatchExecutor(project_root=self._root, fail_fast=fail_fast, dry_run=dry_run)
        results: List[JobResult] = executor.run_jobs(jobs, max_cores=max_cores)
        succeeded_flags = [r.succeeded for r in results]
        overall_ok = all(succeeded_flags)

        # --- Teardown steps ---------------------------------------------------
        teardown_steps = _parse_steps(cfg.get("teardown") or [])
        if teardown_steps:
            _section("Teardown")
            last_ok: Optional[bool] = succeeded_flags[-1] if succeeded_flags else None
            for step in teardown_steps:
                if step.should_run(succeeded=succeeded_flags, last_succeeded=last_ok):
                    self._run_global_step(step, succeeded_flags, last_ok, dry_run)
                else:
                    print(
                        f"{_YELLOW}  Skipping teardown step [{step.display_name()}] "
                        f"(condition '{step.condition}' not met){_RESET}"
                    )

        # --- Summary ----------------------------------------------------------
        elapsed = time.monotonic() - start
        self._print_summary(name, results, elapsed)

        return 0 if overall_ok else 1

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _committed_subjects(self) -> set:
        """Return the set of git commit subject lines (first line) in HEAD history."""
        import subprocess as _sp

        result = _sp.run(
            ["git", "log", "--pretty=format:%s"],
            cwd=self._root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return set()
        return {line.strip() for line in result.stdout.splitlines() if line.strip()}

    def _job_commit_message(self, job: BatchJob) -> Optional[str]:
        """Return the rendered git_commit message for a job, or None if absent."""
        for step in job.post_steps:
            if step.type == "git_commit":
                msg_template: str = step.args.get("message", "")
                if msg_template:
                    return job.render_template(msg_template)
        return None

    def _filter_incomplete_jobs(self, jobs: List[BatchJob]) -> List[BatchJob]:
        """Return only jobs that do not yet have a matching git commit.

        A job is considered complete when its ``git_commit`` post-step message
        appears verbatim as a commit subject in the repository's history.  Jobs
        with no ``git_commit`` step are always included (cannot be detected).
        """
        _section("Resume — scanning git history")
        done = self._committed_subjects()
        remaining: List[BatchJob] = []
        for job in jobs:
            msg = self._job_commit_message(job)
            if msg and msg in done:
                print(f"  {_GREEN}✔ already done:{_RESET}  [{job.index + 1}] {job.name}  ({msg})")
            else:
                reason = f"commit not found: '{msg}'" if msg else "no git_commit step"
                print(f"  {_YELLOW}↻ pending:{_RESET}       [{job.index + 1}] {job.name}  ({reason})")
                remaining.append(job)
        print()
        return remaining

    def _parse_jobs(self, runs: List[Any]) -> List[BatchJob]:
        """Convert the ``runs`` list into ``BatchJob`` instances.

        Entries with an ``expand`` key are expanded via ``BatchExpander``; all
        others are treated as concrete single-run entries.

        Args:
            runs: Raw list from the YAML ``runs`` key.

        Returns:
            Ordered list of ``BatchJob`` objects with sequential indices.
        """
        jobs: List[BatchJob] = []
        index = 0
        for entry in runs:
            if not isinstance(entry, dict):
                continue
            if "expand" in entry:
                expander = BatchExpander(entry, start_index=index)
                expanded = expander.expand()
                jobs.extend(expanded)
                index += len(expanded)
            else:
                # Concrete single-run entry
                pre = _parse_steps(entry.get("pre_steps") or [])
                post = _parse_steps(entry.get("post_steps") or [])
                raw_overrides = list(entry.get("overrides") or [])
                job = BatchJob(
                    task=entry.get("task", "test_sim"),
                    name=entry.get("name", f"run_{index}"),
                    overrides=raw_overrides,
                    pre_steps=pre,
                    post_steps=post,
                    index=index,
                    metadata=dict(entry.get("metadata") or {}),
                    override_context=_parse_override_context(raw_overrides),
                    cores_estimate=_estimate_cores(entry.get("task", "test_sim"), raw_overrides, pre),
                )
                jobs.append(job)
                index += 1
        return jobs

    def _run_global_step(
        self,
        step: BatchStep,
        succeeded: List[bool],
        last: Optional[bool],
        dry_run: bool,
    ) -> bool:
        """Execute a setup or teardown step, catching errors.

        Args:
            step: Step to run.
            succeeded: Succeeded flags collected so far.
            last: Last job's success flag.
            dry_run: If True, only print without executing.

        Returns:
            ``True`` if the step ran without errors.
        """
        print(f"  → [{step.display_name()}]")
        if dry_run:
            print(f"    [dry_run] Would run step: {step.type} args={step.args}")
            return True
        try:
            # Global steps have no owning job; create a dummy for template rendering
            dummy = BatchJob(name="global", task="", index=0)
            run_step(step, dummy)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"{_RED}  Step [{step.display_name()}] failed: {exc}{_RESET}")
            return False

    @staticmethod
    def _print_summary(name: str, results: List[JobResult], elapsed: float) -> None:
        """Print a summary table of all job outcomes.

        Args:
            name: Batch name.
            results: List of job results.
            elapsed: Total wall-clock seconds.
        """
        n_ok = sum(1 for r in results if r.succeeded)
        n_fail = len(results) - n_ok

        _section(f"Batch '{name}' — Summary")
        print(f"  Total jobs  : {len(results)}")
        print(f"  Succeeded   : {_GREEN}{n_ok}{_RESET}")
        print(f"  Failed      : {_RED}{n_fail}{_RESET}")
        print(f"  Wall time   : {elapsed:.1f}s\n")

        for r in results:
            status = f"{_GREEN}✔{_RESET}" if r.succeeded else f"{_RED}✘{_RESET}"
            print(f"  {status}  [{r.job.index + 1}] {r.job.task} — {r.job.name}")
            if r.error:
                print(f"       Error: {r.error}")
            for sname, exc in r.pre_step_errors.items():
                print(f"       Pre-step '{sname}' error: {exc}")
            for sname, exc in r.post_step_errors.items():
                print(f"       Post-step '{sname}' error: {exc}")
        print()
