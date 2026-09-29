"""Hydra dispatch module.

Unified Hydra entry point: receives the composed ``Config`` and delegates
to the appropriate controller function in :mod:`logic.controllers.jobs.pipeline_runner`.
This module owns only the Hydra decorator, profiling lifecycle, and the single-level
dispatch table.

Supported tasks
---------------
- ``train`` / ``meta_train`` / ``hpo`` → :func:`~logic.controllers.jobs.pipeline_runner.run_training`
- ``eval``                              → :func:`~logic.controllers.jobs.pipeline_runner.run_evaluation`
- ``test_sim`` / ``hpo_sim``            → :func:`~logic.controllers.jobs.pipeline_runner.run_simulation`
- ``gen_data``                          → :func:`~logic.controllers.jobs.pipeline_runner.run_data_generation`

Example::

    python main.py train model=am env.name=vrpp env.num_loc=50
    python main.py eval eval.model_path=./weights/best.pt
    python main.py test_sim sim.days=31
    python main.py gen_data data.problem=vrpp
"""

import hydra
from hydra.core.config_store import ConfigStore

from logic.src.configs import Config
from logic.src.constants import CONFIGS_DIR

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

_TRAINING_TASKS = frozenset({"train", "meta_train", "hpo"})
_SIM_TASKS = frozenset({"test_sim", "hpo_sim", "sim_hpo"})
_TASK_ALIASES = {"evaluation": "eval", "sim_hpo": "hpo_sim"}


def _run_task(cfg: Config) -> float:
    """Dispatch ``cfg.task`` to the responsible controller function.

    Args:
        cfg: Fully composed Hydra configuration object.

    Returns:
        Scalar result (loss, reward, or 0.0 depending on the task).

    Raises:
        ValueError: If ``cfg.task`` is not a recognised task name.
    """
    task = _TASK_ALIASES.get(cfg.task, cfg.task)

    if task in _TRAINING_TASKS:
        from logic.controllers.jobs.pipeline_runner import run_training

        return run_training(cfg)

    if task == "eval":
        from logic.controllers.jobs.pipeline_runner import run_evaluation

        return run_evaluation(cfg)

    if task in _SIM_TASKS:
        from logic.controllers.jobs.pipeline_runner import run_simulation

        return run_simulation(cfg)

    if task == "gen_data":
        from logic.controllers.jobs.pipeline_runner import run_data_generation

        return run_data_generation(cfg)

    raise ValueError(f"Unknown task: {task!r}")


@hydra.main(version_base=None, config_path=CONFIGS_DIR, config_name="config")
def hydra_entry_point(cfg: Config) -> float:
    """Unified Hydra entry point for all configuration-driven commands.

    Wraps :func:`_run_task` with optional profiling support.

    Args:
        cfg: The Hydra ``Config`` object (structured configuration).

    Returns:
        Scalar result of the executed task.
    """
    if cfg.tracking.profile:
        from logic.src.tracking.profiling import start_global_profiling, stop_global_profiling

        start_global_profiling(log_dir=cfg.tracking.log_dir)

    try:
        return _run_task(cfg)
    finally:
        if cfg.tracking.profile:
            # pyrefly: ignore [unbound-name]
            stop_global_profiling()
