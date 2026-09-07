"""Pipeline runner for Hydra-driven tasks.

Unified entry point for all configuration-driven pipeline execution tasks:

- Neural model training and evaluation: ``train``, ``meta_train``, ``hpo``, ``eval``
- Simulation engine execution: ``test_sim``, ``hpo_sim`` / ``sim_hpo``
- Dataset generation: ``gen_data``

Hydra drives configuration for all tasks in this module; the underlying business
logic is implemented in the corresponding feature pipelines.

Example::

    python main.py train model=am env.name=vrpp env.num_loc=50
    python main.py eval eval.model_path=./weights/best.pt
    python main.py test_sim sim.days=31 sim.policies=regular,gurobi,alns
    python main.py gen_data data.problem=vrpp data.graph_sizes=[50]
"""

from typing import Any, List

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROOT_KEYS: List[str] = [
    "seed",
    "device",
    "experiment_name",
    "task",
    "output_dir",
    "run_name",
    "start",
    "tracking",
]

_TRAINING_TASKS = frozenset({"train", "meta_train", "hpo"})
_SIM_TASKS = frozenset({"test_sim", "hpo_sim", "sim_hpo"})

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _print_config(cfg: Any, label: str, filter_keys: Any = None) -> None:
    """Pretty-print a filtered view of a Hydra config.

    Args:
        cfg: The Hydra configuration object.
        label: Section header label (e.g. ``"TRAINING"``).
        filter_keys: Optional list of top-level keys to include.
    """
    from omegaconf import OmegaConf

    print("\n" + "=" * 80)
    print(f"HYDRA CONFIGURATION — {label}".center(80))
    print("=" * 80)
    display_cfg = OmegaConf.masked_copy(cfg, filter_keys) if filter_keys else cfg
    print(OmegaConf.to_yaml(display_cfg, resolve=False))
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Model Training & Evaluation
# ---------------------------------------------------------------------------


def run_training(cfg: Any) -> float:
    """Dispatch a training, meta-RL, or HPO run.

    Delegates to :func:`~logic.src.pipeline.features.train.run_hpo` when
    ``cfg.hpo.n_trials > 0``, otherwise to
    :func:`~logic.src.pipeline.features.train.run_training`.

    Args:
        cfg: Hydra ``Config`` object (structured config).

    Returns:
        Scalar result value (loss, reward, or HPO best metric).
    """
    from logic.src.pipeline.features.train import run_hpo
    from logic.src.pipeline.features.train import run_training as _train

    if cfg.tracking.verbose:
        _print_config(
            cfg,
            "TRAINING",
            filter_keys=_ROOT_KEYS + ["env", "model", "train", "rl", "optim"],  # type: ignore[arg-type]
        )

    if cfg.hpo.n_trials > 0:
        return run_hpo(cfg)
    return _train(cfg)


def run_evaluation(cfg: Any) -> float:
    """Run model evaluation.

    Args:
        cfg: Hydra ``Config`` object (structured config).

    Returns:
        0.0 on success.
    """
    from logic.src.pipeline.features.eval import run_evaluate_model

    if cfg.tracking.verbose:
        _print_config(cfg, "EVALUATION", filter_keys=_ROOT_KEYS + ["eval"])  # type: ignore[arg-type]

    run_evaluate_model(cfg)
    return 0.0


# ---------------------------------------------------------------------------
# Simulation & Data Generation
# ---------------------------------------------------------------------------


def run_simulation(cfg: Any) -> float:
    """Run the WSmart-Route simulator or simulation HPO.

    Dispatches on ``cfg.task``:

    - ``"test_sim"``              → standard multi-day simulator test.
    - ``"hpo_sim"`` / ``"sim_hpo"`` → policy hyperparameter optimisation.

    Args:
        cfg: Hydra ``Config`` object (structured config).

    Returns:
        0.0 on success.

    Raises:
        ValueError: If ``cfg.task`` is not a recognised simulation task.
    """
    task = cfg.task

    if task == "test_sim":
        from logic.src.pipeline.features.test import run_simulator_test

        if cfg.tracking.verbose:
            _print_config(cfg, "SIMULATION", filter_keys=_ROOT_KEYS + ["sim"])  # type: ignore[arg-type]
        run_simulator_test(cfg)
        return 0.0

    if task in ("hpo_sim", "sim_hpo"):
        from logic.src.pipeline.simulations.hpo.hpo_handler import run_hpo_sim

        if cfg.tracking.verbose:
            _print_config(cfg, "SIMULATION HPO", filter_keys=_ROOT_KEYS + ["hpo_sim"])  # type: ignore[arg-type]
        run_hpo_sim(cfg)
        return 0.0

    raise ValueError(f"Unknown simulation task: {task!r}")


def run_data_generation(cfg: Any) -> float:
    """Generate datasets for training, validation, or testing.

    Initialises the WSmart+ tracking run, calls the dataset generator, and
    records whether that operation completed or failed.

    Args:
        cfg: Hydra ``Config`` object (structured config).

    Returns:
        0.0 on success.
    """
    import logic.src.tracking as wst
    from logic.src.data.generators import generate_datasets

    if cfg.tracking.verbose:
        _print_config(cfg, "DATA GENERATION", filter_keys=_ROOT_KEYS + ["data"])  # type: ignore[arg-type]

    experiment_name = cfg.experiment_name or f"gen_data_{cfg.data.problem}"
    wst.init(experiment_name=experiment_name)
    try:
        generate_datasets(cfg)
    except Exception:
        run = wst.get_active_run()
        if run is not None:
            run.set_tag("status", "failed")
            run.flush()
        raise
    else:
        run = wst.get_active_run()
        if run is not None:
            run.set_tag("status", "completed")
            run.flush()
    return 0.0
