"""Training engine for WSmart-Route.

Setup helpers live in ``engine_setup``; this module keeps the training loop and re-exports those names.

Attributes:
    run_training: Run single model training (or curriculum if configured).
    _run_curriculum_stages: Orchestrate sequential curriculum training stages.
    _run_single_stage: Execute one training stage and return (val_reward, state_dict).
    _build_stage_config: Build a stage-specific config with updated graph settings.
    _run_training_via_zenml: Run training through the ZenML pipeline.
    _build_experiment_name: Derive a human-readable experiment name from the Hydra config.
    _build_callbacks: Instantiate training callbacks from Hydra config.
    _log_training_params: Log training configuration parameters.
    _track_val_dataset: Track the validation dataset if available.

Example:
    >>> from logic.src.pipeline.features.train.engine import run_training
    >>> run_training(cfg)
    200000.0
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger

import logic.src.pipeline.features.train.zenml_train_pipeline as zenml_train_pipeline_module
import logic.src.tracking as wst
from logic.src.configs import Config
from logic.src.pipeline.features.train.model_factory.builder import create_model
from logic.src.pipeline.learning.common.trainer import WSTrainer
from logic.src.tracking.logging.pylogger import get_pylogger

try:
    from logic.src.tracking.integrations.zenml_bridge import configure_zenml_stack
except ImportError:
    configure_zenml_stack = None  # type: ignore[assignment]

from .engine_setup import (
    _build_callbacks as _build_callbacks,
)
from .engine_setup import (
    _build_experiment_name as _build_experiment_name,
)
from .engine_setup import (
    _build_stage_config as _build_stage_config,
)
from .engine_setup import (
    _get_primary_graph as _get_primary_graph,
)
from .engine_setup import (
    _log_training_params as _log_training_params,
)

if TYPE_CHECKING:
    from logic.src.pipeline.learning.common.base.module import RL4COLitModule

logger = get_pylogger(__name__)


def _prepare_stage_model(cfg: Any, prev_state_dict: Optional[Dict[str, Any]]) -> Any:
    """Build the stage model and warm-start it from the previous curriculum weights."""
    if torch.cuda.is_available() and cfg.train.precision in ("16-mixed", "bf16-mixed"):
        torch.set_float32_matmul_precision("medium")

    model = cast("RL4COLitModule", create_model(cfg))
    if prev_state_dict is not None:
        try:
            model.load_state_dict(prev_state_dict, strict=False)
            logger.info("Curriculum warm-start: loaded weights from previous stage.")
        except Exception as exc:
            logger.warning(f"Curriculum warm-start failed (continuing from scratch): {exc}")
    return model


def _attach_run_sinks(run: Any, cfg: Any, sinks: Optional[List[Any]], run_tags: Dict[str, str]) -> None:
    """Attach caller sinks, or the optional MLflow bridge when none were passed."""
    if sinks is not None:
        for sink in sinks:
            run.add_sink(sink)
        return
    mlflow_enabled = bool(getattr(getattr(cfg, "tracking", None), "mlflow_enabled", False))
    if mlflow_enabled:
        with contextlib.suppress(Exception):
            tracking = getattr(cfg, "tracking", None)
            wst.MLflowBridge.attach(
                run,
                mlflow_tracking_uri=str(getattr(tracking, "mlflow_tracking_uri", "mlruns")),
                experiment_name=str(getattr(tracking, "mlflow_experiment_name", "wsmart-route")),
                run_name=run.run_id[:8],
                tags=run_tags,
            )


def _carry_bin_fill(cfg: Any, model: Any, resume: Optional[str]) -> None:
    """Start time training from a stored bin fill unless a resume checkpoint already has one."""
    init_fill = getattr(cfg.train, "init_fill_from", None) or None
    if init_fill and not resume:
        saved = torch.load(init_fill, map_location="cpu", weights_only=False).get("time_training_fill")
        if saved is None:
            raise ValueError(f"train.init_fill_from={init_fill} has no time-training fill")
        model._resume_fill = saved
        logger.info("Starting the bin fill from checkpoint: %s", init_fill)


def _publish_stage_weights(cfg: Any, trainer: Any, model: Any, run: Any, save_final: bool) -> None:
    """Reload the best checkpoint and save the stage weights when a path is configured."""
    ckpt_path = getattr(getattr(trainer, "checkpoint_callback", None), "best_model_path", None)
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
            logger.info("Loaded best checkpoint for state handoff: %s", ckpt_path)
        except Exception as exc:
            logger.warning("Could not reload best checkpoint (%s); using last-epoch weights.", exc)

    if save_final and cfg.train.final_model_path and not getattr(trainer, "is_global_zero", True):
        logger.info("Rank %s: rank 0 saves the final model weights.", getattr(trainer, "global_rank", "?"))
    elif save_final and cfg.train.final_model_path:
        logger.info("Saving final model weights to: %s", cfg.train.final_model_path)
        model.save_weights(cfg.train.final_model_path)
        logger.info("Final model weights saved successfully.")
        run.log_artifact(cfg.train.final_model_path, artifact_type="model")
    elif not cfg.train.final_model_path:
        logger.info("No final_model_path provided, skipping weight saving.")


def _run_single_stage(
    cfg: Any,
    sinks: Optional[List[Any]],
    prev_state_dict: Optional[Dict[str, Any]] = None,
    save_final: bool = True,
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """Execute one training stage.

    Args:
        cfg: Root configuration for this stage.
        sinks: Optional list of tracking sinks.
        prev_state_dict: State dict from the previous curriculum stage for warm-starting.
        save_final: Whether to save the final model weights on this stage.

    Returns:
        Tuple of (val_reward, state_dict).
    """
    seed_everything(cfg.seed)

    # Ensure log directory exists
    os.makedirs(cfg.tracking.log_dir or "logs", exist_ok=True)

    model = _prepare_stage_model(cfg, prev_state_dict)

    callbacks = _build_callbacks(cfg)
    trainer = WSTrainer(
        # n_days is the last day; a stage starting at start_day trains n_days - start_day days.
        max_epochs=max(
            int(_get_primary_graph(cfg, "n_days", 1)) - int(_get_primary_graph(cfg, "start_day", 0) or 0), 1
        ),
        project_name="wsmart-route",
        experiment_name=cfg.experiment_name,
        accelerator=cfg.device if cfg.device != "cuda" else "auto",
        devices=cfg.train.devices,
        strategy=cfg.train.strategy,
        gradient_clip_val=(cfg.rl.max_grad_norm if cfg.rl.algorithm != "ppo" else 0.0),
        logger=CSVLogger(cfg.tracking.log_dir or "logs", name=""),
        callbacks=callbacks,  # type: ignore[arg-type]
        precision=cfg.train.precision,
        log_every_n_steps=cfg.tracking.log_step,
        model_weights_path=cfg.train.model_weights_path,
        logs_dir=cfg.tracking.log_dir,
        reload_dataloaders_every_n_epochs=cfg.train.reload_dataloaders_every_n_epochs,
        enable_progress_bar=False,
        tracking_cfg=cfg.tracking,
    )

    experiment_name = cfg.experiment_name or _build_experiment_name(cfg)
    tracker = wst.init(experiment_name=experiment_name)

    task_cfg = getattr(cfg, getattr(cfg, "task", "train"), cfg)
    env_cfg = getattr(task_cfg, "env", getattr(cfg, "env", None))
    policy_cfg = getattr(task_cfg, "policy", task_cfg)
    model_cfg = getattr(policy_cfg, "model", getattr(cfg, "model", None))

    run_tags = {
        "algorithm": str(getattr(cfg.rl, "algorithm", "")),
        "model": str(getattr(model_cfg, "name", getattr(cfg, "model_name", "")) if model_cfg else ""),
        "problem": str(getattr(env_cfg, "name", "") if env_cfg else ""),
        "num_loc": str(getattr(env_cfg, "num_loc", "") if env_cfg else _get_primary_graph(cfg, "num_loc", "")),
        "seed": str(cfg.seed),
    }
    run = tracker.start_run(experiment_name, run_type="training", tags=run_tags)
    run.__enter__()
    _attach_run_sinks(run, cfg, sinks, run_tags)

    try:
        _log_training_params(run, cfg)

        val_ds = getattr(cfg.train, "val_dataset", None)
        if val_ds and os.path.exists(val_ds):
            run.log_dataset_event(
                "load",
                file_path=val_ds,
                metadata={
                    "variable_name": "val_ds",
                    "source_file": "features/train/engine.py",
                    "source_line": 204,
                },
            )
            run.watch_file(val_ds)

        # train.resume continues an interrupted run from a Lightning checkpoint: weights,
        # optimizer, epoch counter and, for time training, the carried bin fill.
        resume = getattr(cfg.train, "resume", None) or None
        if resume:
            logger.info("Resuming training from checkpoint: %s", resume)
        # train.init_fill_from starts time training from the bin fill stored in another run's
        # checkpoint (e.g. RL continuing the days after imitation); a resumed checkpoint's fill wins.
        _carry_bin_fill(cfg, model, resume)
        trainer.fit(model, ckpt_path=resume)

        # Reload best checkpoint so state_dict carries optimal weights into the next
        # curriculum stage (or into the final save), not just the last training epoch.
        _publish_stage_weights(cfg, trainer, model, run, save_final)

        val_reward = trainer.callback_metrics.get("val/reward", torch.tensor(0.0)).item()
        run.log_metric("best/val_reward", val_reward)
        run.__exit__(None, None, None)
        return val_reward, model.state_dict()

    except Exception as exc:
        run.__exit__(type(exc), exc, exc.__traceback__)
        raise


def _run_curriculum_stages(
    cfg: Any,
    sinks: Optional[List[Any]],
    curriculum_graphs: list,
) -> float:
    """Orchestrate sequential curriculum training.

    Iterates over each graph config in curriculum_graphs, building a stage-specific
    config, training from scratch (or warm-starting), and passing weights forward.

    Args:
        cfg: Root Hydra configuration.
        sinks: Optional tracking sinks.
        curriculum_graphs: Ordered list of graph config entries from train.curriculum_graphs.

    Returns:
        Validation reward from the final stage.
    """
    prev_state_dict: Optional[Dict[str, Any]] = None
    final_reward = 0.0
    n_stages = len(curriculum_graphs)

    for i, graph_cfg in enumerate(curriculum_graphs):
        is_last = i == n_stages - 1
        num_loc = getattr(graph_cfg, "num_loc", None) or (
            graph_cfg.get("num_loc") if hasattr(graph_cfg, "get") else "?"
        )
        n_days = getattr(graph_cfg, "n_days", None) or (graph_cfg.get("n_days") if hasattr(graph_cfg, "get") else "?")
        logger.info(f"Curriculum stage {i + 1}/{n_stages}: num_loc={num_loc}, n_days={n_days}")

        stage_cfg = _build_stage_config(cfg, graph_cfg, stage_idx=i)
        final_reward, state_dict = _run_single_stage(
            stage_cfg,
            sinks,
            prev_state_dict=prev_state_dict,
            save_final=is_last,
        )
        prev_state_dict = state_dict

    return final_reward


def run_training(cfg: Config, sinks: Optional[List[Any]] = None) -> float:
    """Run model training, dispatching to curriculum if configured.

    Args:
        cfg: Root Hydra configuration object.
        sinks: Optional list of tracking sinks (e.g. :class:`ZenMLBridge`)
            to attach to the WSTracker run. When ``None`` (the default),
            :class:`MLflowBridge` is auto-attached if
            ``cfg.tracking.mlflow_enabled`` is ``True``.

    Returns:
        Validation reward from the best epoch (or final curriculum stage).
    """
    # ZenML dispatch (opt-in)
    tracking = getattr(cfg, "tracking", None)
    zenml_enabled = bool(getattr(tracking, "zenml_enabled", False))
    if zenml_enabled and sinks is None:
        return _run_training_via_zenml(cfg)

    # Curriculum dispatch
    task_cfg = getattr(cfg, getattr(cfg, "task", "train"), cfg)
    env_cfg = getattr(task_cfg, "env", getattr(cfg, "env", None))
    curriculum_graphs = list(getattr(env_cfg, "curriculum_graphs", None) or [])
    if len(curriculum_graphs) > 1:
        logger.info(f"Curriculum learning enabled: {len(curriculum_graphs)} stage(s).")
        return _run_curriculum_stages(cfg, sinks, curriculum_graphs)

    # Single-stage training: build a stage config from curriculum_graphs[0] when available
    # so that env.graph is always populated (env.graph was removed from EnvConfig schema).
    stage_cfg = _build_stage_config(cfg, curriculum_graphs[0], stage_idx=0) if curriculum_graphs else cfg
    val_reward, _ = _run_single_stage(stage_cfg, sinks, save_final=True)
    return val_reward


# ---------------------------------------------------------------------------
# ZenML dispatch
# ---------------------------------------------------------------------------


def _run_training_via_zenml(cfg: Config) -> float:
    """Dispatch training to the ZenML training pipeline.

    Called when ``cfg.tracking.zenml_enabled`` is ``True`` and no external
    sinks were injected (i.e. the call originates from the CLI, not from
    inside a ZenML step).

    Args:
        cfg: Root configuration object.

    Returns:
        Best validation reward returned by the ZenML pipeline.
    """
    tracking = getattr(cfg, "tracking", None)
    mlflow_uri = str(getattr(tracking, "mlflow_tracking_uri", "mlruns"))
    stack_name = str(getattr(tracking, "zenml_stack_name", "wsmart-route-stack"))

    # The extra processes of a multi-device run (relaunched by Lightning with LOCAL_RANK set)
    # train directly: the run is tracked once, by rank 0, and they open no ZenML client.
    if int(os.environ.get("LOCAL_RANK", "0")) > 0:
        return run_training(cfg, sinks=[])
    if configure_zenml_stack is None or not configure_zenml_stack(mlflow_uri, stack_name=stack_name):
        logger.warning("ZenML stack configuration failed — falling back to direct training.")
        return run_training(cfg, sinks=[])
    try:
        result = zenml_train_pipeline_module.training_pipeline(cfg)
        return result if isinstance(result, float) else 0.0
    except Exception as exc:
        logger.warning(f"ZenML training pipeline failed — falling back to direct training: {exc}")
        return run_training(cfg, sinks=[])
