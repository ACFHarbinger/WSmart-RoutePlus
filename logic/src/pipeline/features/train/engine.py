"""Training engine for WSmart-Route.

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

import hydra
import omegaconf
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger

import logic.src.pipeline.features.train.zenml_train_pipeline as zenml_train_pipeline_module
import logic.src.tracking as wst
from logic.src.configs import Config
from logic.src.interfaces import ITraversable
from logic.src.pipeline.callbacks import SpeedMonitor
from logic.src.pipeline.features.train.model_factory.builder import create_model
from logic.src.pipeline.rl.common.trainer import WSTrainer
from logic.src.tracking.logging.pylogger import get_pylogger

try:
    from logic.src.tracking.integrations.zenml_bridge import configure_zenml_stack
except ImportError:
    configure_zenml_stack = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from logic.src.pipeline.rl.common.base.module import RL4COLitModule

logger = get_pylogger(__name__)


def _build_experiment_name(cfg: Any) -> str:
    """Derive a human-readable experiment name from the Hydra config.

    Args:
        cfg: Root configuration object.

    Returns:
        Human-readable experiment name string.
    """
    task = getattr(cfg, "task", "train")
    task_cfg = getattr(cfg, task, cfg)
    env = getattr(task_cfg, "env", getattr(cfg, "env", None))
    policy = getattr(task_cfg, "policy", task_cfg)
    model = getattr(policy, "model", getattr(cfg, "model", None))

    parts = [
        getattr(env, "name", "env") if env else "env",
        str(getattr(env, "num_loc", "")) if env else "",
        getattr(model, "name", getattr(cfg, "model_name", "")) if model else "",
        getattr(cfg.rl, "algorithm", "") if hasattr(cfg, "rl") else "",
    ]
    return "-".join(p for p in parts if p)


def _build_callbacks(cfg: Any) -> list:
    """Instantiate training callbacks from Hydra config.

    Handles ``ITraversable`` configs (direct ``_target_`` dicts and nested
    name-to-config dicts), links ``CleanProgressBar`` ↔ ``TerminalChartCallback``
    when both are present, and always includes a :class:`SpeedMonitor`.

    Args:
        cfg: Root configuration object.

    Returns:
        List of instantiated callback objects.
    """
    callbacks: list = [SpeedMonitor(epoch_time=True)]
    if cfg.train.callbacks:
        for cb_cfg in cfg.train.callbacks:
            cb_cfg_obj: object = cb_cfg
            if isinstance(cb_cfg_obj, ITraversable):
                if "_target_" in cb_cfg_obj:
                    callbacks.append(hydra.utils.instantiate(cb_cfg_obj))
                else:
                    for _, actual_cfg in cb_cfg_obj.items():
                        actual_cfg_obj: object = actual_cfg
                        if isinstance(actual_cfg_obj, ITraversable) and "_target_" in actual_cfg_obj:
                            callbacks.append(hydra.utils.instantiate(actual_cfg_obj))
            else:
                with contextlib.suppress(Exception):
                    callbacks.append(hydra.utils.instantiate(cb_cfg))

    progress_bar = next((c for c in callbacks if c.__class__.__name__ == "CleanProgressBar"), None)
    terminal_chart = next((c for c in callbacks if c.__class__.__name__ == "TerminalChartCallback"), None)
    if progress_bar is not None and terminal_chart is not None:
        progress_bar.set_chart_callback(terminal_chart)  # type: ignore[attr-defined]

    return callbacks


def _get_primary_graph(cfg: Any, key: str, default: Any = None) -> Any:
    """Return a field from the primary graph config.

    Prioritizes ``env.graph`` (dynamically injected for stages) then falls back
    to ``env.curriculum_graphs[0]`` (the source of truth for root configs).

    Args:
        cfg: Root Hydra configuration object.
        key: Attribute name to retrieve (e.g. 'num_loc', 'n_days').
        default: Value to return when the key is absent.

    Returns:
        The attribute value or *default*.
    """
    try:
        task = getattr(cfg, "task", "train")
        task_cfg = getattr(cfg, task, cfg)
        env = getattr(task_cfg, "env", getattr(cfg, "env", None))
        if env is None:
            return default

        # 1. Try injected env.graph (DictConfigs from _build_stage_config)
        graph = getattr(env, "graph", None)
        if graph is not None:
            val = graph.get(key) if hasattr(graph, "get") else getattr(graph, key, None)
            if val is not None:
                return val

        # 2. Try curriculum_graphs[0] (Root Configs)
        graphs = getattr(env, "curriculum_graphs", None) or []
        primary = graphs[0] if graphs else None
        if primary is not None:
            val = primary.get(key) if hasattr(primary, "get") else getattr(primary, key, None)
            if val is not None:
                return val

        return default
    except Exception:
        return default


def _build_stage_config(cfg: Any, graph_cfg: Any, stage_idx: Optional[int] = None) -> Any:
    """Build a modified root config for a curriculum stage.

    Converts the root config to a plain dict, overwrites env.graph fields from
    the stage's graph_cfg, disables curriculum (to prevent infinite recursion),
    and returns a new unstructured DictConfig.

    Args:
        cfg: Root Hydra configuration object.
        graph_cfg: Graph config entry (DictConfig or dict) for this stage.

    Returns:
        Modified DictConfig for this stage.
    """
    if not OmegaConf.is_config(cfg):
        # run_training may be called programmatically with the Config dataclass.
        cfg = OmegaConf.structured(cfg)
    raw: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)  # type: ignore[assignment]
    assert isinstance(raw, dict), "cfg must be convertible to a plain dict"

    # Collect graph fields from the curriculum entry
    if hasattr(graph_cfg, "items"):
        g: Dict[str, Any] = dict(graph_cfg.items())
    elif hasattr(graph_cfg, "__dict__"):
        g = vars(graph_cfg)
    else:
        g = {}

    task = str(raw.get("task", "train"))
    task_cfg = raw.setdefault(task, {})
    assert isinstance(task_cfg, dict), f"{task} config must be a dictionary"
    env_graph: Dict[str, Any] = task_cfg.setdefault("env", {}).setdefault("graph", {})
    _GRAPH_KEYS = (
        "num_loc",
        "n_samples",
        "n_days",
        "start_day",
        "focus_graph",
        "focus_size",
        "area",
        "waste_type",
        "vertex_method",
        "distance_method",
        "dm_filepath",
        "edge_threshold",
        "edge_method",
        "load_dataset",
    )
    for key in _GRAPH_KEYS:
        val = g.get(key)
        if val is not None:
            env_graph[key] = val

    # If the stage doesn't specify load_dataset, clear it so the stage generates
    # its own data rather than loading a pre-built file sized for a different graph.
    if g.get("load_dataset") is None:
        env_graph.pop("load_dataset", None)

    # Override env.graph.reward with the per-graph reward config when present.
    # This allows each curriculum/eval graph to use different objective weights.
    # builder.py and data.py both read from env.graph.reward as the primary source.
    stage_reward = g.get("reward")
    if stage_reward is not None:
        reward_dict: Dict[str, Any]
        if hasattr(stage_reward, "items"):
            reward_dict = dict(stage_reward.items())
        elif hasattr(stage_reward, "__dict__"):
            reward_dict = vars(stage_reward)
        else:
            reward_dict = {}
        if reward_dict:
            env_graph["reward"] = reward_dict

    # Clear curriculum_graphs to prevent recursive dispatch in the stage
    task_env = task_cfg.setdefault("env", {})
    task_env["curriculum_graphs"] = []

    # Filter eval_graphs to only include the one corresponding to this stage
    # if stage_idx is provided and eval_graphs exists.
    if stage_idx is not None:
        eval_graphs = task_env.get("eval_graphs", [])
        if eval_graphs and stage_idx < len(eval_graphs):
            task_env["eval_graphs"] = [eval_graphs[stage_idx]]
        elif eval_graphs:
            # Fallback if eval_graphs is shorter than curriculum: empty list
            # which usually defaults to using env.graph
            task_env["eval_graphs"] = []

    return OmegaConf.create(raw)


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

    if torch.cuda.is_available() and cfg.train.precision in ("16-mixed", "bf16-mixed"):
        torch.set_float32_matmul_precision("medium")

    model_raw = create_model(cfg)
    model = cast("RL4COLitModule", model_raw)

    # Warm-start from the previous curriculum stage
    if prev_state_dict is not None:
        try:
            model.load_state_dict(prev_state_dict, strict=False)
            logger.info("Curriculum warm-start: loaded weights from previous stage.")
        except Exception as exc:
            logger.warning(f"Curriculum warm-start failed (continuing from scratch): {exc}")

    callbacks = _build_callbacks(cfg)
    trainer = WSTrainer(
        max_epochs=_get_primary_graph(cfg, "n_days", 1),
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

    if sinks is not None:
        for sink in sinks:
            run.add_sink(sink)
    else:
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
                    "source_line": 151,
                },
            )
            run.watch_file(val_ds)

        trainer.fit(model)

        # Reload best checkpoint so state_dict carries optimal weights into the next
        # curriculum stage (or into the final save), not just the last training epoch.
        ckpt_path = getattr(getattr(trainer, "checkpoint_callback", None), "best_model_path", None)
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
                logger.info("Loaded best checkpoint for state handoff: %s", ckpt_path)
            except Exception as exc:
                logger.warning("Could not reload best checkpoint (%s); using last-epoch weights.", exc)

        if save_final and cfg.train.final_model_path:
            logger.info("Saving final model weights to: %s", cfg.train.final_model_path)
            model.save_weights(cfg.train.final_model_path)
            logger.info("Final model weights saved successfully.")
            run.log_artifact(cfg.train.final_model_path, artifact_type="model")
        elif not cfg.train.final_model_path:
            logger.info("No final_model_path provided, skipping weight saving.")

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


def _log_training_params(run: wst.Run, cfg: Any) -> None:
    """Flatten and log relevant config sections as run parameters.

    Args:
        run: WSTracker run object.
        cfg: Root configuration object.
    """
    sections = {}
    for attr in ("train", "rl", "env", "model"):
        section = getattr(cfg, attr, None)
        if section is None:
            continue
        try:
            sections[attr] = omegaconf.OmegaConf.to_container(section, resolve=True)
        except Exception:
            with contextlib.suppress(Exception):
                sections[attr] = {k: getattr(section, k) for k in dir(section) if not k.startswith("_")}

    sections["seed"] = cfg.seed
    sections["experiment_name"] = getattr(cfg, "experiment_name", "")
    run.log_params(sections)


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

    if configure_zenml_stack is None or not configure_zenml_stack(mlflow_uri, stack_name=stack_name):
        logger.warning("ZenML stack configuration failed — falling back to direct training.")
        return run_training(cfg, sinks=[])
    try:
        result = zenml_train_pipeline_module.training_pipeline(cfg)
        return result if isinstance(result, float) else 0.0
    except Exception as exc:
        logger.warning(f"ZenML training pipeline failed — falling back to direct training: {exc}")
        return run_training(cfg, sinks=[])
