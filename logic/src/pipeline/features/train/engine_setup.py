"""Config and callback helpers for the training engine.

Moved from ``logic.src.pipeline.features.train.engine``.
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Optional

import hydra
import omegaconf
from omegaconf import OmegaConf

import logic.src.tracking as wst
from logic.src.interfaces import ITraversable
from logic.src.pipeline.callbacks import SpeedMonitor

try:
    from logic.src.tracking.integrations.zenml_bridge import configure_zenml_stack
except ImportError:
    configure_zenml_stack = None  # type: ignore[assignment]


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
        "init_accumulation_days",
        "state_refresh_fraction",
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
