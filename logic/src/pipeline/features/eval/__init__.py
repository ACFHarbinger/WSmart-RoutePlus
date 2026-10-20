"""Evaluation pipeline entry point.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

"""

import contextlib
import random
from collections.abc import Sequence
from typing import Any, List, Optional

import numpy as np
import torch

import logic.src.tracking as wst
from logic.src.configs import Config
from logic.src.tracking.logging.pylogger import get_pylogger

from .engine import eval_dataset, eval_dataset_mp, get_best
from .validation import validate_eval_config

try:
    from logic.src.pipeline.features.eval.zenml_eval_pipeline import eval_pipeline
    from logic.src.tracking.integrations.zenml_bridge import configure_zenml_stack
except ImportError:
    configure_zenml_stack = None  # type: ignore[assignment]
    eval_pipeline = None  # type: ignore[assignment]

logger = get_pylogger(__name__)


def run_evaluate_model(cfg: Config, sinks: Optional[List[Any]] = None) -> None:
    """Main entry point for the evaluation script.

    Args:
        cfg: Root Hydra configuration with ``cfg.eval`` containing evaluation
            parameters.
        sinks: Optional list of tracking sinks (e.g. :class:`ZenMLBridge`)
            to attach to the WSTracker run.  When ``None`` (the default),
            :class:`MLflowBridge` is auto-attached if
            ``cfg.tracking.mlflow_enabled`` is ``True``.
    """
    # ----- ZenML dispatch (opt-in) -----
    tracking = getattr(cfg, "tracking", None)
    zenml_enabled = bool(getattr(tracking, "zenml_enabled", False))
    if zenml_enabled and sinks is None:
        _run_eval_via_zenml(cfg)
        return

    # Validate and sanitize config values
    validate_eval_config(cfg)

    ev = cfg.eval

    random.seed(ev.seed)
    np.random.seed(ev.seed)
    torch.manual_seed(ev.seed)

    # Resolve strategy from decoding config
    strategy = ev.decoding.strategy if ev.decoding else "greedy"

    # Resolve beam widths
    beam_widths: List[int] = []
    bw = ev.decoding.beam_width if ev.decoding else None
    if bw is None:
        beam_widths = [0]
    elif isinstance(bw, Sequence) and not isinstance(bw, str):  # list, tuple or OmegaConf ListConfig
        beam_widths = [int(w) for w in bw]
    else:
        beam_widths = [bw]

    # --- Centralised experiment tracking ---
    model_path = ev.policy.model.load_path if ev.policy else "unknown"
    _ev_graph = getattr(getattr(ev, "env", None), "graph", None)
    _ev_num_loc = getattr(_ev_graph, "num_loc", "")
    experiment_name = f"eval-{ev.problem}-{_ev_num_loc}loc-{strategy}"
    tracker = wst.init(experiment_name=experiment_name)
    run_tags = {
        "problem": ev.problem,
        "num_loc": str(_ev_num_loc),
        "strategy": strategy,
        "model_path": str(model_path),
        "seed": str(ev.seed),
        "val_size": str(ev.val_size),
    }
    run = tracker.start_run(experiment_name, run_type="evaluation", tags=run_tags)
    run.__enter__()

    # ----- Attach secondary sinks -----
    if sinks is not None:
        for sink in sinks:
            run.add_sink(sink)
    else:
        mlflow_enabled = bool(getattr(tracking, "mlflow_enabled", False))
        if mlflow_enabled:
            with contextlib.suppress(Exception):
                wst.MLflowBridge.attach(
                    run,
                    mlflow_tracking_uri=str(getattr(tracking, "mlflow_tracking_uri", "mlruns")),
                    experiment_name=str(getattr(tracking, "mlflow_experiment_name", "wsmart-route")),
                    run_name=run.run_id[:8],
                    tags=run_tags,
                )

    # Log evaluation params
    run.log_params(
        {
            "eval.problem": ev.problem,
            "eval.strategy": strategy,
            "eval.val_size": ev.val_size,
            "eval.eval_batch_size": ev.eval_batch_size,
            "eval.offset": ev.offset,
            "eval.model_path": str(model_path),
            "eval.beam_widths": str(beam_widths),
            "eval.num_loc": _ev_num_loc,
            "eval.area": getattr(_ev_graph, "area", ""),
            "eval.data_distribution": str(ev.data_distribution),
            "eval.multiprocessing": ev.multiprocessing,
        }
    )

    try:
        for beam_width in beam_widths:
            datasets = ev.datasets
            if not datasets:
                print("No datasets for evaluation found. Skipping evaluation.")
                run.__exit__(None, None, None)
                return

            datasets = [datasets] if isinstance(datasets, str) else [str(d) for d in datasets]

            for dataset_path in datasets:
                temp = ev.decoding.temperature if ev.decoding else 1.0
                eval_dataset(dataset_path, beam_width, temp, cfg, strategy=strategy, run=run)

        run.__exit__(None, None, None)
    except Exception as exc:
        run.__exit__(type(exc), exc, exc.__traceback__)
        raise


# ---------------------------------------------------------------------------
# ZenML dispatch
# ---------------------------------------------------------------------------


def _run_eval_via_zenml(cfg: Config) -> None:
    """Dispatch evaluation to the ZenML evaluation pipeline.

    Called when ``cfg.tracking.zenml_enabled`` is ``True`` and no external
    sinks were injected.

    Args:
        cfg: Configuration object containing evaluation parameters.
    """
    tracking = getattr(cfg, "tracking", None)
    mlflow_uri = str(getattr(tracking, "mlflow_tracking_uri", "mlruns"))
    stack_name = str(getattr(tracking, "zenml_stack_name", "wsmart-route-stack"))

    if configure_zenml_stack is None or not configure_zenml_stack(mlflow_uri, stack_name=stack_name):
        logger.warning("ZenML stack configuration failed — falling back to direct evaluation.")
        run_evaluate_model(cfg, sinks=[])
        return

    try:
        if eval_pipeline is not None:
            eval_pipeline(cfg)
    except Exception as exc:
        logger.warning(f"ZenML evaluation pipeline failed — falling back to direct evaluation: {exc}")
        run_evaluate_model(cfg, sinks=[])


__all__ = ["run_evaluate_model", "eval_dataset", "eval_dataset_mp", "get_best", "validate_eval_config"]
