"""Helper functions and utilities for the WSTracker Lightning integration.

This module provides support logic for the :class:`TrackingCallback`,
including hook registration mechanisms, visualization orchestration (attention
heatmaps, embeddings, loss landscapes), and metric extraction utilities. It
acts as the glue between PyTorch Lightning's lifecycle and the WSTracker API.

Attributes:
    register_monitoring_hooks: Configures model hooks for activation/gradient tracking.
    run_periodic_visualisations: Triggers high-level artifact generation.
    log_checkpoint_artifact: Registers model checkpoints in the run database.

Example:
    >>> from logic.src.tracking.helpers.lightning_helpers import extract_metrics
    >>> metrics = extract_metrics(trainer.callback_metrics, prefix="train/")
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, cast

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from logic.src.tracking.core.run import get_active_run
from logic.src.tracking.logging.pylogger import get_pylogger

try:
    from logic.src.tracking.hooks import (
        add_activation_statistics_hook,
        add_gradient_monitoring_hooks,
        add_gradient_nan_detector_hook,
        add_weight_distribution_monitor,
        compute_activation_statistics,
        register_hooks_with_run,
    )
except ImportError:
    add_activation_statistics_hook = None  # type: ignore[assignment]
    add_gradient_monitoring_hooks = None  # type: ignore[assignment]
    add_gradient_nan_detector_hook = None  # type: ignore[assignment]
    add_weight_distribution_monitor = None  # type: ignore[assignment]
    compute_activation_statistics = None  # type: ignore[assignment]
    register_hooks_with_run = None  # type: ignore[assignment]

try:
    from logic.src.tracking.hooks.activation_hooks import remove_all_hooks as remove_act_hooks
except ImportError:
    remove_act_hooks = None  # type: ignore[assignment]

try:
    from logic.src.tracking.hooks.gradient_hooks import remove_all_hooks as remove_grad_hooks
except ImportError:
    remove_grad_hooks = None  # type: ignore[assignment]

try:
    from logic.src.tracking.logging.visualization.heatmaps import plot_attention_heatmaps
except ImportError:
    plot_attention_heatmaps = None  # type: ignore[assignment]

try:
    from logic.src.tracking.logging.visualization.embeddings import (  # pyrefly: ignore[missing-import]  # optional viz module is not in this package
        log_weight_distributions,
        project_node_embeddings,
    )
except ImportError:
    log_weight_distributions = None  # type: ignore[assignment]
    project_node_embeddings = None  # type: ignore[assignment]

try:
    from logic.src.tracking.logging.visualization.landscape import (  # pyrefly: ignore[missing-import]  # optional viz module is not in this package
        plot_loss_landscape,
    )
except ImportError:
    plot_loss_landscape = None  # type: ignore[assignment]

try:
    from logic.src.tracking.profiling.profiler import _profiler_instance
except ImportError:
    _profiler_instance = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from logic.src.configs.tracking import TrackingConfig

logger = get_pylogger(__name__)


def _opt(cfg: Optional["TrackingConfig"], attr: str, default: Any = False) -> Any:
    """Reads a configuration flag with a fallback value.

    Args:
        cfg: The tracking configuration object.
        attr: Attribute name to read.
        default: Fallback value if configuration is None or attribute is missing.

    Returns:
        Any: The configuration value or default.
    """
    if cfg is None:
        return default
    return getattr(cfg, attr, default)


# ---------------------------------------------------------------------------
# Hook Management
# ---------------------------------------------------------------------------


def register_monitoring_hooks(
    cfg: Optional["TrackingConfig"], pl_module: pl.LightningModule, hook_data: Dict[str, Any]
) -> None:
    """Configures monitoring hooks on the module based on settings.

    Registers hooks for gradients, activations, weight distributions, and
    NaN detection if enabled in the tracking configuration.

    Args:
        cfg: The tracking configuration defining which hooks to enable.
        pl_module: The Lightning module to instrument.
        hook_data: Shared dictionary to store hook handles for later removal.
    """
    if add_activation_statistics_hook is None:
        logger.debug("Hooks module not available, skipping hook registration")
        return

    if _opt(cfg, "log_gradients"):
        try:
            hook_data["gradient"] = cast(Callable[..., Any], add_gradient_monitoring_hooks)(pl_module)
        except Exception:
            logger.debug("Failed to register gradient hooks", exc_info=True)

    if _opt(cfg, "log_activations"):
        try:
            hook_data["activation"] = add_activation_statistics_hook(pl_module)
        except Exception:
            logger.debug("Failed to register activation hooks", exc_info=True)

    if _opt(cfg, "log_weights"):
        try:
            hook_data["weight"] = cast(Callable[..., Any], add_weight_distribution_monitor)(pl_module)
        except Exception:
            logger.debug("Failed to register weight hooks", exc_info=True)

    if _opt(cfg, "nan_guard"):
        try:
            hook_data["nan_guard"] = cast(Callable[..., Any], add_gradient_nan_detector_hook)(
                pl_module, raise_on_nan=True
            )
        except Exception:
            logger.debug("Failed to register NaN guard hooks", exc_info=True)


def log_hook_stats_to_run(run: Any, epoch: int, hook_data: Dict[str, Any]) -> None:
    """Processes and logs accumulated hook statistics to the WSTracker run.

    Args:
        run: The active tracking run.
        epoch: Current epoch index for metric step logging.
        hook_data: Dictionary containing active hook handles and statistics.
    """
    if register_hooks_with_run is None:
        return

    # Gradient stats
    if "gradient" in hook_data:
        with contextlib.suppress(Exception):
            register_hooks_with_run(hook_data["gradient"], run, prefix="train/hooks")

    # Activation stats
    if "activation" in hook_data:
        try:
            stats = cast(Callable[..., Any], compute_activation_statistics)(
                hook_data["activation"].get("statistics", {})
            )
            for name, stat in stats.items():
                for k, v in stat.items():
                    if isinstance(v, (int, float)):
                        run.log_metric(f"train/act/{name}/{k}", float(v), step=epoch)
        except Exception:
            pass

    # Weight distribution stats
    if "weight" in hook_data:
        with contextlib.suppress(Exception):
            register_hooks_with_run(hook_data["weight"], run, prefix="train/hooks")


def remove_monitoring_hooks(hook_data: Dict[str, Any]) -> None:
    """Detaches and clears all registered monitoring hooks from the model.

    Args:
        hook_data: Dictionary containing currently active hook handles.
    """
    if remove_grad_hooks is None or remove_act_hooks is None:
        return

    for key in ["gradient", "nan_guard"]:
        if key in hook_data:
            with contextlib.suppress(Exception):
                remove_grad_hooks(hook_data[key])

    for key in ["activation"]:
        if key in hook_data:
            with contextlib.suppress(Exception):
                remove_act_hooks(hook_data[key])

    hook_data.clear()


# ---------------------------------------------------------------------------
# Visualisations & Profiling
# ---------------------------------------------------------------------------


def _log_attention_heatmaps(
    log_dir: str,
    cfg: Optional["TrackingConfig"],
    pl_module: pl.LightningModule,
    epoch: int,
) -> None:
    """Generates and logs attention heatmap visualizations.

    Args:
        log_dir: Base directory for storing visualization artifacts.
        cfg: Tracking configuration object.
        pl_module: The Lightning module containing the policy networks.
        epoch: Current epoch index for versioning the artifacts.
    """
    if _opt(cfg, "log_attention_heatmaps"):
        try:
            if plot_attention_heatmaps is not None:
                output_dir = os.path.join(log_dir, "attention_heatmaps")
                plot_attention_heatmaps(pl_module.policy, output_dir, epoch=epoch)

                run = get_active_run()
                if run is not None:
                    run.log_artifact(output_dir, artifact_type="visualization")
        except Exception:
            logger.debug("Failed to plot attention heatmaps", exc_info=True)


def _log_embeddings(
    log_dir: str,
    cfg: Optional["TrackingConfig"],
    pl_module: pl.LightningModule,
    epoch: int,
) -> None:
    """Generates and logs weight distribution and node embedding projections.

    Args:
        log_dir: Base directory for storing visualization artifacts.
        cfg: Tracking configuration object.
        pl_module: The Lightning module containing the policy networks.
        epoch: Current epoch index.
    """
    if _opt(cfg, "log_embeddings"):
        try:
            if log_weight_distributions is not None and project_node_embeddings is not None:
                tb_log_dir = os.path.join(log_dir, "tensorboard")
                log_weight_distributions(pl_module.policy, epoch, tb_log_dir)

                # Project embeddings if a validation batch is available
                if hasattr(pl_module, "val_dataset") and pl_module.val_dataset is not None:
                    try:
                        val_loader = pl_module.val_dataloader()
                        x_batch = next(iter(val_loader))
                        if isinstance(x_batch, torch.Tensor):
                            x_batch = x_batch.to(pl_module.device)
                            project_node_embeddings(pl_module.policy, x_batch, tb_log_dir, epoch=epoch)
                        else:
                            project_node_embeddings(pl_module.policy, x_batch, tb_log_dir, epoch=epoch)
                    except Exception:
                        logger.debug("Failed to project node embeddings", exc_info=True)
        except Exception:
            logger.debug("Failed to log embeddings", exc_info=True)


def _log_loss_landscape(
    log_dir: str,
    cfg: Optional["TrackingConfig"],
    pl_module: pl.LightningModule,
    epoch: int,
) -> None:
    """Computes and logs a 2D projection of the loss landscape.

    Args:
        log_dir: Base directory for output.
        cfg: Tracking configuration object.
        pl_module: The Lightning module to perturb.
        epoch: Current epoch index.
    """
    if _opt(cfg, "log_loss_landscape"):
        try:
            if plot_loss_landscape is not None:
                output_dir = os.path.join(log_dir, "loss_landscape")

                # Requires root cfg on pl_module — safe access
                root_cfg = getattr(pl_module, "cfg", None)
                if root_cfg is not None and hasattr(pl_module, "policy"):
                    plot_loss_landscape(
                        pl_module.policy,
                        root_cfg,
                        output_dir,
                        epoch=epoch,
                        size=20,
                        batch_size=8,
                        resolution=10,
                    )
                    run = get_active_run()
                    if run is not None:
                        run.log_artifact(output_dir, artifact_type="visualization")
        except Exception:
            logger.debug("Failed to plot loss landscape", exc_info=True)


def run_periodic_visualisations(
    cfg: Optional["TrackingConfig"],
    pl_module: pl.LightningModule,
    epoch: int,
) -> None:
    """Dispatches all scheduled visualization tasks for the current epoch.

    Args:
        cfg: The tracking configuration defining which plots are enabled.
        pl_module: The Lightning module being trained.
        epoch: Current epoch index.
    """
    log_dir = _opt(cfg, "log_dir", "logs")
    _log_attention_heatmaps(log_dir, cfg, pl_module, epoch)
    _log_embeddings(log_dir, cfg, pl_module, epoch)
    _log_loss_landscape(log_dir, cfg, pl_module, epoch)


def log_execution_profiling_report() -> None:
    """Gathers the latest profiling data and logs the report to the active run."""
    try:
        if _profiler_instance is not None:
            report = _profiler_instance.get_report()
            report.log_to_run()
    except Exception:
        logger.debug("Failed to log profiling report", exc_info=True)


# ---------------------------------------------------------------------------
# General Utilities
# ---------------------------------------------------------------------------


def extract_metrics(
    callback_metrics: Dict[str, Any],
    prefix: str,
) -> Dict[str, float]:
    """Filters and formats metrics from a callback dictionary by prefix.

    Args:
        callback_metrics: Dictionary of metrics (usually trainer.callback_metrics).
        prefix: Metric name prefix to filter by (e.g., 'val/').

    Returns:
        Dict[str, float]: Filtered mapping of metric names to scalars.
    """
    result: Dict[str, float] = {}
    for k, v in callback_metrics.items():
        if k.startswith(prefix):
            with contextlib.suppress(TypeError, ValueError):
                result[k] = float(v.item() if hasattr(v, "item") else v)
    return result


def log_checkpoint_artifact(
    run: Any,
    cb: ModelCheckpoint,
    epoch: int,
) -> None:
    """Registers best and last model checkpoints as run artifacts.

    Args:
        run: The active tracking run.
        cb: The ModelCheckpoint callback instance.
        epoch: Current epoch index for checkpoint metadata.
    """
    for attr, label in [("best_model_path", "best_checkpoint"), ("last_model_path", "last_checkpoint")]:
        path: Optional[str] = getattr(cb, attr, None)
        if path and os.path.exists(path):
            run.log_artifact(
                path,
                name=label,
                artifact_type="checkpoint",
                metadata={
                    "epoch": epoch,
                    "monitor": cb.monitor,
                    "best": attr == "best_model_path",
                },
            )
