"""Tracking configuration dataclass for WSmart-Route.

Attributes:
    TrackingConfig: Tracking configuration dataclass for WSmart-Route.

Example:
    >>> from logic.src.configs.tracking import TrackingConfig
    >>> config = TrackingConfig()
    >>> print(config)
    TrackingConfig(name='', project_name=None, use_mlflow=False, use_tensorboard=False, use_wandb=True)
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrackingConfig:
    """Centralised tracking backend configuration.

    Controls both the native WSTracker (SQLite) and the optional MLflow
    secondary sink.  When ``mlflow_enabled`` is ``True``, every metric,
    parameter, and artifact logged to WSTracker is also mirrored to the
    configured MLflow server.

    Attributes:
            ``artifacts/`` subtree for the native WSTracker backend.
        mlflow_enabled: When ``True``, attach an :class:`MLflowBridge`
            sink to every new run so data is dual-written to MLflow.
        mlflow_tracking_uri: MLflow tracking server URI.  Supports local
            paths (``mlruns``), ``http://`` addresses, and Databricks
            workspace URIs.
        mlflow_experiment_name: MLflow experiment name to create/reuse.
            UI.  Defaults to the first 8 characters of the WSTracker
            run UUID when ``None``.
        ray_tune_storage_path: Root directory for Ray Tune trial logs and
            checkpoints.  Defaults to ``ray_results/`` in the project root.
        ray_tune_mlflow_enabled: When ``True``, each Ray Tune trial also
            logs to MLflow via ``MLflowLoggerCallback``.  Requires
            ``mlflow_enabled`` to also be ``True`` for the URI/experiment
            settings to be available.
        zenml_enabled: When ``True``, training, simulation, and HPO runs are
            orchestrated through versioned ZenML pipelines instead of being
            dispatched directly.  Requires ``zenml`` to be installed
            (``uv add "zenml[mlflow]"``).
            default local store (``~/.zenml``).  Supported formats:
            ``sqlite:///path/to/store.db`` and
            ``mysql://user:password@host/database``.
        zenml_stack_name: Name of the ZenML stack to activate for pipeline
            runs.  The stack must include an ``mlflow_tracker`` experiment
            tracker component pointing at ``mlflow_tracking_uri``.
        wandb_mode: Weights & Biases mode (``'online'``, ``'offline'``,
            ``'disabled'``).
        no_tensorboard: If ``True``, disable TensorBoard logging.
        no_progress_bar: If ``True``, disable the progress bar.
        log_dir: Directory to save logs.
        verbose: If ``True``, enable verbose logging.
        profile: If ``True``, enable function-level execution time profiling.
        log_step: Frequency of metric logging (e.g., every N steps).
        log_level: Logging level (e.g., ``'INFO'``, ``'DEBUG'``).
        log_file: Path to the log file.
        log_gradients: Enabling per-layer gradient monitoring hooks.
        log_weights: Enabling model weight distribution logging.
        log_activations: Capture activation statistics (mean, std, sparsity) per epoch.
        log_attention: Capture attention weights per epoch.
        log_memory: Track GPU/CPU memory snapshots per epoch.
        log_throughput: Track training step throughput (samples/sec).
        log_embeddings: Project node embeddings to TensorBoard at end of training.
        log_loss_landscape: Plot loss landscape periodically.
        log_attention_heatmaps: Plot attention weight heatmaps periodically.
        log_profiling_report: Generate and log execution profiling report at train end.
        nan_guard: Enable gradient NaN/Inf detection hooks (raises on detection).
        viz_every_n_epochs: Frequency for expensive visualizations (embeddings,
            loss landscape, attention heatmaps).  ``0`` means only at train end.
    """

    mlflow_enabled: bool = False
    mlflow_tracking_uri: str = "test_mlruns" if os.environ.get("TEST_MODE") == "true" else "mlruns"
    mlflow_experiment_name: str = "wsmart-route"
    ray_tune_storage_path: str = "ray_results"
    ray_tune_mlflow_enabled: bool = False
    zenml_enabled: bool = False
    zenml_stack_name: str = "wsmart-route-stack"

    # --- Logging & CLI Control (Migrated from root Config) ---
    wandb_mode: str = "offline"
    no_tensorboard: bool = False
    no_progress_bar: bool = False
    log_dir: str = "logs"
    verbose: bool = True
    profile: bool = False
    log_step: int = 10
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # --- Instrumentation & Diagnostics ---
    log_gradients: bool = False
    log_weights: bool = False
    log_activations: bool = False
    log_attention: bool = False
    log_memory: bool = False
    log_throughput: bool = False
    log_embeddings: bool = False
    log_loss_landscape: bool = False
    log_attention_heatmaps: bool = False
    log_profiling_report: bool = False
    nan_guard: bool = False
    viz_every_n_epochs: int = 0
