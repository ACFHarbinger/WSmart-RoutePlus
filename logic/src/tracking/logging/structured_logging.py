"""Structured JSON logging utilities for ELK stack integration.

Provides handlers and formatters to output logs in JSON format,
optionally sending them via TCP to Logstash.

Attributes:
    get_structured_logger: Returns a logger configured for JSON output.
    log_test_metric: Helper to log a specific test metric.
    log_benchmark_metric: Helper to log benchmark results.

Example:
    >>> logger = get_structured_logger("test_run", "logs/test.json")
    >>> logger.info("Training started", extra={"epoch": 1})
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt  # noqa: F401
import pandas as pd
import wandb
from omegaconf import DictConfig

from logic.src.configs import Config

from .json_formatter import JsonFormatter
from .logstash_handler import LogstashTcpHandler


def get_structured_logger(
    name: str = "wsmart.structured",
    level: int = logging.INFO,
    logstash_host: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Get a logger configured for structured JSON output.

    Args:
        name: Name of the logger instance. Defaults to "wsmart.structured".
        level: Logging level (e.g. logging.INFO). Defaults to logging.INFO.
        logstash_host: Optional hostname of a Logstash listener. Defaults to None.
        log_file: Optional local file path for JSON logging. Defaults to None.

    Returns:
        logging.Logger: The configured structured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if not logger.handlers:
        # Console handler for local debugging
        console = logging.StreamHandler()
        console.setFormatter(JsonFormatter())
        logger.addHandler(console)

        # File handler if path provided
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(exist_ok=True, parents=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(JsonFormatter())
            logger.addHandler(file_handler)

        # Logstash handler if host provided
        if logstash_host:
            ls_handler = LogstashTcpHandler(host=logstash_host)
            logger.addHandler(ls_handler)

    return logger


def log_test_metric(name: str, value: Any, logger_name: str = "wsmart.structured") -> None:
    """Convenience function to log a test metric.

    Args:
        name: The name of the metric.
        value: The value of the metric.
        logger_name: The logger to use. Defaults to "wsmart.structured".
    """
    logger = logging.getLogger(logger_name)
    extra = {"type": "test_metric", "metric_name": name, "metric_value": value}
    logger.info(f"Metric: {name}={value}", extra={"extra_fields": extra})


def log_benchmark_metric(
    benchmark: str,
    metrics: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    logger_name: str = "wsmart.benchmark",
) -> None:
    """Convenience function to log benchmark metrics.

    Args:
        benchmark: The name of the benchmark.
        metrics: Dictionary of metric values.
        metadata: Optional metadata dictionary. Defaults to None.
        logger_name: The logger to use. Defaults to "wsmart.benchmark".
    """
    # Ensure logger is initialized with a file if not already
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        get_structured_logger(name=logger_name, log_file="logs/benchmarks/benchmarks.jsonl")

    extra = {
        "benchmark": benchmark,
        "metrics": metrics,
        "metadata": metadata or {},
        "type": "performance_benchmark",
    }
    logger.info(f"Benchmark: {benchmark}", extra={"extra_fields": extra})


def log_training(
    loss_keys: List[str], table_df: pd.DataFrame, cfg: Union[Config, DictConfig], plot_logs: bool = False
) -> Union[List[str], Tuple[List[str], str, List[int], pd.DataFrame, str, str]]:
    """Logs comprehensive training history to Parquet, WandB, and generates Plots.

    Args:
        loss_keys: List of loss column names.
        table_df: DataFrame containing training stats.
        cfg: Root Hydra configuration with training and RL parameters.
        plot_logs: Whether to generate plots. Defaults to False.

    Returns:
        Optional[List[str]]: List of paths to generated plots, or None if plotting is disabled.
    """
    rl = cfg.rl

    log_dir: str = getattr(rl, "log_dir", "logs")
    save_dir: str = getattr(rl, "save_dir", "outputs")
    checkpoints_dir: str = getattr(rl, "checkpoints_dir", "outputs")
    wandb_mode: str = getattr(rl, "wandb_mode", "disabled")
    _run_name: str = getattr(rl, "run_name", "run")  # noqa: F841

    output_dir: str = os.path.join(
        log_dir,
        os.path.relpath(save_dir, start=checkpoints_dir),
    )
    os.makedirs(output_dir, exist_ok=True)
    table_df.to_parquet(os.path.join(output_dir, "table.parquet"), engine="pyarrow")

    swapped_df: pd.DataFrame = table_df.swaplevel(axis=1)
    swapped_df.columns = ["_".join(col).strip() for col in swapped_df.columns]
    if wandb_mode != "disabled":
        wandb_table: Any = wandb.Table(dataframe=swapped_df)
        wandb.log({"training_table": wandb_table})

    train = cfg.train
    xname: str = "day" if train.train_time else "epoch"
    x_values: List[int] = list(range(table_df.shape[0]))
    if plot_logs:
        from logic.src.utils.plotting.log_visualization import plot_training_logs  # lazy — avoids circular import

        return plot_training_logs(loss_keys, xname, x_values, swapped_df, output_dir, wandb_mode)
    return (loss_keys, xname, x_values, swapped_df, output_dir, wandb_mode)
