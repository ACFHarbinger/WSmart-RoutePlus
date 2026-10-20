"""Simulation and training logging modules.

This package provides granular logging components for different aspects of
the system, including metric tracking, disk storage, GUI integration, and
statistical analysis of experiment results.

Attributes:
    log_values: Records scalar values and tensors to the log.
    log_epoch: Summarizes epoch-level metrics.
    get_loss_stats: Computes training progress statistics.
    setup_system_logger: Initializes file and stream handlers.
    sort_log: Reorders log entries by timestamp or epoch.
    log_to_json: Serializer for experiment logs.
    log_to_json2: Alternative serializer for nested data.
    log_to_pickle: Binary serializer for efficient storage.
    update_log: Adds new entries to an existing log file.
    load_log_dict: Deserializes logs from disk.
    output_stats: Prints experiment results summary to console.
    runs_per_policy: Groups results by solver/policy name.
    final_simulation_summary: Generates an end-of-run simulation report.
    send_daily_output_to_gui: Pushes simulation state to the desktop app.
    send_final_output_to_gui: Pushes final results to the desktop app.

Example:
    >>> from logic.src.tracking.logging.modules import setup_system_logger
    >>> setup_system_logger("experiment.log")
"""

from logic.src.tracking.logging.modules.summary_display import final_simulation_summary

from .analysis import load_log_dict, output_stats, runs_per_policy
from .gui import send_daily_output_to_gui, send_final_output_to_gui
from .metrics import get_loss_stats, log_epoch, log_values
from .storage import (
    log_to_json,
    log_to_json2,
    log_to_pickle,
    setup_system_logger,
    sort_log,
    update_log,
)

__all__ = [
    "log_values",
    "log_epoch",
    "get_loss_stats",
    "setup_system_logger",
    "sort_log",
    "log_to_json",
    "log_to_json2",
    "log_to_pickle",
    "update_log",
    "load_log_dict",
    "output_stats",
    "runs_per_policy",
    "final_simulation_summary",
    "send_daily_output_to_gui",
    "send_final_output_to_gui",
]
