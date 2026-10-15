"""Logging systems for terminal, file, and GUI communication.

This module acts as a facade, delegating to specialized sub-modules for
analysis, GUI communication, metrics logging, and storage.

Attributes:
    log_values: Records generic metric values.
    log_epoch: Logs training progress for an epoch.
    get_loss_stats: Computes loss statistics from log buffers.
    setup_system_logger: Initializes main system loggers.
    sort_log: Sorts log records by timestamp.
    log_to_json: Serializes log buffer to JSON.
    log_to_json2: Alternative JSON serialization.
    log_to_pickle: Serializes log buffer to Pickle.
    update_log: Appends new records to a log file.
    load_log_dict: Loads a log file into a dictionary.
    output_stats: Prints statistical analysis of logs.
    runs_per_policy: Aggregates run counts by policy.
    final_simulation_summary: Produces end-of-sim summary table.
    send_daily_output_to_gui: Dispatches daily updates to the GUI layer.
    send_final_output_to_gui: Dispatches final results to the GUI layer.

Example:
    >>> from logic.src.tracking.logging.log_utils import log_epoch
    >>> log_epoch(epoch=1, loss=0.5, accuracy=0.90)
"""

from .modules.analysis import (
    _ACCEPTANCE_CRITERIA,
    _MANDATORY_PREFIXES,
    _ROUTE_IMPROVERS,
    _ROUTING_ALGORITHM_ENGINES,
    _ROUTING_ALGORITHMS,
    ResultsDB,
    _parse_slug,
    display_per_policy_simulation_summary,
    display_simulation_summary_table,
    final_simulation_summary,
    load_log_dict,
    output_stats,
    runs_per_policy,
)
from .modules.gui import send_daily_output_to_gui, send_final_output_to_gui
from .modules.metrics import get_loss_stats, log_epoch, log_values
from .modules.storage import (
    log_to_json,
    log_to_json2,
    log_to_pickle,
    setup_system_logger,
    sort_log,
    update_log,
    update_policy_log_section,
)

__all__ = [
    "ResultsDB",
    "_ACCEPTANCE_CRITERIA",
    "_MANDATORY_PREFIXES",
    "_ROUTING_ALGORITHM_ENGINES",
    "_ROUTING_ALGORITHMS",
    "_ROUTE_IMPROVERS",
    "_parse_slug",
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
    "display_simulation_summary_table",
    "display_per_policy_simulation_summary",
    "send_daily_output_to_gui",
    "send_final_output_to_gui",
    "update_policy_log_section",
]
