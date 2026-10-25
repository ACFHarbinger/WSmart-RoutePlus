"""Progress display, failure records and final metrics output for the simulator.

Holds :func:`display_log_metrics` (writes the aggregated mean and std sections of each
policy log and prints the summary table), the :class:`SimpleCounter` and
:class:`ProgressUpdater` progress wrappers that ``sequential_simulations`` drives, and
``_failed_sample``, the failure record that keeps a failed sample's partial KPIs. Moved
unchanged from ``manager.py``, which re-exports them.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

from logic.src.constants import ROOT_DIR, SIM_METRICS
from logic.src.tracking.logging.modules.storage import update_policy_log_section
from logic.src.tracking.logging.modules.summary_display import display_simulation_summary_table


def display_log_metrics(
    output_dir: str,
    size: int,
    n_samples: int,
    days: int,
    area: str,
    policies: List[str],
    log: Dict[str, Union[List[float], Dict[str, float]]],
    log_std: Optional[Dict[str, Union[List[float], Dict[str, float]]]] = None,
    lock: Optional[Any] = None,
    waste_type: str = "",
    data_distribution: str = "",
    run_name: Optional[str] = None,
) -> None:
    """
    Pretty-prints aggregated simulation results to the console.

    Args:
        output_dir: Base directory for results.
        size: Graph size (number of bins).
        n_samples: Number of samples/seeds processed.
        days: Simulation duration in days.
        area: Geographic area name.
        policies: List of policy names.
        log: Dict mapping policy name to mean metrics list.
        log_std: Dict mapping policy name to std dev metrics list (optional).
        lock: Thread/process lock for safe printing/logging.
        waste_type: Waste type suffix for directory naming.
        data_distribution: Data distribution subdirectory (e.g. 'gamma3', 'emp').
        run_name: Run name subdirectory for organizing outputs.
    """
    abs_output_dir = os.path.join(
        ROOT_DIR,
        output_dir,
        f"{days}days",
        f"{area}{size}_{waste_type}" if waste_type else f"{area}{size}",
        data_distribution,
        *(run_name,) if run_name else (),
    )

    for pol_name, mean_vals in log.items():
        pol_log_path = os.path.join(abs_output_dir, f"log_{pol_name}_{n_samples}N.json")
        mean_dict = dict(zip(SIM_METRICS, mean_vals, strict=False)) if isinstance(mean_vals, list) else mean_vals
        update_policy_log_section(pol_log_path, "mean", mean_dict, lock=lock)
        if log_std is not None and pol_name in log_std:
            std_vals = log_std[pol_name]
            std_dict = dict(zip(SIM_METRICS, std_vals, strict=False)) if isinstance(std_vals, list) else std_vals
            update_policy_log_section(pol_log_path, "std", std_dict, lock=lock)

    display_simulation_summary_table(
        log,
        title=f"Final Simulation Summary: [bold cyan]{area} {size} nodes[/] ({n_samples} Samples, {days} Days)",
        lock=lock,
    )


# Internal counter for display
class SimpleCounter:
    """
    Simple counter for display.

    Attributes:
        value: The current value of the counter.
    """

    def __init__(self, val=0):
        """
        Initializes the simple counter.

        Args:
            val: Initial value. Defaults to 0.
        """
        self.value = val

    def update(self, n=1):
        """
        Updates the counter by adding n.

        Args:
            n: Value to add.
        """
        self.value += n


# Progress bar wrapper to maintain compatibility with RunningState which calls .update(1)
class ProgressUpdater:
    """
    Wrapper for the progress bar to maintain compatibility with RunningState.

    Calls the process_display_updates function to update the progress bar.

    Attributes:
        display: The display object to update (SimulationMonitor).
        shared_metrics: The shared metrics dictionary to update.
        log_tmp: The temporary log dictionary to update.
        last_reported_days: The last reported days dictionary to update.
        policy_names: The list of policy names.
        loop_tic: The loop start timestamp.
        counter: The counter object (SimpleCounter).
        process_display_updates: The function to update the display.
    """

    def __init__(self, display, shared_metrics, log_tmp, last_reported_days, policy_names, loop_tic, counter):
        """
        Initializes the progress updater wrapper.

        Args:
            display: The display object to update.
            shared_metrics: The shared metrics dictionary to update.
            log_tmp: The temporary log dictionary to update.
            last_reported_days: The last reported days dictionary to update.
            policy_names: The list of policy names.
            loop_tic: The loop tic value.
            counter: The counter object.
        """
        self.display = display
        self.shared_metrics = shared_metrics
        self.log_tmp = log_tmp
        self.last_reported_days = last_reported_days
        self.policy_names = policy_names
        self.loop_tic = loop_tic
        self.counter = counter

        from logic.src.pipeline.features.test.orchestrator.monitor import process_display_updates

        self.process_display_updates = process_display_updates

    def set_values(self, display, shared_metrics, log_tmp, last_reported_days, policy_names, loop_tic, counter) -> None:
        """
        Set the values for the progress bar.

        Args:
            display: Display object to update.
            shared_metrics: Shared metrics dictionary to update.
            log_tmp: Temporary log dictionary to update.
            last_reported_days: Last reported days dictionary to update.
            policy_names: List of policy names.
            loop_tic: Loop tic value.
            counter: Counter object.
        """
        self.display = display
        self.shared_metrics = shared_metrics
        self.log_tmp = log_tmp
        self.last_reported_days = last_reported_days
        self.policy_names = policy_names
        self.loop_tic = loop_tic
        self.counter = counter

    def update(self, n=1) -> None:
        """
        Update the progress bar.

        Args:
            n: Number of steps to update the progress bar.
        """
        if self.display:
            self.process_display_updates(
                self.display,
                self.shared_metrics,
                self.log_tmp,
                self.last_reported_days,
                self.policy_names,
                self.loop_tic,
                self.counter,
            )


def _failed_sample(
    err: Any,
    pol_name: str,
    sample_id: Any,
    error: str,
    traceback_text: str = "",
) -> Dict[str, Any]:
    """Keep partial KPIs on a failed sample so aggregates can count ``n_failed``."""
    payload = err if isinstance(err, dict) else {}
    entry: Dict[str, Any] = {
        "policy": payload.get("policy", pol_name),
        "sample": sample_id,
        "sample_id": payload.get("sample_id", sample_id),
        "error": payload.get("error", error),
        "traceback": payload.get("traceback", traceback_text),
        "status": "failed",
        "error_type": payload.get("error_type") or payload.get("exception_type"),
        "exception_type": payload.get("exception_type") or payload.get("error_type"),
        "day": payload.get("failed_day", payload.get("day")),
    }
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        entry["metrics"] = dict(metrics)
    return entry
