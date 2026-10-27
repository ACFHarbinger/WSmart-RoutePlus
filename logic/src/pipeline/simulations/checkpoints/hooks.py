"""
Checkpoint hooks for simulation lifecycle.

This module provides the CheckpointHook class, which orchestrates the
timing of checkpoint saves and manages state during simulation errors.

Attributes:
    CheckpointHook: Orchestrator for checkpoint timing and error handling.

Example:
    >>> # hook = CheckpointHook(checkpoint, 5, state_getter)
    >>> # hook.after_day()
"""

import contextlib
import time
import traceback
from typing import Any, Callable, Dict, Optional

from loguru import logger

try:
    from logic.src.tracking.core.run import (  # pyrefly: ignore[missing-import]  # tracking is excluded from the minimal package
        get_active_run,
    )
except ImportError:
    get_active_run = None  # type: ignore[assignment,misc]


class CheckpointHook:
    """
    Orchestrates checkpoint timing and error handling during simulation.

    Attributes:
        checkpoint: The checkpoint object managing persistence.
        checkpoint_interval: How often to save checkpoints (in days).
        day: Current simulation day.
        tic: Timestamp of when the current day started (for performance metrics).
        state_getter: Callable to retrieve the current simulation state for saving.
    """

    def __init__(self, checkpoint, checkpoint_interval: int, state_getter: Optional[Callable[[], Any]] = None):
        """Initialize Class.

        Args:
            checkpoint: The SimulationCheckpoint instance.
            checkpoint_interval: Frequency of checkpoints in days.
            state_getter: Function to retrieve the current simulation state.
        """
        self.checkpoint = checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.day = 0
        self.tic: Optional[float] = None
        self.state_getter = state_getter

    def get_current_day(self) -> int:
        """Get current day.

        Returns:
            The current simulation day index.
        """
        return self.day

    def set_timer(self, tic: float) -> None:
        """Set timer.

        Args:
            tic: Performance counter timestamp.
        """
        self.tic = tic

    def before_day(self, day: int) -> None:
        """Before day.

        Args:
            day: The day index about to start.
        """
        self.day = day

    def after_day(self, tic: Optional[float] = None, delete_previous: bool = False) -> None:
        """After day.

        Args:
            tic: Performance counter timestamp.
            delete_previous: If True, delete the last checkpoint before saving new one.
        """
        previous_checkpoint_day = self.checkpoint.find_last_checkpoint_day()
        if tic:
            self.tic = tic
        if (
            self.checkpoint
            and self.checkpoint_interval > 0
            and self.day % self.checkpoint_interval == 0
            and self.state_getter
        ):
            state_snapshot = self.state_getter()
            self.checkpoint.save_state(state_snapshot, self.day)
        if delete_previous:
            self.checkpoint.delete_checkpoint_day(previous_checkpoint_day)

    def on_error(self, error: Exception) -> Dict[str, Any]:
        """On error.

        Args:
            error: The exception that triggered the crash.

        Returns:
            Dictionary with crash diagnostics and partial results.
        """
        execution_time = time.perf_counter() - self.tic if self.tic else 0
        day = self.get_current_day()
        info = self.checkpoint.get_simulation_info()
        policy = info.get("policy", "")
        sample_id = info.get("sample", 0)
        logger.error(f"Crash in {policy} #{sample_id} at day {day}: {error}")

        traceback.print_exc()
        if self.checkpoint and self.state_getter:
            try:
                state_snapshot = self.state_getter()
                self.checkpoint.save_state(state_snapshot, self.day)
            except Exception as save_error:
                logger.error(f"Failed to save emergency checkpoint: {save_error}")

        with contextlib.suppress(Exception):
            run = get_active_run() if get_active_run is not None else None
            if run is not None:
                run.log_params(
                    {
                        "sim.crashed": True,
                        "sim.crash_day": int(day),
                        "sim.crash_policy": str(policy),
                        "sim.crash_error_type": type(error).__name__,
                        "sim.crash_execution_time": float(execution_time),
                    }
                )
                run.log_metric("sim/crash_day", float(day))

        return {
            "policy": policy,
            "sample_id": sample_id,
            "day": self.day,
            "error": str(error),
            "error_type": type(error).__name__,
            "execution_time": execution_time,
            "success": False,
        }

    def on_completion(self, policy: Optional[str] = None, sample_id: Optional[int] = None) -> None:
        """On completion.

        Args:
            policy: Optional policy name for cleanup.
            sample_id: Optional sample ID for cleanup.
        """
        if self.checkpoint:
            self.checkpoint.clear(policy, sample_id)
        if self.state_getter:
            state_snapshot = self.state_getter()
            self.checkpoint.save_state(state_snapshot, self.day, end_simulation=True)

        with contextlib.suppress(Exception):
            run = get_active_run() if get_active_run is not None else None
            if run is not None:
                run.log_params(
                    {
                        "sim.completed": True,
                        "sim.completion_day": int(self.day),
                    }
                )
