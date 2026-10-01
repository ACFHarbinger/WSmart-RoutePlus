"""
Checkpoint file persistence management.

This module provides the SimulationCheckpoint class, which handles saving
and loading the simulation state to/from disk for fault tolerance.

Attributes:
    SimulationCheckpoint: Manager for checkpoint file persistence.

Example:
    >>> # cp = SimulationCheckpoint(output_dir="results", policy="HGS")
    >>> # cp.save_state(state, day=10)
"""

import contextlib
import hashlib
import os
import pickle
from datetime import datetime
from typing import Any, Optional, Tuple

from loguru import logger

try:
    from logic.src.tracking.core.run import get_active_run
except ImportError:
    get_active_run = None  # type: ignore[assignment,misc]


def _temporary_checkpoint_dir(output_dir: str, checkpoint_dir: str) -> str:
    """Return the mid-run checkpoint directory for one results directory.

    ``output_dir`` is the run's results directory, which already includes
    ``sim.run_name``. A relative ``checkpoint_dir`` is stored under that
    directory, beside the final-artifact folder, so two runs of the same
    policy and sample do not share files. Completion clears only this live
    directory. An absolute ``checkpoint_dir`` is an explicit location and is
    used as a base with a stable results-directory namespace.
    """
    if os.path.isabs(checkpoint_dir):
        run_key = hashlib.sha256(os.fsencode(os.path.realpath(output_dir))).hexdigest()
        return os.path.join(checkpoint_dir, run_key, "live")
    if ".." in checkpoint_dir.replace("\\", "/").split("/"):
        raise ValueError("Relative checkpoint_dir must stay within the run directory")
    return os.path.join(os.path.abspath(output_dir), f"{checkpoint_dir}_live")


class SimulationCheckpoint:
    """
    Manages checkpoint file persistence for a single simulation run.
    Attributes:
        checkpoint_dir: Directory for temporary checkpoints.
        output_dir: Directory for final simulation artifacts.
        policy: The name of the policy being simulated.
        sample_id: The index of the current waste sample.
    """

    def __init__(self, output_dir: str, checkpoint_dir: str = "temp", policy: str = "", sample_id: int = 0):
        """Initialize Class.

        Args:
            output_dir: Root directory for permanent outputs.
            checkpoint_dir: Subdirectory for temporary state files.
            policy: Policy identifier string.
            sample_id: Integer sample ID.
        """
        self.checkpoint_dir = _temporary_checkpoint_dir(output_dir, checkpoint_dir)
        self.output_dir = (
            os.path.join(os.path.dirname(self.checkpoint_dir), "final")
            if os.path.isabs(checkpoint_dir)
            else os.path.join(output_dir, checkpoint_dir)
        )
        self.policy = policy
        self.sample_id = sample_id

    def get_simulation_info(self) -> dict:
        """Get simulation info.

        Returns:
            Dictionary with policy and sample_id.
        """
        return {"policy": self.policy, "sample": self.sample_id}

    def get_checkpoint_file(self, day: Optional[int] = None, end_simulation: bool = False) -> str:
        """Get checkpoint file.

        Args:
            day: Specific day to retrieve, or None for the latest.
            end_simulation: If True, look in the output_dir instead of temp.

        Returns:
            Full path to the checkpoint file.
        """
        parent_dir = self.output_dir if end_simulation else self.checkpoint_dir
        if day is not None:
            return os.path.join(parent_dir, f"checkpoint_{self.policy}_{self.sample_id}_day{day}.pkl")
        last_day = self.find_last_checkpoint_day()
        return os.path.join(parent_dir, f"checkpoint_{self.policy}_{self.sample_id}_day{last_day}.pkl")

    def save_state(self, state: Any, day: int = 0, end_simulation: bool = False) -> None:
        """Save state.

        Args:
            state: The simulation state dictionary to persist.
            day: The simulation day index.
            end_simulation: Whether to save as a final result.
        """
        checkpoint_data = {
            "state": state,
            "policy": self.policy,
            "sample_id": self.sample_id,
            "day": day,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0",
        }

        checkpoint_file = self.get_checkpoint_file(day, end_simulation)
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        with open(checkpoint_file, "wb") as f:
            pickle.dump(checkpoint_data, f)

        with contextlib.suppress(Exception):
            run = get_active_run() if get_active_run is not None else None
            if run is not None:
                run.log_metric("checkpoint/day_saved", float(day))
                run.log_dataset_event(
                    "save",
                    file_path=checkpoint_file,
                    metadata={
                        "event": "checkpoint_save",
                        "policy": self.policy,
                        "sample_id": self.sample_id,
                        "day": day,
                        "end_simulation": end_simulation,
                        "variable_name": "checkpoint_data",
                        "source_file": "checkpoints/persistence.py",
                        "source_line": 87,
                    },
                )

    def load_state(self, day: Optional[int] = None) -> Tuple[Optional[Any], int]:
        """Load state.

        Args:
            day: Specific day to load, or None for the latest available.

        Returns:
            Tuple of (state_dictionary, resumed_day_index).
        """
        checkpoint_files = []

        if day is not None:
            checkpoint_files.append(self.get_checkpoint_file(day))

        checkpoint_files.append(self.get_checkpoint_file(self.find_last_checkpoint_day()))

        for checkpoint_file in checkpoint_files:
            if os.path.exists(checkpoint_file):
                try:
                    with open(checkpoint_file, "rb") as f:
                        checkpoint_data = pickle.load(f)

                    if (
                        checkpoint_data.get("policy") == self.policy
                        and checkpoint_data.get("sample_id") == self.sample_id
                    ):
                        resumed_day = checkpoint_data.get("day", 0)
                        with contextlib.suppress(Exception):
                            run = get_active_run() if get_active_run is not None else None
                            if run is not None:
                                run.log_params(
                                    {
                                        "checkpoint.resumed": True,
                                        "checkpoint.resume_day": int(resumed_day),
                                        "checkpoint.resume_policy": self.policy,
                                        "checkpoint.resume_sample_id": self.sample_id,
                                    }
                                )
                                run.log_dataset_event(
                                    "load",
                                    file_path=checkpoint_file,
                                    metadata={
                                        "event": "checkpoint_resume",
                                        "day": int(resumed_day),
                                        "variable_name": "checkpoint_data",
                                        "source_file": "checkpoints/persistence.py",
                                        "source_line": 139,
                                    },
                                )
                        return checkpoint_data["state"], resumed_day
                    else:
                        logger.warning(f"Checkpoint mismatch: expected {self.policy}_{self.sample_id}")
                except Exception as e:
                    logger.error(f"Error loading checkpoint {checkpoint_file}: {e}")
        logger.warning("No valid checkpoint found")
        return None, 0

    def find_last_checkpoint_day(self) -> int:
        """Find last checkpoint day.

        Returns:
            The integer index of the latest day found in the checkpoint directory.
        """
        max_day = 0
        pattern = f"checkpoint_{self.policy}_{self.sample_id}_day"
        if not os.path.exists(self.checkpoint_dir):
            return 0
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith(pattern) and filename.endswith(".pkl"):
                try:
                    day_num = int(filename.split("day")[1].split(".pkl")[0])
                    max_day = max(max_day, day_num)
                except (ValueError, IndexError):
                    continue
        return max_day

    def clear(self, policy: Optional[str] = None, sample_id: Optional[int] = None) -> int:
        """Clear.

        Args:
            policy: Policy filter (defaults to current).
            sample_id: Sample filter (defaults to current).

        Returns:
            Number of files removed.
        """
        if policy is None:
            policy = self.policy
        if sample_id is None:
            sample_id = self.sample_id

        pattern = f"checkpoint_{policy}_{sample_id}"
        removed_count = 0
        if not os.path.exists(self.checkpoint_dir):
            return 0
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith(pattern) and filename.endswith(".pkl"):
                try:
                    os.remove(os.path.join(self.checkpoint_dir, filename))
                    removed_count += 1
                except Exception as e:
                    logger.error(f"Error removing {filename}: {e}")
        return removed_count

    def delete_checkpoint_day(self, day: int) -> bool:
        """Delete checkpoint day.

        Args:
            day: The day index to delete.

        Returns:
            True if the file was found and deleted.
        """
        checkpoint_file = self.get_checkpoint_file(day)
        try:
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing {checkpoint_file}: {e}")
            return False
