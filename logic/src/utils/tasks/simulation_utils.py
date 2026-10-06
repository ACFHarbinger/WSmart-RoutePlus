"""
Utilities for parallel task execution.

This module provides functions to prepare arguments for parallel tasks
and print execution information.

Attributes:
    prepare_parallel_task_args: Prepare argument tuples for parallel task execution.
    print_execution_info: Print information about parallel execution configuration.

Example:
    >>> import simulation_utils
    >>> simulation_utils.prepare_parallel_task_args(policies, n_samples, indices, sample_idx_ls)
    >>> simulation_utils.print_execution_info(task_count, n_cores)
"""

import time
from typing import Any, List, Tuple

import logic.src.constants as udef


def prepare_parallel_task_args(
    policies: List[str],
    n_samples: int,
    indices: List[Any],
    sample_idx_ls: List[List[int]],
) -> List[Tuple[Any, ...]]:
    """
    Prepare argument tuples for parallel task execution.

    Args:
        policies: Expanded policy name list.
        n_samples: Number of samples per policy.
        indices: Bin subset indices per sample.
        sample_idx_ls: Sample index lists per policy.

    Returns:
        List[Tuple[Any, ...]]: List of argument tuples for parallel tasks.
    """
    # Only the samples still listed per policy: a resumed run must skip finished ones.
    if n_samples > 1:
        return [(indices[sid], sid, pol_id) for pol_id in range(len(policies)) for sid in sample_idx_ls[pol_id]]
    else:
        return [(indices[0], 0, pol_id) for pol_id in range(len(policies)) if 0 in sample_idx_ls[pol_id]]


def print_execution_info(task_count: int, n_cores: int) -> None:
    """
    Print information about parallel execution configuration.

    Args:
        task_count (int): Number of tasks.
        n_cores (int): Number of CPU cores.
    """
    if n_cores > 1:
        print(f"\n[INFO] Launching {task_count} WSmart Route simulations on {n_cores} CPU cores...")
        max_lock_timeout = time.strftime("%H:%M:%S", time.gmtime(udef.LOCK_TIMEOUT))
        proc_lock_timeout = time.strftime("%H:%M:%S", time.gmtime(udef.CORE_LOCK_WAIT_TIME))
        print(f"\n[INFO] Maximum lock wait time: {max_lock_timeout} ({proc_lock_timeout} per used thread)")
    else:
        print(f"\n[INFO] Launching {task_count} WSmart Route simulations on a single CPU core...")
