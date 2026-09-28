"""Parallel execution engine for simulations.

Attributes:
    run_parallel_simulations: Executes simulations in parallel using a multiprocessing pool.
    execute_and_monitor_tasks: Executes parallel tasks and monitors their progress.
    handle_shutdown: Handles graceful shutdown on keyboard interrupt.
    cleanup_multiprocessing_pool: Safely closes and joins the multiprocessing pool.

Example:
    >>> from logic.src.pipeline.features.test.orchestrator.parallel_runner import run_parallel_simulations
    >>> run_parallel_simulations(cfg, device, indices, sample_idx_ls, weights_path, lock, manager, n_cores, task_count, log_file, original_stderr, shared_metrics)
    ({'random': [array([164.02479, 181.58765, 167.54913, ...,  76.02663,  83.97055,
         56.46619])], 'ours': [array([156.92104, 138.96408, 165.01206, ..., 110.50811, 110.56278,
         56.57213])], 'ours2': [array([151.85719, 136.31088, 166.38385, ..., 127.60449, 127.63972,
         87.26818])]}, {'random': [nan], 'ours': [nan], 'ours2': [nan]}, [])
"""

import multiprocessing as mp
import os
from multiprocessing.pool import Pool
from typing import Any, Dict, List, Optional, Tuple

import logic.src.constants as udef
from logic.src.configs import Config
from logic.src.pipeline.features.test.orchestrator.monitor import (
    collect_all_task_results,
    initialize_simulation_display,
    monitor_tasks_until_complete,
)
from logic.src.pipeline.features.test.orchestrator.results_handler import aggregate_final_results
from logic.src.utils.tasks.simulation_utils import (
    prepare_parallel_task_args,
    print_execution_info,
)


def run_parallel_simulations(
    cfg: Config,
    device: Any,
    indices: List[Any],
    sample_idx_ls: List[List[int]],
    weights_path: str,
    lock: Any,
    manager: Any,
    n_cores: int,
    task_count: int,
    log_file: Optional[str],
    original_stderr: Any,
    shared_metrics: Any,
    tracking_uri: Optional[str] = None,
    tracking_run_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Execute simulations in parallel using a multiprocessing pool.

    Args:
        cfg: Root configuration with ``cfg.sim`` containing simulation params.
        device: Torch device.
        indices: Bin subset indices per sample.
        sample_idx_ls: Sample index lists per policy.
        weights_path: Path to model weights.
        lock: Multiprocessing lock.
        manager: Multiprocessing Manager.
        n_cores: Number of cores to use.
        task_count: Total number of tasks.
        log_file: Path to log file for worker redirection.
        original_stderr: Original stderr for shutdown messages.
        shared_metrics: Multiprocessing Manager.dict for real-time stats.
        tracking_uri: Directory of the WSTracker database (passed to workers).
        tracking_run_id: Parent run UUID that workers should attach to.

    Returns:
        Tuple containing:
            - log (Dict[str, Any]): Aggregated simulation results.
            - log_std (Optional[Dict[str, Any]]): Standard deviation of results.
            - failed_log (List[Dict[str, Any]]): List of failed simulation results.
    """
    sim = cfg.sim
    udef.update_lock_wait_time(n_cores)
    counter = mp.Value("i", 0)

    # Prepare task arguments
    args = prepare_parallel_task_args(sim.full_policies, sim.graph.n_samples, indices, sample_idx_ls)

    # Print execution info
    print_execution_info(task_count, n_cores)

    # Create multiprocessing pool
    # Use 'spawn' for CUDA to avoid initialization errors, or 'fork' for CPU if requested
    is_cuda = "cuda" in str(device)
    start_method = "spawn" if is_cuda else ("fork" if getattr(sim, "no_cuda", False) else "spawn")
    mp.set_start_method(start_method, force=True)
    from logic.src.pipeline.simulations.simulator import init_single_sim_worker

    pool = Pool(
        processes=n_cores,
        initializer=init_single_sim_worker,
        initargs=(lock, counter, shared_metrics, log_file, cfg, tracking_uri, tracking_run_id, task_count),
    )

    try:
        log, log_std, failed_log = execute_and_monitor_tasks(
            pool, cfg, device, args, weights_path, n_cores, counter, manager, lock, shared_metrics
        )
        return log, log_std, failed_log

    except KeyboardInterrupt:
        handle_shutdown(pool, original_stderr)
        return {}, None, []  # unreachable but satisfies type checker

    finally:
        cleanup_multiprocessing_pool(pool)


def execute_and_monitor_tasks(
    pool: Pool,
    cfg: Config,
    device: Any,
    args: List[Tuple[Any, ...]],
    weights_path: str,
    n_cores: int,
    counter: Any,
    manager: Any,
    lock: Any,
    shared_metrics: Any,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Execute parallel tasks and monitor their progress.

    Args:
        pool: multiprocessing pool for parallel execution.
        cfg: Root configuration.
        device: Torch device.
        args: Task arguments.
        weights_path: Path to model weights.
        n_cores: Number of cores.
        counter: Task counter.
        manager: Multiprocessing Manager.
        lock: Multiprocessing lock.
        shared_metrics: Multiprocessing Manager.dict for real-time stats.

    Returns:
        Tuple containing:
            - log (Dict[str, Any]): Aggregated simulation results.
            - log_std (Optional[Dict[str, Any]]): Standard deviation of results.
            - failed_log (List[Dict[str, Any]]): List of failed simulation results.
    """
    sim = cfg.sim
    policies = sim.full_policies

    # Resolve all names upfront to avoid disagreement with workers
    from logic.src.pipeline.simulations.day_context import resolve_policy_display_name, to_slug

    resolved_info = [resolve_policy_display_name(p, sim) for p in policies]
    policy_names = []
    for _orig_id, display_name in resolved_info:
        policy_names.append(to_slug(display_name))
    no_pbar = cfg.tracking.no_progress_bar
    display = (
        initialize_simulation_display(policy_names, sim.graph.n_samples, sim.graph.n_days) if not no_pbar else None
    )

    log_tmp = manager.dict()
    failed_log = manager.list()
    for pol_name in policy_names:
        log_tmp[pol_name] = manager.list()

    def _update_result(result: Dict[str, Any]) -> None:
        success = result.pop("success", False)
        sample_id = result.pop("sample_id", None)
        if not (isinstance(result, dict) and success):
            error_policy = result.get("policy", "unknown")
            error_sample = sample_id if sample_id is not None else result.get("sample", "unknown")
            error_msg = result.get("error", "Unknown error")
            print(f"Simulation failed: {error_policy} #{error_sample} - {error_msg}")
            result["policy"] = error_policy
            result["sample"] = error_sample
            result["sample_id"] = error_sample
            result["error"] = error_msg
            failed_log.append(result)
            return

        # Get the policy name from the result dict (excluding success/sample_id/etc)
        pol_name = next((k for k in result.keys() if k not in ["success", "sample_id"]), "unknown")

        # Clear from shared_metrics to avoid double counting, NOW safely after we have the final result
        if shared_metrics is not None and sample_id is not None:
            key = f"{pol_name}_{sample_id}"
            if key in shared_metrics:
                del shared_metrics[key]

        log_tmp[pol_name].append(list(result.values())[0])

    from logic.src.pipeline.simulations.simulator import single_simulation

    tasks = []
    for arg_tup in args:
        task = pool.apply_async(
            single_simulation,
            args=(cfg, device, *arg_tup, weights_path, n_cores),
            callback=_update_result,
        )
        tasks.append(task)

    monitor_tasks_until_complete(tasks, display, policy_names, shared_metrics, counter, log_tmp)

    if display:
        display.stop()

    collect_all_task_results(tasks)

    log, log_std = aggregate_final_results(log_tmp, cfg, lock)
    return log, log_std, failed_log


def handle_shutdown(pool: Pool, original_stderr: Any) -> None:
    """
    Handle graceful shutdown on keyboard interrupt.

    Args:
        pool: Multiprocessing pool.
        original_stderr: Original stderr.
    """
    original_stderr.write("\n\n[WARNING] Caught CTRL+C. Forcing immediate shutdown...\n")
    original_stderr.flush()

    try:
        pool.terminate()
        pool.join()
    except Exception:
        pass

    os._exit(1)


def cleanup_multiprocessing_pool(pool: Pool) -> None:
    """
    Safely close and join the multiprocessing pool.

    Args:
        pool: Multiprocessing pool.
    """
    try:
        pool.close()
        pool.join()
    except Exception:
        pool.terminate()
