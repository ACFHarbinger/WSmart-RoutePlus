"""
Simulation Orchestrator.
Refactored into modular components.

Attributes:
    simulator_testing: Main entry point for orchestrating parallel simulation runs.

Example:
    >>> from logic.src.pipeline.features.test.orchestrator import simulator_testing
    >>> simulator_testing(cfg, data_size, device)
    Traceback (most recent call last):
        ...  # doctest: +ELLIPSIS
    SystemExit: 1
"""

import contextlib
import copy
import multiprocessing as mp
import os
import shutil
import signal
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import OmegaConf

import logic.src.constants as udef
import logic.src.tracking as wst
from logic.src.constants.paths import CONFIGS_DIR
from logic.src.configs import Config
from logic.src.pipeline.callbacks import PolicySummaryCallback

# Local imports from modular components
from logic.src.pipeline.features.test.orchestrator.parallel_runner import run_parallel_simulations
from logic.src.pipeline.simulations.repository import load_indices
from logic.src.pipeline.simulations.simulator import (
    display_log_metrics,
    sequential_simulations,
)
from logic.src.tracking.logging.log_utils import (
    output_stats,
    runs_per_policy,
    send_final_output_to_gui,
)
from logic.src.tracking.logging.logger_writer import LoggerWriter, setup_logger_redirection

__all__ = [
    "simulator_testing",
    "LoggerWriter",
    "setup_logger_redirection",
    "runs_per_policy",
    "output_stats",
    "send_final_output_to_gui",
    "display_log_metrics",
    "sequential_simulations",
    "load_indices",
    "run_parallel_simulations",
    "Config",
    "udef",
    "wst",
    "PolicySummaryCallback",
    "run_parallel_simulations",
    "load_indices",
    "display_log_metrics",
    "sequential_simulations",
    "output_stats",
    "runs_per_policy",
    "send_final_output_to_gui",
    "LoggerWriter",
    "setup_logger_redirection",
    "logger",
]


def _expand_other_ref(ref_dict: dict, root_dir: str) -> dict:
    """Expand a {yaml_file: [key1, key2]} reference by loading the YAML file."""
    import yaml

    result = {}
    for yaml_path, keys in ref_dict.items():
        full_path = os.path.join(CONFIGS_DIR, "policies", yaml_path)
        if not os.path.exists(full_path):
            return ref_dict  # Keep original if file not found
        with open(full_path) as f:
            yaml_data = yaml.safe_load(f) or {}
        for key in keys or []:
            if key in yaml_data:
                result[key] = yaml_data[key]
    return result


def _expand_refs_recursive(obj: Any, root_dir: str) -> Any:
    """Recursively expand mandatory_selection and route_improvement YAML references."""
    _expandable = ("mandatory_selection", "route_improvement")
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k in _expandable and isinstance(v, dict):
                result[k] = _expand_other_ref(v, root_dir)
            else:
                result[k] = _expand_refs_recursive(v, root_dir)
        return result
    elif isinstance(obj, list):
        new_list = []
        for item in obj:
            if isinstance(item, dict) and len(item) == 1:
                key = next(iter(item))
                val = item[key]
                if key in _expandable and isinstance(val, dict):
                    new_list.append({key: _expand_other_ref(val, root_dir)})
                else:
                    new_list.append({key: _expand_refs_recursive(val, root_dir)})
            elif isinstance(item, dict):
                new_list.append(_expand_refs_recursive(item, root_dir))
            else:
                new_list.append(item)
        return new_list
    else:
        return obj


def _generate_pruned_config(cfg: Config, root_dir: str) -> str:
    """Generate a pruned YAML config with only task-relevant sections and expanded policy refs."""
    import yaml

    full = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)  # type: ignore[arg-type]
    sim = full.get("sim", {})  # pyrefly: ignore [missing-attribute]
    active_policies: List[str] = sim.get("policies", []) or []

    pruned: dict = {}
    for key in ("task", "seed", "device", "start", "run_name"):
        # pyrefly: ignore [not-iterable]
        if key in full:
            # pyrefly: ignore [bad-index, unsupported-operation]
            pruned[key] = full[key]
    for section in ("sim", "tracking"):
        # pyrefly: ignore [not-iterable]
        if section in full:
            # pyrefly: ignore [bad-index, unsupported-operation]
            pruned[section] = copy.deepcopy(full[section])

    # pyrefly: ignore [missing-attribute]
    p_full = full.get("p", {})
    p_pruned = {}
    for pol_name in active_policies:
        if pol_name in p_full:
            p_pruned[pol_name] = _expand_refs_recursive(copy.deepcopy(p_full[pol_name]), root_dir)
    pruned["p"] = p_pruned

    repo_path = os.path.join(os.path.expanduser("~"), "Repositories", "WSmart-Route")
    yaml_str = yaml.dump(pruned, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return yaml_str.replace(repo_path, "~/Repositories/WSmart-Route")


def simulator_testing(cfg: Config, data_size: int, device: Any) -> None:  # noqa: C901
    """
    Orchestrates the parallel execution of multiple simulation runs.

    Args:
        cfg: Root configuration with ``cfg.sim`` containing simulation params.
        data_size: Number of available data points for the area.
        device: Torch device for neural models.
    """
    from logic.src.pipeline.simulations.day_context import policy_result_key

    sim = cfg.sim
    log_file = setup_logger_redirection(echo_to_terminal=True)
    if OmegaConf.is_config(cfg):
        OmegaConf.set_struct(cfg, False)  # type: ignore[arg-type]
        cfg.tracking.log_file = log_file
        if not cfg.sim.run_name:
            cfg.sim.run_name = f"run{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        OmegaConf.set_struct(cfg, True)  # type: ignore[arg-type]
    else:
        cfg.tracking.log_file = log_file
        if not cfg.sim.run_name:
            cfg.sim.run_name = f"run{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sim = cfg.sim

    # Capture original stderr for shutdown messages if redirected
    original_stderr = sys.stderr
    if isinstance(original_stderr, LoggerWriter):
        original_stderr = original_stderr.terminal

    # Display policy summary
    PolicySummaryCallback().display(cfg)

    # Register immediate shutdown handler for CTRL+C
    def _shutdown_handler(sig: int, frame: Any) -> None:
        original_stderr.write("\n\n[WARNING] Caught CTRL+C (SIGINT). Forcing immediate shutdown...\n")
        original_stderr.flush()
        try:
            os.killpg(os.getpgrp(), signal.SIGKILL)
        except OSError:
            os._exit(1)

    signal.signal(signal.SIGINT, _shutdown_handler)

    manager = mp.Manager()
    try:
        lock = manager.Lock()
        shared_metrics = manager.dict()

        raw_policies = sim.full_policies
        # Resume logs and workers are keyed by the display slug (ctx.pol_name),
        # not the raw expander id. Both the initial sample lists and the resume
        # filter use that same key, in policy order.
        policies = [policy_result_key(p, sim) for p in raw_policies]
        sample_idx_dict: Dict[str, List[int]] = {pol: list(range(sim.graph.n_samples)) for pol in policies}

        # ── Config snapshot ─────────────────────────────────────────────────────
        # Persist the fully-resolved Hydra config (including all CLI overrides)
        # into the results directory so every run is self-contained.
        # pyrefly: ignore [no-matching-overload]
        _snapshot_dir = os.path.join(
            udef.ROOT_DIR,
            "assets",
            sim.output_dir,
            f"{sim.graph.n_days}days",
            f"{sim.graph.area}{sim.graph.num_loc}_{sim.graph.waste_type}"
            if sim.graph.waste_type
            else f"{sim.graph.area}{sim.graph.num_loc}",
            sim.data_distribution,
            sim.run_name,
        )
        try:
            os.makedirs(_snapshot_dir, exist_ok=True)

            target_hydra_dir = os.path.join(_snapshot_dir, "hydra")
            os.makedirs(target_hydra_dir, exist_ok=True)

            try:
                hydra_out_dir = HydraConfig.get().runtime.output_dir
                source_hydra_dir = os.path.join(hydra_out_dir, ".hydra")
            except Exception:
                source_hydra_dir = os.path.join(os.getcwd(), ".hydra")

            if os.path.exists(source_hydra_dir):
                repo_path = os.path.join(os.path.expanduser("~"), "Repositories", "WSmart-Route")
                for fname in ["config.yaml", "hydra.yaml", "overrides.yaml"]:
                    src_file = os.path.join(source_hydra_dir, fname)
                    if os.path.exists(src_file):
                        with open(src_file, "r") as f_in:
                            content = f_in.read()
                        relative_content = content.replace(repo_path, "~/Repositories/WSmart-Route")
                        with open(os.path.join(target_hydra_dir, fname), "w") as f_out:
                            f_out.write(relative_content)
                logger.info(f"Hydra config snapshot saved → {target_hydra_dir}")
            else:
                logger.warning(f"Could not find source .hydra directory at {source_hydra_dir}")

            pruned_path = os.path.join(target_hydra_dir, "pruned_config.yaml")
            try:
                pruned_yaml = _generate_pruned_config(cfg, str(udef.ROOT_DIR))
                with open(pruned_path, "w") as _f:
                    _f.write(pruned_yaml)
                logger.info(f"Pruned config saved → {pruned_path}")
            except Exception as _pruned_err:
                logger.warning(f"Could not save pruned_config.yaml: {_pruned_err}")
        except Exception as _snap_err:
            logger.warning(f"Could not save hydra config snapshot: {_snap_err}")
        # ────────────────────────────────────────────────────────────────────────
        if sim.resume:
            runs_per_policy_any: Any = runs_per_policy
            to_remove = runs_per_policy_any(
                home_dir=str(udef.ROOT_DIR),
                ndays=sim.graph.n_days,
                nbins=[sim.graph.num_loc],
                output_dir=sim.output_dir,
                area=sim.graph.area,
                waste_type=sim.graph.waste_type,
                data_distribution=sim.data_distribution,
                run_name=getattr(sim, "run_name", "") or "",
                nsamples=[sim.graph.n_samples],
                policies=policies,
                lock=lock,
            )[0]
            for pol in policies:
                if len(to_remove[pol]) > 0:
                    sample_idx_dict[pol] = [x for x in sample_idx_dict[pol] if x not in to_remove[pol]]

        sample_idx_ls = [list(val) for val in sample_idx_dict.values()]
        task_count = sum(len(sample_idx) for sample_idx in sample_idx_ls)

        n_cores = sim.cpu_cores
        n_cores = min(task_count, n_cores) if n_cores >= 1 else min(task_count, max(1, mp.cpu_count() - 1))
        if data_size != sim.graph.num_loc:
            indices = load_indices(sim.graph.focus_graph, sim.graph.n_samples, sim.graph.num_loc, data_size, lock)
            if len(indices) == 1:
                indices = [indices[0]] * sim.graph.n_samples
        else:
            indices: List[Optional[Any]] = [None] * sim.graph.n_samples  # type: ignore[no-redef]

        weights_path = os.path.join(udef.ROOT_DIR, "assets", "model_weights")

        # Extract active tracking context so parallel workers can attach to the
        # parent run.  Sequential workers run in the same process and already
        # inherit get_active_run() directly.
        _active_run = wst.get_active_run()
        _tracker = wst.get_tracker()
        tracking_uri: Optional[str] = _tracker.tracking_uri if _tracker is not None else None
        tracking_run_id: Optional[str] = _active_run.run_id if _active_run is not None else None

        if n_cores > 1:
            log, log_std, _failed_log = run_parallel_simulations(
                cfg,
                device,
                indices,
                sample_idx_ls,
                weights_path,
                lock,
                manager,
                n_cores,
                task_count,
                log_file,
                original_stderr,
                shared_metrics,
                tracking_uri=tracking_uri,
                tracking_run_id=tracking_run_id,
            )
        else:
            log, log_std, _failed_log = sequential_simulations(
                cfg, device, indices, sample_idx_ls, weights_path, lock, shared_metrics, task_count
            )

        # pyrefly: ignore [no-matching-overload]
        realtime_log_path = os.path.join(
            udef.ROOT_DIR,
            "assets",
            sim.output_dir,
            f"{sim.graph.n_days}days",
            f"{sim.graph.area}{sim.graph.num_loc}_{sim.graph.waste_type}",
            sim.data_distribution,
            sim.run_name,
            f"log_realtime_{sim.data_distribution}_{sim.graph.n_samples}N.jsonl",
        )
        send_final_output_to_gui(log, log_std, sim.graph.n_samples, policies, realtime_log_path)

        display_log_metrics(
            sim.output_dir,
            sim.graph.num_loc,
            sim.graph.n_samples,
            sim.graph.n_days,
            sim.graph.area,
            policies,
            log,
            log_std,
            lock,
            waste_type=sim.graph.waste_type,
            data_distribution=sim.data_distribution,
            run_name=sim.run_name,
        )
        failed_records = [dict(row) for row in _failed_log]
        if failed_records:
            # SystemExit is not an Exception, so the engine does not rewrite it
            # into a successful return. The process exit code is 1.
            raise SystemExit(f"{len(failed_records)} simulation sample(s) failed")
    finally:
        with contextlib.suppress(Exception):
            manager.shutdown()
