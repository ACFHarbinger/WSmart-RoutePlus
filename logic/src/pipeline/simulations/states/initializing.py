"""
Simulation Initialization State.

This module provides the InitializingState class, which handles the setup of
data, models, and environments before the simulation loop begins.

Attributes:
    InitializingState: State responsible for simulation setup.

Example:
    >>> # from logic.src.pipeline.simulations.states.initializing import InitializingState
    >>> # state = InitializingState()
    >>> # state.handle(ctx)
"""

from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger

from logic.src.constants.paths import CONFIGS_DIR
from logic.src.constants import DAY_METRICS, ROOT_DIR  # noqa: F401  (ROOT_DIR: test patch point)
from logic.src.data.processor import (
    process_data,
    process_model_data,
    setup_basedata,
    setup_dist_path_tup,
)
from logic.src.pipeline.simulations.actions.base import _flatten_config
from logic.src.pipeline.simulations.bins import Bins
from logic.src.pipeline.simulations.checkpoints import SimulationCheckpoint
from logic.src.pipeline.simulations.repository import load_area_and_waste_type_params, load_temporal_params
from logic.src.pipeline.simulations.states.base.base import SimState
from logic.src.pipeline.simulations.states.running import RunningState
from logic.src.tracking.logging.log_utils import setup_system_logger
from logic.src.tracking.logging.logger_writer import setup_logger_redirection
from logic.src.utils.configs.config_loader import load_config
from logic.src.utils.data.loader import load_grid_base
from logic.src.utils.infrastructure.setup_env import setup_env
from logic.src.utils.infrastructure.setup_manager import setup_hrl_manager
from logic.src.utils.infrastructure.setup_sims import get_graph_config
from logic.src.utils.infrastructure.setup_worker import setup_model

if TYPE_CHECKING:
    from logic.src.pipeline.simulations.states.base.base import SimulationContext



def routed_bin_ids(coords: Any) -> List[Any]:
    """Bin IDs of the routed graph in node order (row 0 of ``coords`` is the depot)."""
    return coords["ID"].iloc[1:].tolist()

class InitializingState(SimState):
    """State handles the initialization of simulation data (graph, models, etc.).

    Attributes:
        None
    """

    def handle(self, ctx: SimulationContext) -> None:
        """Handle initialization of simulation state.

        Args:
            ctx: The simulation context object.
        """
        # Seeding for reproducibility (covers both sequential and parallel runs)
        seed = ctx.cfg.sim.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        sim = ctx.cfg.sim
        graph = get_graph_config(sim)

        self._setup_logging_and_dirs(ctx)
        self._load_all_configurations(ctx)

        # Load base data
        data, bins_coordinates, depot = setup_basedata(graph.num_loc, ctx.data_dir, graph.area, graph.waste_type)
        self._setup_capacities(ctx)

        # Checkpoints
        ctx.checkpoint = SimulationCheckpoint(ctx.results_dir, sim.checkpoint_dir, ctx.pol_id_orig, ctx.sample_id)
        saved_state, last_day = self._load_checkpoint_if_needed(ctx)

        # Setup Models
        self._setup_models(ctx)

        # Restore or Initialize State
        if sim.resume and saved_state is not None:
            self._restore_state(ctx, saved_state, last_day)
        else:
            self._initialize_new_state(ctx, data, bins_coordinates, depot)

        # Rebuild on both new and resumed runs, using the restored node order.
        if sim.problem.lower() == "ctop" or getattr(graph, "tm_filepath", None):
            assert ctx.dist_tup is not None
            ctx.shift_hours, ctx.time_matrix, ctx.service_time_h = load_temporal_params(
                coords=ctx.coords,
                distance_matrix=ctx.dist_tup[0],
                tm_filepath=getattr(graph, "tm_filepath", None),
                shift_hours=ctx.shift_hours,
                avg_speed_kmh=ctx.avg_speed_kmh,
                service_time_h=ctx.service_time_h,
                time_unit=getattr(graph, "time_matrix_unit", "seconds"),
            )

        logger.info(f"Initialization complete. Transitioning to RunningState for {ctx.pol_name} policy.")
        ctx.transition_to(RunningState())

    def _setup_logging_and_dirs(self, ctx: SimulationContext) -> None:
        """Configure logging and create output directories.

        Args:
            ctx: The simulation context object.

        Returns:
            None
        """

        def _safe_get(obj: Any, path: str, default: Any = None) -> Any:
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                return obj
            except Exception:
                return default

        log_file = _safe_get(ctx.cfg, "tracking.log_file")
        if log_file is None:
            log_dir = _safe_get(ctx.cfg, "tracking.log_dir", "logs")
            log_file = Path(log_dir) / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

        log_level = _safe_get(ctx.cfg, "tracking.log_level", "INFO")
        setup_system_logger(log_file, log_level)

        # Redirect stderr to the simulation log file for the main process
        setup_logger_redirection(
            log_file=log_file,
            silent=True,
            redirect_stdout=True,  # Redirect stdout to ensure Bins prints etc. reach the log file
            redirect_stderr=True,
            echo_to_terminal=True,  # Still echo to terminal for visibility
        )

        if not os.path.exists(ctx.results_dir):
            os.makedirs(ctx.results_dir)
            logger.info(f"Created results directory: {ctx.results_dir}")
        else:
            logger.info(f"Results directory already exists: {ctx.results_dir}")

    def _load_all_configurations(self, ctx: SimulationContext) -> None:
        """Load all configurations into the simulation context.

        Args:
            ctx: The simulation context object.

        Returns:
            None
        """
        ctx.config = {}

        def _safe_get(obj: Any, path: str, default: Any = None) -> Any:
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                return obj
            except Exception:
                return default

        config_paths = _safe_get(ctx.cfg, "sim.config_path")

        if config_paths and hasattr(config_paths, "items") and hasattr(config_paths, "keys"):
            for key, path in config_paths.items():
                try:
                    loaded = path if hasattr(path, "items") else load_config(path)
                    ctx.config[key] = loaded
                    print(f"\n[INFO] Loaded configuration for '{key}' from {path}")
                except (OSError, ValueError) as e:
                    print(f"\n[Warning] Failed to load config file {path} for {key}: {e}")

        self._load_neural_configs(ctx)

    def _load_neural_configs(self, ctx: SimulationContext) -> None:
        # Priority 1: policy_cfg (new structured format)
        """

        Args:
            ctx: The simulation context object.
        """
        model_name = ""
        if isinstance(ctx.pol_cfg, dict) and "model" in ctx.pol_cfg:
            model_name = ctx.pol_cfg["model"].get("name", "").lower()

        neural_cfg_path = os.path.join(CONFIGS_DIR, "policies", "policy_neural.yaml")
        pol_parts = ctx.pol_name.lower().replace("_", " ").replace("-", " ").split()
        is_neural = any(kw in pol_parts for kw in ["na", "amgat", "am", "ptr", "ddam", "transgcn"]) or any(
            kw in model_name for kw in ["na", "amgat", "am", "ptr", "ddam", "transgcn"]
        )

        # Look for neural keywords in model_name or context's pol_name
        is_neural_arch = any(kw in model_name for kw in ["am", "ptr", "transgcn", "ddam"]) or any(
            kw in ctx.pol_name.lower().split("_") for kw in ["na", "am", "ptr", "transgcn", "ddam"]
        )

        if is_neural and (is_neural_arch or model_name == "") and os.path.exists(neural_cfg_path):
            try:
                neural_cfg = load_config(neural_cfg_path)
                if neural_cfg and ctx.config is not None:
                    if "na" in neural_cfg:
                        # Flatten specific neural sub-configs if they are lists
                        for pol_key, pol_val in neural_cfg["na"].items():
                            ctx.config[pol_key] = _flatten_config(pol_val)
                    else:
                        ctx.config.update(neural_cfg)
                print(f"\n[INFO] Loaded configuration from {neural_cfg_path}")
            except (OSError, ValueError) as e:
                print(f"\n[WARNING] Failed to load neural config {neural_cfg_path}: {e}")

    def _setup_capacities(self, ctx: SimulationContext) -> None:
        """Sets up vehicle capacity and temporal parameters.

        Args:
            ctx: The simulation context object.
        """
        sim = ctx.cfg.sim
        from logic.src.utils.infrastructure.setup_sims import get_graph_config

        graph = get_graph_config(sim)
        capacities, _, _, _, _ = load_area_and_waste_type_params(graph.area, graph.waste_type)
        ctx.vehicle_capacity = capacities

        shift_hours, avg_speed_kmh, service_time_h = load_temporal_params(
            shift_hours=getattr(sim, "shift_hours", 7.0),
            avg_speed_kmh=getattr(sim, "avg_speed_kmh", 35.0),
            service_time_h=getattr(sim, "service_time_h", 1.5 / 60.0),
        )
        ctx.shift_hours = shift_hours
        ctx.avg_speed_kmh = avg_speed_kmh
        ctx.service_time_h = service_time_h

    def _load_checkpoint_if_needed(self, ctx: SimulationContext) -> Tuple[Optional[Any], int]:
        """

        Args:
            ctx: The simulation context object.

        Returns:
            Tuple of (saved_state, last_day_indexed).
        """
        saved_state, last_day = (None, 0)
        if ctx.cfg.sim.resume and ctx.checkpoint is not None:
            saved_state, last_day = ctx.checkpoint.load_state()
            if saved_state is not None and ctx.overall_progress:
                ctx.overall_progress.update(last_day)
        return saved_state, last_day

    def _setup_models(self, ctx: SimulationContext) -> None:
        """

        Args:
            ctx: The simulation context object.
        """
        # If policy config was not correctly loaded in __init__ (common when passing paths in config_path)
        # we re-link it here from the correctly loaded context config registry.
        if (not ctx.pol_cfg or not isinstance(ctx.pol_cfg, dict)) and ctx.pol_id_orig in ctx.config:  # pyrefly: ignore [not-iterable]
            ctx.pol_cfg = ctx.config[ctx.pol_id_orig]  # pyrefly: ignore [unsupported-operation]

        model_name = ""
        if isinstance(ctx.pol_cfg, dict) and "model" in ctx.pol_cfg:
            model_name = ctx.pol_cfg["model"].get("name", "").lower()

        sim = ctx.cfg.sim
        pol_parts = ctx.pol_name.lower().replace("_", " ").replace("-", " ").split()
        is_neural = any(kw in pol_parts for kw in ["na", "amgat", "am", "ptr", "ddam", "transgcn"]) or any(
            kw in model_name for kw in ["na", "amgat", "am", "ptr", "ddam", "transgcn"]
        )
        should_load_neural = is_neural or any(model_name.startswith(kw) for kw in ["am", "ddam", "ptr"])

        configs = None
        # Architecture check: also include common neural architecture keywords
        is_neural_arch = (
            any(kw in model_name for kw in ["am", "ptr", "transgcn", "ddam"])
            or any(kw in pol_parts for kw in ["na", "amgat", "am", "ptr", "ddam"])
            or "amgat" in ctx.pol_name.lower()
        )

        if should_load_neural and (is_neural_arch or model_name == ""):
            # Flatten policy config in case it's a list-based structured config
            flat_pol_cfg = _flatten_config(ctx.pol_cfg)

            # Extract decoding parameters from policy config
            decoding = flat_pol_cfg.get("decoding") or {}
            temp = decoding.get("temperature", 1.0)
            strat = str(decoding.get("strategy", "greedy"))

            model_path_raw = flat_pol_cfg.get("model_path")
            if isinstance(model_path_raw, dict):
                model_paths = model_path_raw
            elif isinstance(model_path_raw, str):
                # If it's a single string, we use it for all policy variations by default
                model_paths = {ctx.pol_name: model_path_raw}
            else:
                model_paths = {}

            ctx.model_env, configs = setup_model(
                ctx.pol_name,
                ctx.model_weights_path or "",
                model_paths,
                ctx.device,
                ctx.lock,  # type: ignore[arg-type]
                temp,
                strat,
            )
            ctx.hrl_manager = setup_hrl_manager(
                ctx.cfg.sim,
                ctx.device,
                configs,
                policy=ctx.pol_name,
                base_path=ctx.model_weights_path,
                worker_model=ctx.model_env,
            )
            self.configs = configs
        elif "swc" in model_name:
            ctx.model_env = setup_env(
                ctx.pol_name,
                sim.server_run,
                sim.gplic_file,
                sim.symkey_name,
                sim.env_file,
            )
            ctx.model_tup = (None, None)
            self.configs = None  # type: ignore[assignment]
        else:
            ctx.model_tup = (None, None)
            self.configs = None  # type: ignore[assignment]

    def _restore_state(self, ctx: SimulationContext, saved_state: Any, last_day: int) -> None:
        """
        Args:
            ctx: The simulation context object.
            saved_state: The state object loaded from the checkpoint.
            last_day: The index of the last completed day.
        """
        (
            ctx.new_data,
            ctx.coords,
            ctx.dist_tup,
            adj_matrix,
            ctx.bins,
            ctx.model_tup,
            ctx.cached,
            ctx.overflows,
            ctx.current_collection_day,
            ctx.daily_log,
            ctx.run_time,
        ) = saved_state
        ctx.start_day = last_day + 1

    def _initialize_new_state(self, ctx: SimulationContext, data: Any, bins_coordinates: Any, depot: Any) -> None:
        """
        Args:
            ctx: The simulation context object.
            data: Raw graph/bin data.
            bins_coordinates: DataFrame of bin coordinates.
            depot: Depot location information.
        """
        sim = ctx.cfg.sim
        from logic.src.utils.infrastructure.setup_sims import get_graph_config

        graph = get_graph_config(sim)

        ctx.new_data, ctx.coords = process_data(data, bins_coordinates, depot, ctx.indices)

        _save_dm = getattr(graph, "save_updated_dm", None)
        if _save_dm and ctx.results_dir:
            _save_dm = os.path.join(ctx.results_dir, _save_dm)

        ctx.dist_tup, adj_matrix = setup_dist_path_tup(
            ctx.coords,
            graph.num_loc,
            graph.distance_method,
            graph.dm_filepath,
            sim.env_file,
            sim.gapik_file,
            sim.symkey_name,
            graph.edge_threshold,
            graph.edge_method,
            graph.area,
            graph.waste_type,
            graph.n_days,
            sim.data_distribution,
            ctx.indices,
            save_updated_dm=_save_dm,
        )

        model_name = ""
        if isinstance(ctx.pol_cfg, dict) and "model" in ctx.pol_cfg:
            model_name = ctx.pol_cfg["model"].get("name", "").lower()

        pol_parts = ctx.pol_name.lower().replace("_", " ").replace("-", " ").split()
        is_neural = any(kw in pol_parts for kw in ["na", "amgat", "am", "ptr", "ddam", "transgcn"]) or any(
            kw in model_name for kw in ["na", "amgat", "am", "ptr", "ddam", "transgcn"]
        )
        # Look for neural keywords in model_name or context's pol_name
        is_neural_arch = any(kw in model_name for kw in ["am", "ptr", "transgcn", "ddam"]) or any(
            kw in ctx.pol_name.lower().split("_") for kw in ["na", "am", "ptr", "transgcn", "ddam"]
        )

        if is_neural and (is_neural_arch or model_name == ""):
            ctx.model_tup = process_model_data(
                ctx.coords,
                ctx.dist_tup[2],
                ctx.device,
                graph.vertex_method,
                ctx.config,
                graph.edge_threshold,
                graph.edge_method,
                graph.area,
                graph.waste_type,
                adj_matrix,
            )

        self._initialize_bins(ctx)

        ctx.cached = [] if sim.cache_regular else None
        if ctx.bins is not None:
            if sim.stats_filepath is not None:
                ctx.bins.set_statistics(sim.stats_filepath)
            if ctx.bins.waste_dataset is not None:
                # File-backed stats-file samples are checked here, before day 1.
                # Generated samples already draw n_days + 1 rows (see _initialize_bins).
                sample_horizon = graph.n_days if sim.stats_filepath is not None else None
                ctx.bins.set_sample_waste(ctx.sample_id, horizon=sample_horizon)

            ctx.bins.set_indices(ctx.indices)
        ctx.daily_log = {key: [] for key in DAY_METRICS}

    def _initialize_bins(self, ctx: SimulationContext) -> None:
        """

        Args:
            ctx: The simulation context object.
        """
        sim = ctx.cfg.sim
        from logic.src.utils.infrastructure.setup_sims import get_graph_config

        graph = get_graph_config(sim)
        data_dist = sim.data_distribution
        # With a stats file, row 0 of each waste sample is the opening stock and day d deposits row d
        # (Bins.load_filling), so an n-day horizon needs n + 1 rows.
        n_rows = graph.n_days + (1 if getattr(sim, "stats_filepath", None) is not None else 0)
        if "gamma" in data_dist:
            ctx.bins = Bins(
                graph.num_loc,
                ctx.data_dir,
                data_dist[:-1],
                area=graph.area,
                waste_type=graph.waste_type,
                waste_file=getattr(graph, "load_dataset", None),
                noise_mean=sim.noise_mean,
                noise_variance=sim.noise_variance,
                n_days=n_rows,
                n_samples=graph.n_samples,
                seed=ctx.cfg.sim.seed + ctx.sample_id,
            )
            # Try to get gamma option from config (e.g., sim.data_distribution="gamma1" -> alpha=1)
            try:
                gamma_option = int(data_dist[-1]) - 1
            except (ValueError, IndexError):
                gamma_option = 0
            ctx.bins.set_gamma_distribution(option=gamma_option)  # type: ignore[attr-defined]
        else:
            # Empirical fills must describe the routed bins: build the grid from the IDs
            # of the processed coordinates (node order), not from out_info row positions.
            grid = None
            if data_dist == "emp" and ctx.coords is not None and "ID" in getattr(ctx.coords, "columns", []):
                grid = load_grid_base(None, graph.area, ctx.data_dir, ids=routed_bin_ids(ctx.coords))
            ctx.bins = Bins(
                graph.num_loc,
                ctx.data_dir,
                data_dist,
                grid=grid,
                area=graph.area,
                waste_type=graph.waste_type,
                waste_file=getattr(graph, "load_dataset", None),
                noise_mean=sim.noise_mean,
                noise_variance=sim.noise_variance,
                n_days=n_rows,
                n_samples=graph.n_samples,
                seed=ctx.cfg.sim.seed + ctx.sample_id,
            )
