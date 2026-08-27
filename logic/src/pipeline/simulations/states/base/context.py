"""
Simulation State Machine Context.

This module provides the SimulationContext class, which manages the lifecycle
of a single simulation run through its various states (Initializing, Running, Finishing).

Attributes:
    SimulationContext: The main context object for simulation state management.

Example:
    >>> from logic.src.pipeline.simulations.states.base.context import SimulationContext
    >>> # context = SimulationContext(cfg, device, indices, sample_id, pol_id, weights, vars)
    >>> # result = context.run()
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from logic.src.constants import (
    ROOT_DIR,
)
from logic.src.pipeline.simulations.states.base.base import SimState
from logic.src.pipeline.simulations.states.initializing import InitializingState
from logic.src.tracking.logging.pylogger import get_pylogger

if TYPE_CHECKING:
    from logic.src.configs import Config
    from logic.src.pipeline.simulations.checkpoints import SimulationCheckpoint


logger = get_pylogger(__name__)


class SimulationContext:
    """Context for simulation state machine.

    Attributes:
        cfg: Root configuration object (Config or DictConfig).
        device: Torch device for computations.
        indices: Subset of bin indices for this simulation sample.
        sample_id: Unique identifier for the simulation sample/seed.
        pol_id: Index of the policy in the configuration's policy list.
        model_weights_path: Path to pretrained neural model weights, if applicable.
        lock: Optional threading.Lock for synchronized operations.
        counter: Shared counter for progress tracking across samples.
        overall_progress: Progress bar/updater object.
        shared_metrics: Shared dictionary for real-time metric reporting.
        exec_time: Total execution time of the simulation.
        start_time: Start timestamp of the simulation run.
        end_time: End timestamp of the simulation run.
        pol_name: Extracted name of the policy.
        pol_cfg: Policy-specific configuration dictionary.
        current_state: The current SimState object.
        result: Dictionary containing the final simulation results.
        data_dir: Path to the simulator data directory.
        results_dir: Path to the simulation results assets directory.
        callback: Optional callback called after each simulation day: (day, metrics) -> None.
    """

    lock: Optional[threading.Lock]
    counter: Optional[Any]
    overall_progress: Optional[Any]
    shared_metrics: Any = None
    callback: Optional[Callable[[int, Dict[str, Any], int], None]] = None
    exec_time: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    pol_name: str = ""
    display_name: str = ""
    pol_id: int = 0
    pol_cfg: Dict[str, Any] = {}

    @property
    def policy(self) -> str:
        """Alias for pol_name for backward compatibility.

        Returns:
            The extracted policy name.
        """
        return self.pol_name

    def __init__(
        self,
        cfg: Union[Config, DictConfig],
        device: torch.device,
        indices: List[int],
        sample_id: int,
        pol_id: int,
        model_weights_path: Optional[str],
        variables_dict: Dict[str, Any],
    ):
        """Initialize Class.

        Args:
            cfg: Typed Hydra configuration or OmegaConf DictConfig.
            device: Target torch device.
            indices: List of bin indices for this sample.
            sample_id: Sample/seed identifier.
            pol_id: Policy index in cfg.sim.full_policies.
            model_weights_path: Path to pretrained model weights.
            variables_dict: Dictionary containing shared resources (lock, counter, etc.).
        """
        self.cfg = cfg
        self.device = device
        self.indices = indices
        self.sample_id = sample_id
        self.pol_id = pol_id
        self.model_weights_path = model_weights_path
        self.variables_dict = variables_dict
        self.callback = variables_dict.get("callback")

        self.lock = variables_dict.get("lock")
        self.counter = variables_dict.get("counter")
        self.overall_progress = variables_dict.get("overall_progress")
        self.shared_metrics = variables_dict.get("shared_metrics")

        self.current_state: Optional[SimState] = None
        self.result: Optional[Dict[str, Any]] = None

        sim = cfg.sim
        self.data_dir = os.path.join(ROOT_DIR, "data", "wsr_simulator")
        self.results_dir = os.path.join(
            ROOT_DIR,
            "assets",
            sim.output_dir,
            f"{sim.graph.n_days}days",
            f"{sim.graph.area}{sim.graph.num_loc}_{sim.graph.waste_type}",
            sim.data_distribution,
            sim.run_name or "",
        )
        os.makedirs(self.results_dir, exist_ok=True)
        policies = sim.full_policies
        raw_policy = policies[pol_id]

        # Use robust utility to extract policy name and ensure config is a plain dict
        from logic.src.utils.infrastructure.setup_sims import deep_sanitize

        self.pol_cfg = deep_sanitize(raw_policy)

        # Use centralized utility for consistent name resolution across processes
        from logic.src.pipeline.simulations.day_context import resolve_policy_display_name, to_slug

        self.pol_id_orig, self.display_name = resolve_policy_display_name(raw_policy, sim)
        self.pol_name = to_slug(self.display_name)

        self._continue_init(variables_dict, pol_id)

    def _continue_init(self, variables_dict: Dict[str, Any], pol_id: int) -> None:
        """continue init.

        Args:
            variables_dict: Dictionary containing shared resources.
            pol_id: Policy index.
        """
        self.start_day: int = 1
        self.checkpoint: Optional[SimulationCheckpoint] = None
        self.bins: Optional[Any] = None
        self.new_data: Optional[pd.DataFrame] = None
        self.coords: Optional[pd.DataFrame] = None
        self.dist_tup: Optional[Tuple[np.ndarray, Any, Any, Any]] = None
        self.model_tup: Optional[Tuple[Any, ...]] = None
        self.model_env: Optional[Any] = None
        self.hrl_manager: Optional[Any] = None
        self.cached: Optional[List[Any]] = None
        self.run_time: float = 0
        self.overflows: int = 0
        self.current_collection_day: int = 0
        self.daily_log: Optional[Dict[str, List[Any]]] = None
        self.attention_dict: Dict[str, List[Any]] = {}
        self.execution_time: float = 0
        self.tic: float = 0
        self.config: Optional[Dict[str, Any]] = None
        self.vehicle_capacity: Optional[float] = None
        self.shift_hours: float = 7.0
        self.avg_speed_kmh: float = 35.0
        self.service_time_h: float = 1.5 / 60.0
        self.transition_to(InitializingState())

    def transition_to(self, state: Optional[SimState]) -> None:
        """Transition to.

        Args:
            state: The SimState object to transition to.
        """
        self.current_state = state
        if self.current_state is not None:
            self.current_state.context = self

    def run(self) -> Optional[Dict[str, Any]]:
        """Run.

        Returns:
            The final result dictionary from the simulation.
        """
        while self.current_state is not None:
            self.current_state.handle(self)
        return self.result

    def get_current_state_tuple(self) -> Tuple[Any, ...]:
        """Get current state tuple.

        Returns:
            A tuple of core simulation state components.
        """
        return (
            self.new_data,
            self.coords,
            self.dist_tup,
            None,
            self.bins,
            self.model_tup,
            self.cached,
            self.overflows,
            self.current_collection_day,
            self.daily_log,
            self.execution_time,
        )
