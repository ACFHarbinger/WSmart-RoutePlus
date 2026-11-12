"""
Simulation state encapsulation and consolidation.

This module provides the SimulationDayContext dataclass and orchestrates
the execution of a single simulation day using the Command Pattern.

Attributes:
    SimulationDayContext: Represents the context of a single simulation day.

Example:
    >>> from logic.src.pipeline.simulations.daily import SimulationDayContext
    >>> context = SimulationDayContext(day=1)
    >>> context.day
    1
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from multiprocessing.synchronize import Lock
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

if TYPE_CHECKING:
    from logic.src.pipeline.simulations.bins import Bins

import numpy as np
import pandas as pd

from logic.src.pipeline.simulations.naming import get_canonical_policy_name
from logic.src.pipeline.simulations.reporting import current_solver_status, reset_solver_status


@dataclass
class SimulationDayContext:
    """
    Context object encapsulating the state of a simulation day.

    Attributes:
        full_policy: Full string identifier of the policy (e.g., 'policy_regular3_gamma1').
        policy_name: Name of the policy.
        bins: The Bins object managing waste levels.
        new_data: DataFrame containing new data for the day.
        coords: DataFrame containing node coordinates.
        distance_matrix: Matrix of distances between nodes.
        distancesC: Integer distance matrix (numpy).
        paths_between_states: Precomputed paths.
        dm_tensor: Tensor version of distance matrix.
        sample_id: ID of the current sample.
        overflows: Current count of overflows.
        day: Current day index.
        model_env: The model environment object.
        model_ls: Tuple of model components.
        n_vehicles: Number of vehicles.
        area: Area identifier.
        realtime_log_path: Path for real-time logging.
        waste_type: Type of waste.
        current_collection_day: Index of the current collection day.
        cached: Cached data.
        device: Torch device.
        lock: Multiprocessing lock.
        hrl_manager: Manager for Hierarchical RL.
        config: Configuration dictionary.
        cost_weight: Weight for length.
        waste_weight: Weight for waste.
        overflow_penalty: Weight for overflows.
        engine: The policy engine type (e.g., 'neural', 'classical').
        threshold: Decision threshold for bin selection.
        seed: Base seed for the simulation.
        search_seed: Offset added to the routing seed (sim.search_seed); the deposits are unchanged.
        policy_seed: Policy-specific seed for RNG isolation.
        display_name: Descriptive name for logging (e.g., 'Regular + ALNS + None').

        # Mutable attributes added during run_day
        daily_log: Dictionary for daily logs.
        tour: The generated tour.
        cost: The cost of the route.
        profit: The profit from collection.
        collected: IDs of collected bins.
        total_collected: Total amount collected.
        ncol: Number of collections.
        new_overflows: Number of new overflows today.
        sum_lost: Amount of waste lost.
        fill: Fill levels.
        total_fill: Total fill levels.
        extra_output: Any extra output from policy.
        mandatory: List of mandatory bins.
        time: Elapsed time for policy execution.
    """

    # Required/Core Fields
    full_policy: str = ""
    policy_name: str = ""
    bins: Optional[Bins] = None
    new_data: Optional[pd.DataFrame] = None
    coords: Optional[pd.DataFrame] = None
    distance_matrix: Optional[Union[np.ndarray, List[List[float]]]] = None
    distancesC: Optional[np.ndarray] = None
    paths_between_states: Optional[Dict[Tuple[int, int], List[int]]] = None
    dm_tensor: Any = None
    sample_id: int = 0
    overflows: int = 0
    day: int = 0
    model_env: Any = None
    model_ls: Tuple[Any, ...] = (None, None, None)
    n_vehicles: int = 1
    area: str = ""
    realtime_log_path: Optional[str] = None
    waste_type: str = ""
    current_collection_day: int = 0
    cached: Optional[List[int]] = None
    device: Any = None
    lock: Optional[Lock] = None
    hrl_manager: Any = None
    config: Optional[Dict[str, Any]] = None
    cost_weight: float = 1.0
    waste_weight: float = 1.0
    overflow_penalty: float = 1.0
    engine: Optional[str] = None
    threshold: Optional[float] = None
    seed: int = 42
    search_seed: int = 0  # sim.search_seed: shifts only the routing search, not the deposits
    policy_seed: Optional[int] = None  # Policy-specific seed for RNG isolation
    display_name: str = ""
    problem: str = "ptp"
    shift_hours: float = 7.0
    time_matrix: Optional[np.ndarray] = None
    avg_speed_kmh: float = 35.0
    service_time_h: float = 1.5 / 60.0
    vehicle_capacity: float = 100.0
    time_windows: Optional[Any] = None
    routing_allowed: bool = True

    # Optional/Mutable Fields
    daily_log: Optional[Dict[str, Any]] = None
    tour: Optional[List[int]] = None
    cost: float = 0.0
    profit: float = 0.0
    collected: Optional[np.ndarray] = None
    total_collected: float = 0.0
    ncol: int = 0
    new_overflows: int = 0
    sum_lost: float = 0.0
    fill: Optional[np.ndarray] = None
    total_fill: Optional[np.ndarray] = None
    extra_output: Any = None
    mandatory: Optional[List[int]] = None
    time: float = 0.0
    time_spent: float = 0.0
    solver_status: str = ""

    capacity_splits: int = 0
    multi_day_context: Any = None
    scenario_tree: Any = None
    trip_loads_pct: Any = None
    trip_loads_kg: Any = None
    capacity_violations: int = 0
    search_context: Any = None
    truth_generator: Any = None
    metrics_phase: str = "unset"
    day_wall_time: float = 0.0
    day_pipeline: Optional[List[str]] = None
    problem_spec: Any = None

    def as_kwargs(self) -> Dict[str, Any]:
        """Field values for ``adapter.execute``, including attributes set during the day."""
        return dict(self.__dict__)

    def __post_init__(self):
        """Initialize any derived or default state if needed."""
        if self.config is None:
            self.config = {}


def resolve_solver_status(
    published: Optional[str],
    tour: Optional[List[int]],
    error: Optional[BaseException] = None,
) -> str:
    """Choose the status string written for one day.

    A status the solver already published wins. Otherwise an exception is
    named, an empty tour is ``empty_tour``, and a returned route is ``ok``.

    Args:
        published: Status noted by the routing algorithm, if any.
        tour: Tour returned for the day.
        error: Exception that aborted the day, if the solver raised.

    Returns:
        The status token stored on the daily record.
    """
    if published:
        return str(published)
    if error is not None:
        return f"{type(error).__name__}: {error}"
    if not tour or len(tour) <= 2:
        return "empty_tour"
    return "ok"


def record_aborted_day(context: Any, error: BaseException) -> Dict[str, Any]:
    """Write one daily record before a routing algorithm exception leaves the day.

    The sample still fails. The record keeps the mandatory set, the day index,
    and the exception type so an aborted day is not a blank log line.

    Args:
        context: The day context.
        error: The exception that aborted the day.

    Returns:
        The daily record stored on ``context.daily_log``.
    """
    from logic.src.pipeline.simulations.actions.base import bind_context

    context = bind_context(context)
    status = resolve_solver_status(
        getattr(context, "solver_status", None) or current_solver_status(),
        getattr(context, "tour", None),
        error,
    )
    context.solver_status = status
    coordinates = getattr(context, "coords", None)
    if coordinates is None:
        coordinates = pd.DataFrame({"ID": []})
    dlog = get_daily_results(
        total_collected=float(getattr(context, "total_collected", 0.0) or 0.0),
        ncol=int(getattr(context, "ncol", 0) or 0),
        cost=float(getattr(context, "cost", 0.0) or 0.0),
        tour=list(getattr(context, "tour", None) or [0]),
        day=int(getattr(context, "day", 0) or 0),
        new_overflows=int(getattr(context, "new_overflows", 0) or 0),
        sum_lost=float(getattr(context, "sum_lost", 0.0) or 0.0),
        coordinates=coordinates,
        profit=float(getattr(context, "profit", 0.0) or 0.0),
        time=float(getattr(context, "time", 0.0) or 0.0),
        mandatory_nodes=getattr(context, "mandatory", None),
        solver_status=status,
        day_wall_time=float(getattr(context, "day_wall_time", 0.0) or 0.0),
    )
    dlog["status"] = "failed"
    dlog["exception_type"] = type(error).__name__
    context.daily_log = dlog
    return dlog


def get_daily_results(
    total_collected: float,
    ncol: int,
    cost: float,
    tour: List[int],
    day: int,
    new_overflows: int,
    sum_lost: float,
    coordinates: pd.DataFrame,
    profit: float,
    time: float,
    mandatory_nodes: Optional[List[int]] = None,
    time_spent: Optional[float] = None,
    solver_status: Optional[str] = None,
    day_wall_time: float = 0.0,
    collection_day: Optional[bool] = None,
    economics: Any = None,
) -> Dict[str, Any]:
    """Format one executed day.

    ``profit`` is the caller's value when ``economics`` is omitted, which is
    what the existing unit tests pass. When rates are passed, profit comes from
    ``DayOutcome``. An empty tour (depot only) records zero collection. There
    is no ``reward`` key and no ambiguous ``days`` key.

    Args:
        total_collected: Waste collected on the executed plan (kg).
        ncol: Bins collected.
        cost: Distance of the executed plan (km).
        tour: Node sequence. Length at most 2 is an empty tour.
        day: Simulation day index.
        new_overflows: Bin-days that reached capacity before routing.
        sum_lost: Waste lost above bin capacity (kg).
        coordinates: Bin table whose ``ID`` column maps node indices.
        profit: Profit already computed by collection, used when economics is omitted.
        time: Policy compute seconds (packing + routing + improvement).
        mandatory_nodes: iloc indices selected before routing.
        time_spent: Physical shift hours. Omitted from the record when None.
        solver_status: Backend status. An empty tour keeps it.
        day_wall_time: Wall seconds for the whole day, including this record.
        collection_day: Overrides the empty-tour test when the caller already knows.
        economics: Optional ``Economics`` rates. When set, profit is recomputed.

    Returns:
        Daily record with canonical KPI keys, the tour, and the mandatory set.
    """
    from logic.src.envs.common.metrics import DayOutcome
    from logic.src.envs.common.objectives import Economics

    empty = not tour or len(tour) <= 2
    collected_kg = 0.0 if empty else float(total_collected)
    distance_km = 0.0 if empty else float(cost)
    collection_count = 0 if empty else int(ncol)
    is_collection = (not empty) if collection_day is None else bool(collection_day)
    outcome = DayOutcome(
        collected_kg=collected_kg,
        distance_km=distance_km,
        overflows=int(new_overflows),
        collection_count=collection_count,
        lost_kg=float(sum_lost),
        policy_time_s=float(time),
        day_wall_time_s=float(day_wall_time),
        time_spent_h=None if time_spent is None else float(time_spent),
        collection_day=is_collection,
    )
    if economics is not None and not empty:
        dlog = cast(Dict[str, Any], outcome.metrics(economics))
    else:
        dlog = cast(Dict[str, Any], outcome.metrics(Economics(revenue_kg=0.0, cost_km=0.0)))
        dlog["profit"] = 0.0 if empty else float(profit)
    dlog["kg/km"] = (dlog["kg"] / dlog["km"]) if dlog["km"] > 0 else 0.0
    dlog["day"] = day
    mandatory_ids: List[int] = []
    for idx in mandatory_nodes or []:
        try:
            mandatory_ids.append(int(coordinates.iloc[idx]["ID"]))
        except (IndexError, KeyError, TypeError, ValueError):
            mandatory_ids.append(idx)
    dlog["mandatory_nodes"] = mandatory_ids
    dlog["solver_status"] = resolve_solver_status(solver_status, tour)
    if tour and len(tour) > 2:
        coordinate_ids = coordinates["ID"].tolist()
        dlog["tour"] = [0 if node == 0 else coordinate_ids[node] for node in tour]
    else:
        dlog["tour"] = [0]
    return dlog


def _run_non_routing_day(
    context: SimulationDayContext,
    commands: List[Any],
    log_actions: List[Any],
    token: Any,
) -> SimulationDayContext:
    """Execute a simulation day where routing is prohibited by time window constraints."""
    from logic.src.envs.common.expand_pool import reset_expand_pool
    from logic.src.pipeline.simulations.pipeline import apply_shared_bin_rng, shared_env_seed

    wall_start = time.perf_counter()
    reset_solver_status()
    context.solver_status = "no_routing"
    context.metrics_phase = "unset"

    sample_id = getattr(context, "sample_id", 0)
    try:
        sample_id = int(sample_id)
    except (TypeError, ValueError):
        sample_id = 0
    env_seed = shared_env_seed(context.seed, sample_id, context.day)
    apply_shared_bin_rng(context.bins, env_seed)

    try:
        for command in commands:
            if command.__class__.__name__ == "FillAction":
                command.execute(context)

        context.tour = [0]
        context.cost = 0.0
        context.profit = 0.0
        context.ncol = 0
        context.total_collected = 0.0
        if getattr(context, "bins", None) is not None:
            context.collected = np.zeros(cast("Bins", context.bins).n)
        else:
            context.collected = None
        context.mandatory = []
        context.time = 0.0
        context.time_spent = 0.0
        context.solver_status = "no_routing"
        context.metrics_phase = "final"

        for command in log_actions:
            context.day_wall_time = time.perf_counter() - wall_start
            command.execute(context)
    finally:
        context.day_wall_time = time.perf_counter() - wall_start
        if isinstance(context.daily_log, dict):
            context.daily_log["day_wall_time"] = context.day_wall_time
        reset_expand_pool(token)
    return context


def run_day(context: SimulationDayContext) -> SimulationDayContext:
    """Run one validated day.

    Waste noise uses a seed shared by every policy. Policy RNG stays
    policy-specific and does not reseed the bins. ``time`` sums the actions
    marked ``timed``. ``day_wall_time`` is the whole call.

    Args:
        context: The simulation context for the day.

    Returns:
        The same context after the pipeline.
    """
    from logic.src.envs.common.expand_pool import reset_expand_pool, set_expand_pool
    from logic.src.envs.common.released_specs import released_spec
    from logic.src.pipeline.simulations.pipeline import (
        apply_shared_bin_rng,
        policy_seed,
        resolve_pipeline,
        seed_policy_rng,
        shared_env_seed,
    )

    # Validate the selected pipeline before reseeding or mutating simulation state.
    configured = context.day_pipeline
    if configured is None and hasattr(context.config, "get"):
        configured = cast(Any, context.config).get("day_pipeline")
    classes = resolve_pipeline(configured)
    commands = [cls() for cls in classes]
    log_actions = [cmd for cmd in commands if cmd.writes and "daily_log" in cmd.writes]

    reset_solver_status()
    context.solver_status = ""
    context.metrics_phase = "unset"

    sample_id = getattr(context, "sample_id", 0)
    try:
        sample_id = int(sample_id)
    except (TypeError, ValueError):
        sample_id = 0
    env_seed = shared_env_seed(context.seed, sample_id, context.day)
    apply_shared_bin_rng(context.bins, env_seed)
    canonical_name = get_canonical_policy_name(context.policy_name)
    # sim.search_seed shifts only the routing search (repeated searches on the same deposits); 0 keeps the seeds.
    context.policy_seed = policy_seed(context.seed + context.search_seed, canonical_name, context.day)
    seed_policy_rng(context.policy_seed)

    spec = getattr(context, "problem_spec", None)
    raw_problem = getattr(context, "problem", "ptp")
    problem_name = raw_problem if isinstance(raw_problem, str) and raw_problem else "ptp"
    if spec is None or type(spec).__name__ == "MagicMock":
        spec = released_spec(problem_name)
    if spec is None:
        from logic.src.envs.spec import PROBLEMS

        spec = PROBLEMS.get(problem_name)
    context.problem_spec = spec
    token = set_expand_pool(bool(spec.expand_pool))

    if not getattr(context, "routing_allowed", True):
        return _run_non_routing_day(context, commands, log_actions, token)

    wall_start = time.perf_counter()
    policy_time = 0.0
    try:
        for command in commands:
            if command in log_actions:
                context.time = policy_time
                context.day_wall_time = time.perf_counter() - wall_start
                if not context.solver_status:
                    context.solver_status = resolve_solver_status(current_solver_status(), context.tour)
            start = time.perf_counter()
            try:
                command.execute(context)
            except Exception as exc:
                context.day_wall_time = time.perf_counter() - wall_start
                if command not in log_actions and context.daily_log is None:
                    record_aborted_day(context, exc)
                raise
            if getattr(command, "timed", False):
                policy_time += time.perf_counter() - start
    finally:
        context.day_wall_time = time.perf_counter() - wall_start
        if isinstance(context.daily_log, dict):
            context.daily_log["day_wall_time"] = context.day_wall_time
        reset_expand_pool(token)
    return context
