"""
Simulation Running State.

This module provides the RunningState class, which orchestrates the main
day-by-day simulation loop and manages checkpoints.

Attributes:
    RunningState: State responsible for the simulation execution loop.

Example:
    >>> # from logic.src.pipeline.simulations.states.running import RunningState
    >>> # state = RunningState()
    >>> # state.handle(ctx)
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from multiprocessing.synchronize import Lock
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

from loguru import logger

from logic.src.constants import METRICS

try:
    from logic.src.tracking.integrations.data_lineage import DataLineageCallback
except ImportError:
    DataLineageCallback = None  # type: ignore[assignment,misc]

from logic.src.pipeline.simulations.checkpoints import CheckpointError, checkpoint_manager
from logic.src.pipeline.simulations.daily import SimulationDayContext, run_day
from logic.src.pipeline.simulations.states.base.base import SimState
from logic.src.pipeline.simulations.states.finishing import FinishingState
from logic.src.tracking.logging.modules.summary_display import final_simulation_summary
from logic.src.utils.infrastructure.setup_sims import get_graph_config

if TYPE_CHECKING:
    from .base import SimulationContext

_CUMULATIVE_METRICS = frozenset(METRICS) - {"kg/km"}


def pad_model_ls(model_tup: Optional[tuple]) -> tuple:
    """Return at least three model slots.

    Neural-agent execution unpacks ``(model_data, graph, profit_vars)`` and the
    branch-and-price policy indexes the same three positions. A missing or short
    tuple must not raise ``ValueError`` before the policy runs.

    Args:
        model_tup: Models loaded for this sample, or None.

    Returns:
        The original tuple when it already has three or more entries, otherwise
        padded with None.
    """
    items = tuple(model_tup) if model_tup else ()
    if len(items) >= 3:
        return items
    return items + (None,) * (3 - len(items))


class RunningState(SimState):
    """State handles the day-by-day simulation loop.

    Attributes:
        None
    """

    def handle(self, ctx: SimulationContext) -> None:
        """Handle the day-by-day simulation loop.

        Args:
            ctx: The simulation context object.
        """
        sim = ctx.cfg.sim
        graph = get_graph_config(sim)

        realtime_log_path = os.path.join(
            ctx.results_dir,
            f"log_realtime_{sim.data_distribution}_{graph.n_samples}N.jsonl",
        )

        # Stored elapsed is already spent. Subtract it so the next interval is
        # that elapsed time plus new work, not a negative clock (DS-17).
        ctx.tic = time.perf_counter() - ctx.run_time

        try:
            assert ctx.checkpoint is not None
            with checkpoint_manager(ctx.checkpoint, sim.checkpoint_days, ctx.get_current_state_tuple) as hook:
                hook.set_timer(ctx.tic)
                iterator = range(ctx.start_day, graph.n_days + 1)
                self._run_simulation_days(ctx, iterator, hook, realtime_log_path)

            logger.info(f"Simulation loop for policy {ctx.pol_name} complete. Processed {graph.n_days} days.")
            ctx.transition_to(FinishingState())
        except CheckpointError as e:
            ctx.result = dict(e.error_result)
            ctx.result["success"] = False
            ctx.result["status"] = "failed"
            ctx.result["exception_type"] = ctx.result.get("error_type") or type(e.__cause__ or e).__name__
            ctx.result["failed_day"] = ctx.result.get("day")
            status_rows = (ctx.daily_log or {}).get("solver_status") or []
            mandatory_rows = (ctx.daily_log or {}).get("mandatory_nodes") or []
            if status_rows:
                ctx.result["solver_status"] = status_rows[-1]
            if mandatory_rows:
                ctx.result["mandatory_nodes"] = mandatory_rows[-1]
            kg_rows = (ctx.daily_log or {}).get("kg") or []
            if kg_rows:
                from logic.src.constants import SIM_METRICS
                from logic.src.envs.common.metrics import METRICS as KPI

                records = []
                for index in range(len(kg_rows)):
                    record = {}
                    for key in SIM_METRICS:
                        column = cast(Dict[str, List[Any]], ctx.daily_log).get(key)
                        if isinstance(column, list) and index < len(column):
                            record[key] = column[index]
                    records.append(record)
                ctx.result["metrics"] = KPI.aggregate_days(records, [key for key in SIM_METRICS if key in records[0]])
            if ctx.result:
                final_simulation_summary(ctx.result, ctx.pol_name, graph.n_samples)
            ctx.transition_to(None)

    def _run_simulation_days(self, ctx, iterator, hook, realtime_log_path):
        """
        Args:
            ctx: The simulation context object.
            iterator: Range of days to simulate.
            hook: Checkpoint hook for state persistence.
            realtime_log_path: Path for saving daily logs.
        """
        log_freq = int(getattr(ctx.cfg.sim, "tracking_log_freq", 1))
        lineage = (
            DataLineageCallback(ctx.pol_name, ctx.sample_id, log_freq=log_freq)
            if DataLineageCallback is not None
            else None
        )
        if lineage:
            lineage.on_simulation_start(ctx)

        for day in iterator:
            hook.before_day(day)

            current_policy_config = self._get_current_policy_config(ctx)
            day_context = self._create_day_context(ctx, day, current_policy_config, realtime_log_path)

            try:
                day_context = run_day(day_context)
            except Exception:
                # The day loop logged the aborted day before raising. Keep that
                # row so the failure result still carries the mandatory set.
                if day_context.daily_log is not None:
                    self._update_metrics(ctx, day, day_context.daily_log)
                raise
            ctx.execution_time = time.perf_counter() - ctx.tic

            self._update_ctx_from_day_context(ctx, day_context)
            cum_metrics = self._update_metrics(ctx, day, day_context.daily_log)
            if lineage:
                lineage.on_step_end(day_context, day)

            hook.after_day(ctx.execution_time)
            if ctx.overall_progress:
                ctx.overall_progress.update(1)

            if ctx.callback:
                ctx.callback(day, cum_metrics, ctx.sample_id)

    def _get_current_policy_config(self, ctx):
        """Standardizes policy config resolution from structured context.

        Args:
            ctx: The simulation context object.

        Returns:
            Dictionary containing resolved configuration for the current day.
        """
        cached = getattr(ctx, "_policy_config_cache", None)
        if cached is not None:
            return dict(cached)

        current_policy_config = {}  # type: ignore[var-annotated]

        # 1. START WITH GLOBAL CONFIGS (mandatory, route_improvement)
        if ctx.config:
            for g_key in ["packing", "route_improvement"]:
                if g_key in ctx.config:
                    current_policy_config[g_key] = ctx.config[g_key]

        # 2. ADD POLICY-SPECIFIC CONFIG FROM CONTEXT
        # This can come from ctx.pol_cfg (structured) or ctx.config[ctx.pol_id_orig] (auto-expanded test-sim)
        if ctx.config and ctx.pol_id_orig in ctx.config:
            pol_data = ctx.config[ctx.pol_id_orig]
            if isinstance(pol_data, (dict, Mapping)):
                # Ensure it's deeply converted or updated properly
                current_policy_config.update(dict(pol_data))

        if ctx.pol_cfg and isinstance(ctx.pol_cfg, (dict, Mapping)):
            current_policy_config.update(dict(ctx.pol_cfg))

        ctx._policy_config_cache = current_policy_config
        return dict(current_policy_config)

    def _create_day_context(self, ctx, day, current_policy_config, realtime_log_path):
        """
        Args:
            ctx: The simulation context object.
            day: Current simulation day.
            current_policy_config: Resolved configuration for the policy.
            realtime_log_path: Path for saving daily logs.

        Returns:
            A new SimulationDayContext object for the day.
        """
        sim = ctx.cfg.sim
        assert ctx.dist_tup is not None
        (distance_matrix, paths_between_states, dm_tensor, distancesC) = ctx.dist_tup

        display_name = getattr(ctx, "display_name", ctx.pol_name)
        time_windows = getattr(ctx, "time_windows", None)
        routing_allowed = True if time_windows is None else time_windows.is_routing_allowed(day)

        return SimulationDayContext(
            full_policy=ctx.pol_name,
            policy_name=ctx.pol_name,
            display_name=display_name,  # pyrefly: ignore [bad-argument-type]
            bins=ctx.bins,
            new_data=ctx.new_data,
            coords=ctx.coords,
            sample_id=ctx.sample_id,
            overflows=ctx.overflows,
            day=day,
            model_env=ctx.model_env,
            model_ls=pad_model_ls(ctx.model_tup),
            n_vehicles=sim.n_vehicles,
            area=sim.graph.area,
            realtime_log_path=realtime_log_path,
            waste_type=sim.graph.waste_type,
            distance_matrix=distance_matrix,
            distancesC=distancesC,
            paths_between_states=paths_between_states,
            dm_tensor=dm_tensor,
            current_collection_day=ctx.current_collection_day,
            cached=ctx.cached,
            device=ctx.device,
            lock=cast(Optional[Lock], ctx.lock),
            hrl_manager=ctx.hrl_manager,
            config=current_policy_config,
            day_pipeline=getattr(sim, "day_pipeline", None),
            cost_weight=sim.policy_configs.get("cost_weight", 1.0),
            waste_weight=sim.policy_configs.get("waste_weight", 1.0),
            overflow_penalty=sim.policy_configs.get("overflow_penalty", 1.0),
            engine=sim.policy_configs.get("engine"),
            threshold=sim.policy_configs.get("threshold"),
            seed=sim.seed,
            search_seed=int(getattr(sim, "search_seed", 0) or 0),
            problem=str(getattr(sim, "problem", "ptp") or "ptp").lower(),
            shift_hours=float(getattr(ctx, "shift_hours", 7.0)),
            time_matrix=getattr(ctx, "time_matrix", None),
            avg_speed_kmh=float(getattr(ctx, "avg_speed_kmh", 35.0)),
            service_time_h=float(getattr(ctx, "service_time_h", 1.5 / 60.0)),
            vehicle_capacity=float(
                getattr(ctx, "vehicle_capacity", 100.0) if getattr(ctx, "vehicle_capacity", None) is not None else 100.0
            ),
            time_windows=time_windows,
            routing_allowed=routing_allowed,
        )

    def _update_ctx_from_day_context(self, ctx, day_context):
        """
        Args:
            ctx: The simulation context object.
            day_context: The SimulationDayContext object after day execution.
        """
        ctx.new_data = day_context.new_data
        ctx.coords = day_context.coords
        ctx.bins = day_context.bins
        ctx.overflows = day_context.overflows
        ctx.cached = day_context.cached

        if ctx.counter:
            with ctx.counter.get_lock():
                ctx.counter.value += 1

    def _update_metrics(self, ctx, day, dlog) -> Dict[str, float]:
        """
        Args:
            ctx: The simulation context object.
            day: Current simulation day.
            dlog: Daily log dictionary.
        """
        if dlog is not None:
            daily_log = ctx.daily_log
            sums = ctx._metric_sums
            list_ids = ctx._metric_list_ids
            list_lengths = ctx._metric_list_lengths

            if not ctx._metric_sums_ready:
                for key, values in daily_log.items():
                    list_ids[key] = id(values)
                    list_lengths[key] = len(values)
                    if key in _CUMULATIVE_METRICS:
                        sums[key] = sum(values)
                ctx._metric_sums_ready = True

            for key, val in dlog.items():
                if key not in daily_log:
                    daily_log[key] = []
                values = daily_log[key]
                if key in _CUMULATIVE_METRICS:
                    if key in sums and list_ids.get(key) == id(values) and list_lengths.get(key) == len(values):
                        sums[key] += val
                    else:
                        sums[key] = sum(values) + val
                values.append(val)
                list_ids[key] = id(values)
                list_lengths[key] = len(values)

            if ctx.shared_metrics is not None:
                cumulative_metrics = {
                    k: sums[k] for k, v in daily_log.items() if k in _CUMULATIVE_METRICS and k in sums
                }
                ctx.shared_metrics[f"{ctx.pol_name}_{ctx.sample_id}"] = {
                    "day": day,
                    "metrics": cumulative_metrics,
                    "daily_delta": dlog,
                    "policy": ctx.pol_name,
                    "sample_id": ctx.sample_id,
                }
                return cumulative_metrics  # pyrefly: ignore [bad-return]
        return {}
