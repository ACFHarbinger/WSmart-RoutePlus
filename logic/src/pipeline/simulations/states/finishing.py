"""
Simulation Finishing State.

This module provides the FinishingState class, which handles the final
aggregation and persistence of simulation results.

Attributes:
    FinishingState: State responsible for simulation cleanup and reporting.

Example:
    >>> # from logic.src.pipeline.simulations.states.finishing import FinishingState
    >>> # state = FinishingState()
    >>> # state.handle(ctx)
"""

from __future__ import annotations

import contextlib
import os
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from logic.src.constants import SIM_METRICS
from logic.src.data.processor import save_matrix_to_excel
from logic.src.tracking.logging.log_utils import (
    display_per_policy_simulation_summary,
    display_simulation_summary_table,
    update_policy_log_section,
)
from logic.src.utils.infrastructure.setup_sims import get_graph_config

try:
    from logic.src.tracking.core.run import get_active_run
    from logic.src.tracking.integrations.simulation import get_sim_tracker
except ImportError:
    get_active_run = None  # type: ignore[assignment,misc]
    get_sim_tracker = None  # type: ignore[assignment,misc]

from .base import SimState

if TYPE_CHECKING:
    from .base import SimulationContext


class FinishingState(SimState):
    """
    State handles final result aggregation and persistence.

    Attributes:
        None
    """

    def handle(self, ctx: SimulationContext) -> None:
        """
        Finalizes simulation, computes aggregate metrics, and saves logs.

        Args:
            ctx: The simulation context object.
        """
        sim = ctx.cfg.sim
        graph = get_graph_config(sim)
        assert ctx.bins is not None

        ctx.execution_time = time.perf_counter() - ctx.tic
        # Sample "time" = total policy time (selection + construction + improvement)
        # summed over the days, not the wall clock of the whole day loop (DS-15).
        policy_time = float(np.sum(ctx.daily_log.get("time", []))) if ctx.daily_log else 0.0

        lg = [
            np.sum(ctx.bins.inoverflow),
            np.sum(ctx.bins.collected),
            np.sum(ctx.bins.ncollections),
            np.sum(ctx.bins.lost),
            ctx.bins.travel,
            (np.sum(ctx.bins.collected) / ctx.bins.travel if ctx.bins.travel > 0 else 0.0),
            np.sum(ctx.bins.collected) - np.sum(ctx.bins.inoverflow) - ctx.bins.travel,
            ctx.bins.profit,
            policy_time,
            ctx.bins.ndays,
        ]

        log_path = os.path.join(
            ctx.results_dir,
            f"log_{ctx.pol_name}_{graph.n_samples}N.json",
        )
        print(ctx.results_dir)

        sample_metrics = dict(zip(SIM_METRICS, lg, strict=False))
        assert ctx.daily_log is not None
        if "time_spent" in ctx.daily_log and ctx.daily_log["time_spent"]:
            sample_metrics["time_spent"] = float(np.sum(ctx.daily_log["time_spent"]))

        daily_dict = {k: list(v) for k, v in ctx.daily_log.items()}

        update_policy_log_section(log_path, "samples", sample_metrics, sample_id=ctx.sample_id, lock=ctx.lock)
        update_policy_log_section(log_path, "daily", daily_dict, sample_id=ctx.sample_id, lock=ctx.lock)

        save_matrix_to_excel(
            ctx.bins.get_fill_history(),
            ctx.results_dir,
            sim.seed,
            sim.data_distribution,
            ctx.pol_name,
            ctx.sample_id,
        )

        # Log the fill-history export as a "save" dataset event
        with contextlib.suppress(Exception):
            run = get_active_run()  # pyrefly: ignore [not-callable]
            if run is not None:
                excel_path = os.path.join(
                    ctx.results_dir,
                    "fill_history",
                    sim.data_distribution,
                    f"{ctx.pol_name}{sim.seed}_sample{ctx.sample_id}.xlsx",
                )
                run.log_dataset_event(
                    "save",
                    file_path=excel_path,
                    metadata={
                        "event": "fill_history_export",
                        "variable_name": "fill_history",
                        "policy": ctx.pol_name,
                        "sample_id": ctx.sample_id,
                        "source_file": "states/finishing.py",
                        "source_line": 90,
                    },
                )

        # Register all output files as tracking artifacts
        _log_result_artifacts(ctx, sim, log_path)

        if ctx.checkpoint:
            ctx.checkpoint.clear()

        ctx.result = {ctx.pol_name: lg, "success": True}

        # Detailed per-policy result table (aggregate + daily)
        if ctx.daily_log is not None:
            display_per_policy_simulation_summary(
                ctx.pol_name,
                ctx.sample_id,
                lg,
                ctx.daily_log,
                lock=ctx.lock,
            )

        # Aggregate results into shared_metrics and display summary table if all tasks are finished
        lock = ctx.lock
        shared_metrics = ctx.shared_metrics
        task_count = ctx.variables_dict.get("task_count", 0)

        if shared_metrics is not None and lock is not None:
            with lock:
                # Store result with a unique key per policy and sample
                shared_metrics[f"res_{ctx.pol_name}_{ctx.sample_id}"] = lg
                finished = shared_metrics.get("finished_tasks", 0) + 1
                shared_metrics["finished_tasks"] = finished

                # If this is the last task, aggregate everything and print the comparative table
                if finished == task_count and task_count > 0:
                    from logic.src.utils.infrastructure.setup_sims import get_pol_name

                    all_aggregated_results = {}
                    policies = sim.full_policies

                    for policy in policies:
                        p_name = get_pol_name(policy)
                        pol_results = []
                        # Collect all samples for this policy
                        for s_id in range(sim.graph.n_samples):
                            res_key = f"res_{p_name}_{s_id}"
                            if res_key in shared_metrics:
                                pol_results.append(shared_metrics[res_key])

                        if pol_results:
                            # Compute mean across samples for each metric
                            mean_metrics = [float(np.mean(m)) for m in zip(*pol_results, strict=False)]
                            all_aggregated_results[p_name] = mean_metrics

                    if all_aggregated_results:
                        display_simulation_summary_table(
                            all_aggregated_results,
                            title=f"Simulation Summary: [bold cyan]{sim.graph.area}[/] ({sim.graph.n_samples} Samples, {sim.graph.n_days} Days)",
                        )

        # Forward final aggregated metrics to the centralised tracker (no-op if no run active)
        if get_sim_tracker is not None:
            sim_tracker = get_sim_tracker(ctx.pol_name, ctx.sample_id)
            if sim_tracker is not None:
                sim_tracker.log_final(SIM_METRICS, lg)

        ctx.transition_to(None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_result_artifacts(ctx: Any, sim: Any, log_path: str) -> None:
    """Register simulation output files with the active tracking run.

    Silently no-ops if no run is active or if any file doesn't exist yet.

    Registered artifact types:

    * ``result`` — per-policy JSON log containing samples, daily, mean, and std
    * ``result`` — Excel fill-history matrix with metadata tag ``fill_history``

    Args:
        ctx: The simulation context object.
        sim: Root configuration object.
        log_path: Path to the per-policy log file.
    """
    with contextlib.suppress(Exception):
        run = get_active_run()  # pyrefly: ignore [not-callable]
        if run is None:
            return

        metadata = {"policy": ctx.pol_name, "sample_id": ctx.sample_id}

        if os.path.exists(log_path):
            run.log_artifact(log_path, artifact_type="result", metadata=metadata)

        # Excel fill-history: {results_dir}/fill_history/{dist}/{policy}{seed}_sample{id}.xlsx
        excel_path = os.path.join(
            ctx.results_dir,
            "fill_history",
            sim.data_distribution,
            f"{ctx.pol_name}{sim.seed}_sample{ctx.sample_id}.xlsx",
        )
        if os.path.exists(excel_path):
            run.log_artifact(
                excel_path,
                artifact_type="result",
                metadata={**metadata, "type": "fill_history"},
            )
