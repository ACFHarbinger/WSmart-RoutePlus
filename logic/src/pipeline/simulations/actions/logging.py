"""
Action for simulation logging.

Attributes:
    LogAction: Command to record daily simulation results.

Example:
    >>> # action = LogAction()
    >>> # action.execute(context)
"""

from typing import Any, Dict

from rich import box
from rich.console import Console
from rich.table import Table

from logic.src.pipeline.simulations.actions.base import SimulationAction, bind_context
from logic.src.pipeline.simulations.sim_day import get_daily_results
from logic.src.tracking.integrations.simulation import get_sim_tracker
from logic.src.tracking.logging.modules.failure_analyzer import FailureAnalyzer
from logic.src.tracking.logging.modules.failure_emit import emit_sim_failure_summary
from logic.src.tracking.logging.modules.gui import send_daily_output_to_gui


class LogAction(SimulationAction):
    """
    Records daily simulation metrics and outputs results.

    Attributes:
        None
    """

    timed = False
    reads = ("tour", "cost", "profit", "bins", "coords", "day")
    writes = ("daily_log",)
    finalizes = ()
    requires_final = ("profit", "cost")

    def execute(self, context: Dict[str, Any]) -> None:
        """
        Log daily results and update GUI.

        Args:
            context: Shared dictionary containing simulation state.
        """

        context = bind_context(context)
        tour = context.tour
        km = context.cost
        total_collected = context.total_collected
        ncol = context.ncol
        profit = context.profit
        new_overflows = context.new_overflows
        coords = context.coords
        day = context.day
        sum_lost = context.sum_lost
        time = context.time
        time_spent = getattr(context, "time_spent", None)
        bins = context.bins
        if getattr(context, "tour", None) is None:
            raise ValueError("logging a day whose tour was never set")
        if getattr(context, "metrics_phase", "final") != "final":
            raise ValueError("logging preliminary metrics; collection has not published the executed plan")
        from logic.src.envs.common.objectives import Economics
        from logic.src.pipeline.simulations.pipeline import per_trip_shift

        economics = None
        revenue = getattr(bins, "revenue", None) if bins is not None else None
        expenses = getattr(bins, "expenses", None) if bins is not None else None
        if isinstance(revenue, (int, float)) and isinstance(expenses, (int, float)) and not isinstance(revenue, bool):
            economics = Economics(revenue_kg=float(revenue), cost_km=float(expenses))
        shift_checked = per_trip_shift(context)

        dlog = get_daily_results(
            total_collected,
            ncol,
            km,
            tour,
            day,
            new_overflows,
            sum_lost,
            coords,
            profit,
            time,
            mandatory_nodes=getattr(context, "mandatory", None),
            time_spent=time_spent,
            solver_status=getattr(context, "solver_status", None),
            day_wall_time=float(getattr(context, "day_wall_time", 0.0) or 0.0),
            economics=economics,
        )

        from logic.src.pipeline.simulations.reporting import current_solver_certificate

        proof = current_solver_certificate()
        dlog["solver_certificate"] = proof

        failure_summary = FailureAnalyzer().analyze(
            new_overflows=new_overflows,
            sum_lost=sum_lost,
            profit=profit,
            fill=getattr(context, "fill", bins.real_c),
            total_fill=getattr(context, "total_fill", bins.c),
            bins_means=bins.means,
            bins_real_c=bins.real_c,
            tour=tour,
            collected=getattr(context, "collected", None),
            coords=coords,
            mandatory=getattr(context, "mandatory", None),
        )
        dlog["failure_analysis"] = failure_summary
        dlog["trip_loads_kg"] = getattr(context, "trip_loads_kg", [])
        dlog["capacity_violations"] = int(getattr(context, "capacity_violations", 0))
        dlog["capacity_splits"] = int(getattr(context, "capacity_splits", 0))
        context.daily_log = dlog
        policy_name = getattr(context, "display_name", context.policy_name)

        # Pretty terminal output
        console = Console()
        table = Table(
            title=f"Day {day} Results: [bold cyan]{policy_name}[/]",
            box=box.ROUNDED,
            header_style="bold magenta",
            border_style="blue",
            title_style="bold white on blue",
            expand=False,
        )

        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="bold green")

        table.add_row("Profit", f"${profit:,.2f}")
        table.add_row("Collected", f"{total_collected:,.2f} kg")
        table.add_row("Bins Collected", str(ncol))
        table.add_row("Mandatory Bins", str(len(dlog.get("mandatory_nodes") or [])))
        table.add_row("Mandatory Set", str(dlog.get("mandatory_nodes", [])))
        table.add_row("Solver Status", str(dlog.get("solver_status", "")))
        table.add_row("Distance", f"{km:.2f} km")
        table.add_row("Efficiency", f"{dlog.get('kg/km', 0):.2f} kg/km")
        if time_spent is not None and (shift_checked or time_spent > 0):
            table.add_row("Shift Time Spent", f"{time_spent:.2f} h")
        table.add_row("Overflows", f"[bold red]{new_overflows}[/]")
        table.add_row("Waste Lost", f"[bold red]{sum_lost:.2f} kg")
        table.add_row("Day Wall Time", f"{float(getattr(context, 'day_wall_time', 0.0) or 0.0):.4f} s")
        table.add_row("Policy Time", f"[bold red]{time:.4f} s")

        # Display the full tour as bin IDs
        # node 0 is the depot, others are bins at their respective iloc positions in coords
        tour_ids_mapped = []
        for node_idx in tour:
            if node_idx == 0:
                tour_ids_mapped.append("[blue]0[/]")
            else:
                try:
                    # Using iloc as node_idx corresponds to the row position in the coordinates DataFrame
                    bin_id = int(coords.iloc[node_idx]["ID"])
                    tour_ids_mapped.append(f"[blue]{bin_id}[/]")
                except (IndexError, KeyError):
                    tour_ids_mapped.append(f"[blue]{node_idx}[/]")

        tour_str = " - ".join(tour_ids_mapped)
        table.add_row("Tour", tour_str)

        # Sync print with lock if available to prevent interleaved output
        lock = getattr(context, "lock", None)
        if lock:
            with lock:
                console.print(table)
        else:
            console.print(table)

        send_daily_output_to_gui(
            dlog,
            policy_name,
            context.sample_id,
            day,
            context.total_fill,
            context.collected,
            bins.real_c,
            context.realtime_log_path,
            tour,
            coords,
            context.lock,
            mandatory=getattr(context, "mandatory", None),
            failure_analysis=failure_summary,
        )

        emit_sim_failure_summary(
            failure_summary,
            policy_name,
            context.sample_id,
            day,
            context.realtime_log_path,
            getattr(context, "lock", None),
        )

        # Forward day-level metrics to the centralised tracker (no-op if no run active)
        sim_tracker = get_sim_tracker(context.policy_name, context.sample_id)
        if sim_tracker is not None:
            sim_tracker.log_day(
                day,
                {key: dlog[key] for key in ("profit", "ncol", "kg", "overflows", "km", "kg_lost", "time")},
            )
