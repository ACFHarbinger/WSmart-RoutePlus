"""Console tables of the simulation results.

Attributes:
    final_simulation_summary: Print one policy's final summary.
    display_simulation_summary_table: Table of every policy's mean and std.
    daily_row_is_visible: Whether a day's row is shown in the per-policy table.
    display_per_policy_simulation_summary: Day-by-day table for each policy.

Example:
    >>> display_simulation_summary_table(log, log_std)
"""

import traceback
from typing import Any, Dict, List, Optional

from loguru import logger
from rich import box
from rich.console import Console
from rich.table import Table

import logic.src.constants as udef

# Rich assumes 80 columns when stdout is not a terminal (a log file, a pipe), which squeezes the
# metric columns of the summary tables to nothing next to the long policy names.
_NON_TERMINAL_WIDTH = 200


def _console() -> Console:
    """Console for the summary tables, wide enough for every column when output is redirected."""
    console = Console()
    if not console.is_terminal:
        console = Console(width=_NON_TERMINAL_WIDTH)
    return console


def final_simulation_summary(log: Dict[str, Any], policy: str, n_samples: int) -> None:
    """Log a final summary of simulation statistics for a policy.

    Args:
        log: The summary log dictionary containing policy keys.
        policy: The specific policy name to summarize.
        n_samples: Total number of samples involved in the run.
    """
    if policy not in log:
        logger.warning(f"Policy {policy} not found in log for summary.")
        return

    # Use the pretty table for a single policy if that's all we have,
    # or if we want to maintain the single-policy logging behavior.
    display_simulation_summary_table(
        {policy: log[policy]},
        title=f"Simulation Summary: [bold cyan]{policy}[/] ({n_samples} samples)",
    )


def display_simulation_summary_table(  # noqa: C901
    log: Dict[str, Any],
    title: str = "Simulation Summary",
    lock: Optional[Any] = None,
) -> None:
    """Display a pretty comparative table of simulation results for multiple policies.

    Args:
        log: Dictionary mapping policy names to metric dictionaries or lists.
        title: Title for the table. Defaults to "Simulation Summary".
        lock: Optional lock for thread-safe printing. Defaults to None.
    """
    if not log:
        return

    console = _console()
    table = Table(
        title=title,
        box=box.DOUBLE_EDGE,
        header_style="bold magenta",
        border_style="blue",
        title_style="bold white on blue",
        show_header=True,
        expand=False,
    )

    # Core metrics to display in the table
    has_time_spent = any(
        isinstance(stats, dict) and "time_spent" in stats and stats["time_spent"] > 0 for stats in log.values()
    )
    display_metrics = [
        ("Profit", "profit"),
        ("Collected", "kg"),
        ("Lost", "kg_lost"),
        ("Col", "ncol"),
        ("Dist", "km"),
        ("Eff", "kg/km"),
        ("Over", "overflows"),
        ("Days", "days_simulated"),
        ("ColDays", "collection_days"),
        ("Time", "time"),
    ]
    if has_time_spent:
        display_metrics.insert(-1, ("ShiftTime", "time_spent"))

    table.add_column("Policy", style="cyan", no_wrap=True)
    for label, _ in display_metrics:
        table.add_column(label, justify="right")

    for pol, stats in log.items():
        row = [pol]
        for _, key in display_metrics:
            if isinstance(stats, dict):
                if key == "days_simulated" and "days_simulated" not in stats and "days" in stats:
                    val = stats.get("days", 0.0)
                else:
                    val = stats.get(key, 0.0)
            elif isinstance(stats, (list, tuple)):
                try:
                    idx = udef.SIM_METRICS.index(key)
                    val = stats[idx]
                except (ValueError, IndexError):
                    val = 0.0
            else:
                val = 0.0

            if key == "profit":
                row.append(f"${val:,.2f}")
            elif key in ["kg", "kg_lost"]:
                row.append(f"{val:,.1f}kg")
            elif key == "km":
                row.append(f"{val:,.1f}km")
            elif key == "kg/km":
                row.append(f"{val:.2f}")
            elif key in ["overflows", "ncol", "days", "days_simulated", "collection_days"]:
                color = "red" if key == "overflows" and val > 0 else "white"
                if key in ("days", "days_simulated", "collection_days"):
                    color = "cyan"
                row.append(f"[{color}]{int(val)}[/]")
            elif key == "time":
                row.append(f"{val:.2f}s")
            elif key == "time_spent":
                row.append(f"{val:.2f}h")
            else:
                row.append(f"{val:.2f}")
        table.add_row(*row)

    if lock:
        with lock:
            console.print(table)
    else:
        console.print(table)


def daily_row_is_visible(km_val: float, mandatory: Any, solver_status: Any) -> bool:
    """Return whether one day belongs in the printed daily table.

    Days that travelled are always shown. A day that did not travel is shown
    when it still has a mandatory set or a solver status other than a skip or
    a plain success, so an empty tour does not erase that evidence.

    Args:
        km_val: Distance recorded for the day.
        mandatory: Mandatory bin ids recorded for the day.
        solver_status: Solver status recorded for the day.

    Returns:
        True when the row should be printed.
    """
    if km_val > 0 or mandatory:
        return True
    return solver_status not in (None, "", "ok", "skipped", 0)


def display_per_policy_simulation_summary(  # noqa: C901
    pol_name: str,
    sample_id: int,
    aggregate_metrics: List[float],
    daily_log: Dict[str, List[Any]],
    title_prefix: str = "Results for",
    lock: Optional[Any] = None,
) -> None:
    """Display detailed results for a single policy simulation run.

    Args:
        pol_name: Name of the policy.
        sample_id: ID of the sample/seed.
        aggregate_metrics: List of aggregate metrics for the entire run.
        daily_log: Dictionary of daily metrics.
        title_prefix: Prefix for the table titles.
        lock: Optional lock for thread-safe printing.
    """
    console = _console()

    # 1. STATISTICS SUMMARY TABLE
    summary_title = f"{title_prefix} [bold cyan]{pol_name}[/] (Sample #{sample_id}) - [yellow]Stats Summary[/]"

    def _print_tables():  # noqa: C901
        try:
            # Using rule for better separation
            console.rule(f"[bold white on blue] {summary_title} [/]")
            display_simulation_summary_table({pol_name: aggregate_metrics}, title=None, lock=None)

            # 2. DAILY PERFORMANCE TABLE (Filtered)
            daily_title = f"{title_prefix} [bold cyan]{pol_name}[/] (Sample #{sample_id}) - [green]Daily Routes[/]"

            table = Table(
                box=box.MINIMAL_DOUBLE_HEAD,
                header_style="bold magenta",
                border_style="green",
                show_header=True,
                expand=False,
            )

            columns = [
                ("Day", "day", "cyan"),
                ("Status", "solver_status", "magenta"),
                ("Mandatory", "mandatory_nodes", "yellow"),
                ("Tour", "tour", "white"),
                ("Profit", "profit", "green"),
                ("KG", "kg", "white"),
                ("Lost", "kg_lost", "red"),
                ("Col", "ncol", "white"),
                ("Dist", "km", "white"),
                ("Eff", "kg/km", "yellow"),
                ("Over", "overflows", "red"),
            ]
            if "time_spent" in daily_log and any(
                isinstance(v, (int, float)) and v > 0 for v in daily_log["time_spent"]
            ):
                columns.append(("ShiftTime", "time_spent", "white"))

            for label, _, style in columns:
                table.add_column(
                    label,
                    style=style,
                    justify="right" if label not in ("Tour", "Mandatory", "Status") else "left",
                )

            # Use km list length as the authoritative iteration count
            kms = daily_log.get("km", [])
            mandatory_rows = daily_log.get("mandatory_nodes", [])
            status_rows = daily_log.get("solver_status", [])

            has_active_days = False
            for i, km_val in enumerate(kms):
                mandatory = mandatory_rows[i] if i < len(mandatory_rows) else None
                status = status_rows[i] if i < len(status_rows) else None
                # A zero-km day still carries the mandatory set and the solver
                # status. Hiding it is what made empty tours look like "no selection".
                if not daily_row_is_visible(km_val, mandatory, status):
                    continue
                has_active_days = True
                row = []
                for _, key, _ in columns:
                    if key == "day" or key is None:
                        # "Day" column — synthetic 1-based index (not stored in daily_log)
                        row.append(str(i + 1))
                        continue
                    vals = daily_log.get(key, [])
                    val = vals[i] if i < len(vals) else None

                    if key == "mandatory_nodes":
                        mand_str = str(val) if val else "[]"
                        if len(mand_str) > 40:
                            mand_str = mand_str[:37] + "..."
                        row.append(mand_str)
                    elif key == "tour":
                        tour_str = str(val) if val is not None else "[]"
                        if len(tour_str) > 50:
                            tour_str = tour_str[:47] + "..."
                        row.append(tour_str)
                    elif key == "profit":
                        if val is not None:
                            color = "green" if val > 0 else "red"
                            row.append(f"[{color}]${val:,.2f}[/]")
                        else:
                            row.append("-")
                    elif key in ["kg", "kg_lost"]:
                        row.append(f"{val:,.1f}kg" if val is not None else "0.0kg")
                    elif key == "km":
                        row.append(f"{val:,.1f}km" if val is not None else "0.0km")
                    elif key == "kg/km":
                        row.append(f"{val:.2f}" if val is not None else "0.00")
                    elif key == "overflows":
                        if val is not None:
                            color = "red" if val > 0 else "white"
                            row.append(f"[{color}]{int(val)}[/]")
                        else:
                            row.append("0")
                    elif key == "ncol":
                        row.append(f"{int(val)}" if val is not None else "0")
                    elif key == "time_spent":
                        row.append(f"{val:.2f}h" if val is not None else "0.00h")
                    else:
                        row.append(str(val))
                table.add_row(*row)

            if has_active_days:
                console.print("\n")
                console.rule(f"[bold white on green] {daily_title} [/]")
                console.print(table)
            else:
                console.print("\n[yellow]No routes performed during this simulation run (all KM=0).[/]")

            console.print("\n")
        except Exception as e:
            print(f"\n[ERROR] Failed to display simulation summary: {e}")
            traceback.print_exc()

    if lock:
        with lock:
            _print_tables()
    else:
        _print_tables()
