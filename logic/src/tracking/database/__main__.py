"""Entrypoint for the tracking database CLI.

This module provides the command-line interface for the WSmart-Route tracking
system. It exposes subcommands for database inspection, maintenance, analytics,
and data export.

Attributes:
    main: Function responsible for parsing arguments and dispatching commands.

Example:
    $ python -m logic.src.tracking.database stats --experiment AM-VRPP-50
"""

import argparse
from typing import Optional, Sequence

from logic.src.tracking.database import (
    clean_database,
    compact_database,
    export_run,
    inspect_database,
    metrics_summary,
    prune_database,
    stats_database,
)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parses command-line arguments and executes the requested database command.

    Supported subcommands:
        inspect: Print database overview and recent runs.
        clean: Delete all data, preserve schema.
        compact: Integrity check, WAL checkpoint, VACUUM.
        stats: Show comprehensive database statistics.
        metrics: Show per-metric evolution statistics.
        prune: Remove stale or failed runs.
        export: Serializes a run record to JSON.
    """
    parser = argparse.ArgumentParser(
        prog="commands",
        description="WSmart-Route tracking database management commands.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("inspect", help="Print database overview and recent runs.")
    sub.add_parser("clean", help="Delete all data, preserve schema.")
    sub.add_parser("compact", help="Integrity check, WAL checkpoint, VACUUM.")

    p_stats = sub.add_parser("stats", help="Show comprehensive database statistics.")
    p_stats.add_argument("--experiment", default="", metavar="NAME", help="Limit statistics to one experiment.")

    p_metrics = sub.add_parser("metrics", help="Show per-metric statistics.")
    p_metrics.add_argument("--key", default="", metavar="KEY", help="Drill into a specific metric key.")
    p_metrics.add_argument("--experiment", default="", metavar="NAME", help="Limit to one experiment.")

    p_prune = sub.add_parser("prune", help="Remove stale runs.")
    p_prune.add_argument("--older-than", type=int, default=30, metavar="DAYS")
    p_prune.add_argument("--status", default="failed", choices=["failed", "running", "completed", "all"])
    p_prune.add_argument("--experiment", default="", metavar="NAME")
    p_prune.add_argument("--dry-run", action="store_true")

    p_export = sub.add_parser("export", help="Dump a run to JSON.")
    p_export.add_argument("--run-id", default="", metavar="UUID")
    p_export.add_argument("--experiment", default="", metavar="NAME")
    p_export.add_argument("--latest", action="store_true")
    p_export.add_argument("--output", "-o", default="", metavar="FILE")

    args = parser.parse_args(argv)

    if args.command == "inspect":
        inspect_database()
    elif args.command == "clean":
        clean_database()
    elif args.command == "compact":
        compact_database()
    elif args.command == "prune":
        prune_database(args.older_than, args.status, args.experiment, args.dry_run)
    elif args.command == "export":
        export_run(args.run_id, args.experiment, args.latest, args.output)
    elif args.command == "stats":
        stats_database(args.experiment)
    elif args.command == "metrics":
        metrics_summary(args.key, args.experiment)


if __name__ == "__main__":
    main()
