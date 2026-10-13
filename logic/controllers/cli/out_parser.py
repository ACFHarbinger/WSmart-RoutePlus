"""CLI parser for output artefact operations.

Integrates with the WSmart+ Route CLI registry as the ``clean_results`` and
``excel_summary`` sub-commands, and can also be run standalone::

    # Remove targeted simulation runs via main CLI
    python main.py clean_results \\
        --results-dir outputs/simulator/30_days/riomaior_100 \\
        --distribution emp \\
        --constructor alns \\
        --improver ftsp \\
        --dry-run

    # Aggregate simulation results into Excel via main CLI
    python main.py excel_summary
    python main.py excel_summary --output-path my_summary.xlsx --dirs 31_days/riomaior_104

    # Standalone with subcommands
    uv run python -m logic.controllers.cli.out_parser clean \\
        --results-dir outputs/simulator/30_days/riomaior_100 \\
        --ms-strategy lookahead --distribution gamma3

    uv run python -m logic.controllers.cli.out_parser excel
    uv run python -m logic.controllers.cli.out_parser excel --output-path my_summary.xlsx

All filter options for ``clean`` are optional; an omitted option matches *any* value.
Multiple values can be passed to the same option (space-separated).

Attributes:
    add_output_args: Register ``clean_results`` arguments on an existing sub-parser.
    add_excel_summary_args: Register ``excel_summary`` arguments on an existing sub-parser.
    main: Standalone CLI entry point with ``clean`` / ``excel`` subcommands.
"""

from __future__ import annotations

import argparse
import os
import sys

from logic.src.constants import ROOT_DIR
from logic.src.utils.output.excel_summary import discover_and_aggregate
from logic.src.utils.output.remover import PolicyFilter, remove_targeted_runs

_DEFAULT_EXCEL_OUTPUT = os.path.join(ROOT_DIR, "outputs", "simulator", "simulation_summary.xlsx")


def add_output_args(parser: argparse.ArgumentParser) -> None:
    """Register ``clean_results`` arguments on *parser*.

    Args:
        parser: An :class:`argparse.ArgumentParser` (or sub-parser) to populate.
    """
    parser.add_argument(
        "--results-dir",
        required=True,
        metavar="DIR",
        help=(
            "Path to the results directory, "
            "e.g. 'outputs/simulator/30_days/riomaior_100'.  "
            "Supports any area / day-count combination."
        ),
    )
    parser.add_argument(
        "--distribution",
        nargs="+",
        default=[],
        metavar="DIST",
        help="Distribution(s) to target: emp  gamma1  gamma2  gamma3 …",
    )
    parser.add_argument(
        "--constructor",
        nargs="+",
        default=[],
        metavar="CON",
        help="Route constructor(s) to target: alns  hgs  aco_hh  bpc  pg_clns  sans  psoma …",
    )
    parser.add_argument(
        "--ms-strategy",
        nargs="+",
        default=[],
        metavar="MS",
        help="Mandatory-selection strategy/ies: lookahead  last_minute  regular …",
    )
    parser.add_argument(
        "--improver",
        nargs="+",
        default=[],
        metavar="IMP",
        help="Route improver(s): ftsp  fast_tsp  rls  rds  none …",
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Use exact token matching instead of the default substring matching.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be removed without modifying any files.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-item output; only print the final count.",
    )


def add_excel_summary_args(parser: argparse.ArgumentParser) -> None:
    """Register ``excel_summary`` arguments on *parser*.

    Args:
        parser: An :class:`argparse.ArgumentParser` (or sub-parser) to populate.
    """
    parser.add_argument(
        "--output-path",
        default=_DEFAULT_EXCEL_OUTPUT,
        metavar="PATH",
        help=(f"Destination ``.xlsx`` file.  Defaults to '{_DEFAULT_EXCEL_OUTPUT}'."),
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=[],
        metavar="DIR",
        help=(
            "Whitelist of sub-directories under outputs/simulator/ to include, "
            "e.g. '31_days/riomaior_104 30_days/riomaior_104'.  "
            "When omitted, all directories are scanned."
        ),
    )


def _run_from_namespace(args: argparse.Namespace) -> int:
    """Execute the clean-results action from parsed *args*.

    Args:
        args: Parsed namespace produced by :func:`add_output_args`.

    Returns:
        Exit code (0 = success / items removed, 1 = nothing matched or error).
    """
    if not any([args.distribution, args.constructor, args.ms_strategy, args.improver]):
        print(
            "[ERROR] At least one filter option must be provided "
            "(--distribution, --constructor, --ms-strategy, --improver).\n"
            "Run with --help for usage.",
            file=sys.stderr,
        )
        return 1

    policy_filter = PolicyFilter(
        distributions=args.distribution,
        constructors=args.constructor,
        ms_strategies=args.ms_strategy,
        improvers=args.improver,
        exact_match=args.exact,
    )

    removed = remove_targeted_runs(
        results_dir=args.results_dir,
        policy_filter=policy_filter,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )

    if args.quiet:
        action = "Would remove" if args.dry_run else "Removed"
        print(f"{action} {len(removed)} artefact(s).")

    return 0 if removed else 1


def _run_excel_summary_from_namespace(args: argparse.Namespace) -> int:
    """Execute the excel-summary action from parsed *args*.

    Args:
        args: Parsed namespace produced by :func:`add_excel_summary_args`.

    Returns:
        Exit code (0 = success, 1 = no data found or save error).
    """
    import logic.src.utils.output.excel_summary as _es

    if getattr(args, "dirs", None):
        _es.SELECTED_DIRS = list(args.dirs)

    print("Collecting simulation results...")
    df = discover_and_aggregate()

    if df.empty:
        print("No simulation data found.")
        return 1

    output_path = args.output_path
    print(f"Found {len(df)} policy entries. Exporting to Excel...")

    cols_priority = ["SourceDir", "Distribution", "Policy"]
    df = df.sort_values(cols_priority)

    try:
        df.to_excel(output_path, index=False)
        print(f"SUCCESS: Summary saved to {output_path}")
        return 0
    except Exception as e:
        print(f"ERROR: Failed to save Excel file: {e}", file=sys.stderr)
        return 1


def main(argv=None) -> int:
    """Standalone CLI entry point with ``clean`` / ``excel`` subcommands.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        prog="python -m logic.controllers.cli.out_parser",
        description=(
            "WSmart+ Route output artefact utilities.\n\n"
            "  clean  — remove targeted simulation runs\n"
            "  excel  — aggregate results into an Excel summary"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    clean_parser = subparsers.add_parser(
        "clean",
        help="Remove targeted simulation runs from output artefacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_output_args(clean_parser)

    excel_parser = subparsers.add_parser(
        "excel",
        help="Aggregate simulation results into a single Excel summary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_excel_summary_args(excel_parser)

    args = parser.parse_args(argv)

    if args.subcommand == "clean":
        return _run_from_namespace(args)
    return _run_excel_summary_from_namespace(args)


if __name__ == "__main__":
    sys.exit(main())
