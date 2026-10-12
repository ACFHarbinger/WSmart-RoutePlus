"""
Centralized registry for the WSmart+ Route CLI parser.

Attributes:
    get_parser: Returns the main ConfigsParser with all subcommands registered.

Example:
    >>> from logic.controllers.cli.registry import get_parser
    >>> parser = get_parser()
    >>> parser.parse_args(["test_suite", "-v"])
    Namespace(command='test_suite', verbose=True, ...)
"""

from logic.controllers.cli.base import ConfigsParser
from logic.controllers.cli.bench_parser import add_benchmark_args
from logic.controllers.cli.drive_parser import add_drive_sync_args
from logic.controllers.cli.fs_parser import add_files_args
from logic.controllers.cli.out_parser import add_excel_summary_args, add_output_args
from logic.controllers.cli.tgt_parser import add_ms_update_args, add_ri_update_args
from logic.controllers.cli.ts_parser import add_test_suite_args


def get_parser() -> ConfigsParser:
    """
    Creates and returns the main ConfigsParser with all subcommands registered.

    Returns:
        The main ConfigsParser with all subcommands registered.
    """
    parser = ConfigsParser(description="WSmart+ Route Unified CLI Framework")
    parser.add_argument("--profile", action="store_true", help="Enable function-level execution profiling")
    subparsers = parser.add_subparsers(dest="command", help="The command to execute", required=True)

    # Files
    files_parser = subparsers.add_parser("file_system", help="File system operations")
    add_files_args(files_parser)

    # Test Suite
    ts_parser = subparsers.add_parser("test_suite", help="Run the test suite")
    add_test_suite_args(ts_parser)

    # Benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    add_benchmark_args(bench_parser)

    # Clean Results
    clean_parser = subparsers.add_parser(
        "clean_results",
        help="Remove targeted simulation runs from output artefacts",
    )
    add_output_args(clean_parser)

    # Excel summary
    excel_parser = subparsers.add_parser(
        "excel_summary",
        help="Aggregate simulation results into a single Excel summary",
    )
    add_excel_summary_args(excel_parser)

    # Update mandatory-selection config overrides
    update_ms_parser = subparsers.add_parser(
        "update_ms",
        help="Batch-update mandatory_selection overrides in policy YAML files",
    )
    add_ms_update_args(update_ms_parser)

    # Update route-improvement config overrides
    update_ri_parser = subparsers.add_parser(
        "update_ri",
        help="Batch-update route_improvement overrides in policy YAML files",
    )
    add_ri_update_args(update_ri_parser)

    # Google Drive sync of the data and outputs directories
    sync_parser = subparsers.add_parser(
        "sync_drive",
        help="Copy data and outputs to or from a Google Drive folder (rclone, copy only)",
    )
    add_drive_sync_args(sync_parser)

    # TUI
    return parser
