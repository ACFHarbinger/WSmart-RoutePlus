"""Parser dispatch module.

Unified argparse entry point: receives the ``(command, opts)`` tuple from
:func:`logic.controllers.cli.parse_params` and delegates to the appropriate
function in :mod:`logic.controllers.jobs.ops_runner`.  This module owns only the
dispatch table, a pretty-printer helper, and the outermost exception handler.

Supported commands
------------------
- ``"benchmark"``                       → :func:`~logic.controllers.jobs.ops_runner.run_benchmarks`
- ``"test_suite"``                      → :func:`~logic.controllers.jobs.ops_runner.run_test_suite`
- ``"file_system"`` + sub-command       → :func:`~logic.controllers.jobs.ops_runner.run_file_system`
- ``"clean_results"`` / ``"excel_summary"`` → :func:`~logic.controllers.jobs.ops_runner.run_output_command`
- ``"update_ms"`` / ``"update_ri"``     → :func:`~logic.controllers.jobs.ops_runner.run_target_update`
- ``"sync_drive"``                    → :func:`~logic.cloud.cli.run_sync` (rclone copy to or from Google Drive)

Example::

    >>> from logic.controllers.parser_dispatch import parser_entry_point
    >>> from logic.controllers.cli import parse_params
    >>> parser_entry_point(parse_params())
"""

import io
import pprint
import sys
import traceback
from typing import Any, Dict, Optional, Tuple, Union

from logic.controllers.jobs.ops_runner import (
    _OUTPUT_COMMANDS,
    _TARGET_COMMANDS,
    run_benchmarks,
    run_file_system,
    run_output_command,
    run_target_update,
    run_test_suite,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _pretty_print_args(
    comm: str,
    opts: Dict[str, Any],
    inner_comm: Optional[str] = None,
) -> None:
    """Format and print a command's options dictionary.

    Args:
        comm: Primary command name (e.g. ``'benchmark'``).
        opts: Options dictionary to display.
        inner_comm: Optional sub-command name (e.g. ``'update'`` for
            ``file_system``).
    """
    buffer = io.StringIO()
    printer = pprint.PrettyPrinter(width=1, indent=1, sort_dicts=False, stream=buffer)
    printer.pprint(opts)
    output = buffer.getvalue()

    lines = output.splitlines()
    lines[0] = lines[0].lstrip("{")
    lines[-1] = lines[-1].rstrip("}")
    formatted = (
        comm
        + ("" if inner_comm is None else f" {inner_comm}")
        + ": {\n"
        + "\n".join(f" {line}" for line in lines)
        + "\n}"
    )
    print(formatted, end="\n\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parser_entry_point(
    args: Tuple[Union[str, Tuple[str, str]], Dict[str, Any]],
) -> None:
    """Unified entry point for all argparse-driven commands.

    Routes execution to the appropriate :mod:`logic.controllers.jobs.ops_runner` function
    based on the command extracted from ``args``.  Each ops function calls
    ``sys.exit`` internally; this function only handles unexpected top-level
    exceptions.

    Args:
        args: ``(comm, opts)`` as returned by
            :func:`logic.controllers.cli.parse_params`.  ``comm`` may be a
            plain string or a ``(command, sub_command)`` tuple.

    Raises:
        SystemExit: Always — delegates to the individual ops functions.
    """
    comm, opts = args

    if opts.get("profile"):
        from logic.src.tracking.profiling.profiler import start_global_profiling

        start_global_profiling(log_dir=opts.get("log_dir", "logs"))

    inner_comm: Optional[str] = None
    exit_code = 0

    try:
        if isinstance(comm, tuple) and len(comm) > 1:
            comm, inner_comm = comm
            _pretty_print_args(comm, opts, inner_comm)
            assert comm == "file_system"
            run_file_system(opts, inner_comm)

        else:
            _pretty_print_args(comm, opts)

            if comm == "benchmark":
                run_benchmarks(opts)
            elif comm in _OUTPUT_COMMANDS:
                run_output_command(comm, opts)
            elif comm in _TARGET_COMMANDS:
                run_target_update(comm, opts)
            elif comm == "sync_drive":
                from logic.cloud.cli import run_sync

                run_sync(opts)
            else:
                assert comm == "test_suite"
                run_test_suite(opts)

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(f"\n{e}")
        exit_code = 1
        print(
            "\nFinished {}{} command execution with exit code: {}".format(
                comm,
                f" ({inner_comm}) " if inner_comm is not None else "",
                exit_code,
            )
        )
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(exit_code)
