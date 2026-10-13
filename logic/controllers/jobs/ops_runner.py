"""Operations controller.

Entry point for all argparse-driven operational commands that are **not**
part of the model or simulation pipeline.  These commands maintain the
project's health and output artefacts:

- ``benchmark``       — performance benchmarks (neural decoders, solvers, LS)
- ``test_suite``      — pytest-based test runner
- ``file_system``     — file update / delete / cryptographic operations
- ``clean_results``   — remove targeted simulation runs from output artefacts
- ``excel_summary``   — aggregate simulation results into a single Excel file
- ``update_ms``       — batch-update mandatory-selection strategy overrides
- ``update_ri``       — batch-update route-improver overrides

These commands share no domain logic; they are grouped here because they are
all triggered through the argparse CLI path (``parser_entry_point``) and all
have the same operational nature — they inspect or mutate project files rather
than running computations.

Example::

    python main.py benchmark --subset all --device auto
    python main.py test_suite --module test_models --verbose
    python main.py file_system update --target_entry outputs/simulations/ --output_key cost
    python main.py clean_results --results-dir outputs/simulations/30_days/riomaior_100 --dry-run
    python main.py excel_summary --output-path outputs/simulations/simulation_summary.xlsx
    python main.py update_ms --constructors aco_hh alns bpc hgs --file ms_service_level --keys service_level1
    python main.py update_ri --constructors aco_hh alns --file ri_ftsp --keys ftsp
"""

import argparse
import sys
import traceback
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def run_benchmarks(opts: Dict[str, Any]) -> None:
    """Run the performance benchmark suite.

    Args:
        opts: Validated options dict from
            :func:`~logic.controllers.cli.bench_parser.validate_benchmark_args`.

    Raises:
        SystemExit: Exits with code 0 on success, 1 on failure.
    """
    from logic.benchmark.benchmark_suite import run_benchmarks as _run

    exit_code = 0
    try:
        _run(opts)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(f"\n{e}")
        exit_code = 1
    finally:
        print(f"\nFinished benchmark command execution with exit code: {exit_code}")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(exit_code)


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


def _execute_test_suite(opts: Dict[str, Any]) -> int:
    """Invoke :class:`~logic.test.PyTestRunner` with the given options.

    Args:
        opts: Validated options dict from
            :func:`~logic.controllers.cli.ts_parser.validate_test_suite_args`.
            Expected keys: ``test_dir``, ``list``, ``list_tests``, ``module``,
            ``test_class``, ``test_method``, ``verbose``, ``coverage``,
            ``markers``, ``failed_first``, ``maxfail``, ``capture``, ``tb``,
            ``parallel``, ``keyword``.

    Returns:
        pytest exit code (0 = all passed).
    """
    from logic.test import PyTestRunner

    runner = PyTestRunner(test_dir=opts["test_dir"])

    if opts["list"]:
        runner.list_modules()
        return 0

    if opts["list_tests"]:
        runner.list_tests(opts["module"][0] if opts["module"] else None)
        return 0

    return runner.run_tests(
        modules=opts["module"],
        test_class=opts["test_class"],
        test_method=opts["test_method"],
        verbose=opts["verbose"],
        coverage=opts["coverage"],
        markers=opts["markers"],
        failed_first=opts["failed_first"],
        maxfail=opts["maxfail"],
        capture=opts["capture"],
        tb_style=opts["tb"],
        parallel=opts["parallel"],
        keyword=opts["keyword"],
    )


def run_test_suite(opts: Dict[str, Any]) -> None:
    """Run the test suite and exit with the pytest exit code.

    Args:
        opts: Validated options dict from the ``test_suite`` sub-parser.

    Raises:
        SystemExit: Exits with code 0 on success, 1 on error.
    """
    exit_code = 0
    try:
        exit_code = _execute_test_suite(opts)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(f"\n{e}")
        exit_code = 1
    finally:
        print(f"\nFinished test_suite command execution with exit code: {exit_code}")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(exit_code)


# ---------------------------------------------------------------------------
# File system
# ---------------------------------------------------------------------------


def run_file_system(opts: Dict[str, Any], inner_comm: str) -> None:
    """Execute a file-system sub-command.

    Args:
        opts: Validated options dict from
            :func:`~logic.controllers.cli.fs_parser.validate_file_system_args`.
        inner_comm: The resolved sub-command: ``"update"``, ``"delete"``, or
            ``"cryptography"``.

    Raises:
        SystemExit: Exits with code 0 on success, 1 on failure.
    """
    from logic.controllers.cli.fs_parser import (
        delete_file_system_entries,
        perform_cryptographic_operations,
        update_file_system_entries,
    )

    exit_code = 0
    try:
        if inner_comm == "update":
            update_file_system_entries(opts)
        elif inner_comm == "delete":
            delete_file_system_entries(opts)
        else:
            assert inner_comm == "cryptography"
            perform_cryptographic_operations(opts)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(f"\n{e}")
        exit_code = 1
    finally:
        print(f"\nFinished file_system ({inner_comm}) command execution with exit code: {exit_code}")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(exit_code)


# ---------------------------------------------------------------------------
# Output artefact management
# ---------------------------------------------------------------------------

_OUTPUT_COMMANDS = frozenset({"clean_results", "excel_summary"})


def run_output_command(comm: str, opts: Dict[str, Any]) -> None:
    """Execute a simulation-output management command.

    Args:
        comm: Either ``"clean_results"`` or ``"excel_summary"``.
        opts: Validated options dict for the chosen command.

    Raises:
        SystemExit: Exits with the integer exit code returned by the
            sub-command (0 = success).
    """
    from logic.controllers.cli.out_parser import (
        _run_excel_summary_from_namespace,
    )
    from logic.controllers.cli.out_parser import (
        _run_from_namespace as _run_clean,
    )

    exit_code = 0
    try:
        ns = argparse.Namespace(**opts)
        if comm == "clean_results":
            exit_code = _run_clean(ns)
        else:
            assert comm == "excel_summary"
            exit_code = _run_excel_summary_from_namespace(ns)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(f"\n{e}")
        exit_code = 1
    finally:
        print(f"\nFinished {comm} command execution with exit code: {exit_code}")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(exit_code)


# ---------------------------------------------------------------------------
# Policy target updates
# ---------------------------------------------------------------------------

_TARGET_COMMANDS = frozenset({"update_ms", "update_ri"})


def run_target_update(comm: str, opts: Dict[str, Any]) -> None:
    """Batch-update policy YAML overrides.

    Args:
        comm: Either ``"update_ms"`` (mandatory-selection strategy) or
            ``"update_ri"`` (route-improver).
        opts: Validated options dict for the chosen command.

    Raises:
        SystemExit: Exits with the integer exit code returned by the
            sub-command (0 = success, 1 = error or nothing matched).
    """
    from logic.controllers.cli.tgt_parser import (
        _run_ms_from_namespace,
        _run_ri_from_namespace,
    )

    exit_code = 0
    try:
        ns = argparse.Namespace(**opts)
        if comm == "update_ms":
            exit_code = _run_ms_from_namespace(ns)
        else:
            assert comm == "update_ri"
            exit_code = _run_ri_from_namespace(ns)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(f"\n{e}")
        exit_code = 1
    finally:
        print(f"\nFinished {comm} command execution with exit code: {exit_code}")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(exit_code)
