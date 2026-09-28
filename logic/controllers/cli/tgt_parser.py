"""CLI parser for updating policy YAML config overrides.

Integrates with the WSmart+ Route CLI registry as the ``update_ms`` and
``update_ri`` sub-commands, and can also be run standalone::

    # Update mandatory-selection strategy via main CLI
    python main.py update_ms \\
        --constructors aco_hh alns bpc hgs \\
        --file ms_service_level \\
        --keys service_level1 service_level2

    # Update route improver via main CLI
    python main.py update_ri \\
        --constructors aco_hh alns bpc hgs \\
        --file ri_ftsp \\
        --keys ftsp

    # Standalone with subcommands
    uv run python -m logic.controllers.cli.tgt_parser ms \\
        --constructors aco_hh alns \\
        --file ms_last_minute \\
        --keys last_minute_cf70 last_minute_cf90

    uv run python -m logic.controllers.cli.tgt_parser ri \\
        --constructors aco_hh alns \\
        --file ri_cls \\
        --keys default

All constructor names match the ``policy_{constructor}.yaml`` filenames in
``logic/configs/policies/``.  Use ``--list-strategies`` / ``--list-improvers``
to discover available files and ``--list-keys`` to see valid key names.

Attributes:
    add_ms_update_args: Register ``update_ms`` arguments on an existing sub-parser.
    add_ri_update_args: Register ``update_ri`` arguments on an existing sub-parser.
    main: Standalone CLI entry point with ``ms`` / ``ri`` subcommands.
"""

from __future__ import annotations

import argparse
import sys

from logic.src.utils.target.policy_link_updater import (
    list_available_ms_strategies,
    list_available_ri_improvers,
    list_improver_keys,
    list_strategy_keys,
    update_mandatory_selection,
    update_route_improvement,
)

_ALL_CONSTRUCTORS = [
    "aco_hh",
    "alns",
    "bpc",
    "hgs",
    "pg_clns",
    "psoma",
    "sans",
    "swc_tcf",
]


def add_ms_update_args(parser: argparse.ArgumentParser) -> None:
    """Register ``update_ms`` arguments on *parser*.

    Args:
        parser: An :class:`argparse.ArgumentParser` (or sub-parser) to populate.
    """
    parser.add_argument(
        "--constructors",
        nargs="+",
        default=_ALL_CONSTRUCTORS,
        metavar="CON",
        help=(f"Route constructor(s) whose policy file(s) to update.  Defaults to all: {' '.join(_ALL_CONSTRUCTORS)}"),
    )
    parser.add_argument(
        "--file",
        metavar="MS_FILE",
        help=(
            "Mandatory-selection strategy file stem, e.g. 'ms_service_level'.  "
            "Must be present in logic/configs/policies/other/.  "
            "Required unless --list-strategies or --list-keys is used."
        ),
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        metavar="KEY",
        help=(
            "Keys to activate from the strategy file, "
            "e.g. 'service_level1 service_level2'.  "
            "Required unless --list-strategies or --list-keys is used."
        ),
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="Print available mandatory-selection strategy files and exit.",
    )
    parser.add_argument(
        "--list-keys",
        metavar="MS_FILE",
        help="Print available keys for the given strategy file and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying any files.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file output; only print the final count.",
    )


def add_ri_update_args(parser: argparse.ArgumentParser) -> None:
    """Register ``update_ri`` arguments on *parser*.

    Args:
        parser: An :class:`argparse.ArgumentParser` (or sub-parser) to populate.
    """
    parser.add_argument(
        "--constructors",
        nargs="+",
        default=_ALL_CONSTRUCTORS,
        metavar="CON",
        help=(f"Route constructor(s) whose policy file(s) to update.  Defaults to all: {' '.join(_ALL_CONSTRUCTORS)}"),
    )
    parser.add_argument(
        "--file",
        metavar="RI_FILE",
        help=(
            "Route-improver file stem, e.g. 'ri_ftsp'.  "
            "Must be present in logic/configs/policies/other/.  "
            "Required unless --list-improvers or --list-keys is used."
        ),
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        metavar="KEY",
        help=(
            "Keys to activate from the improver file, e.g. 'ftsp'.  "
            "Required unless --list-improvers or --list-keys is used."
        ),
    )
    parser.add_argument(
        "--list-improvers",
        action="store_true",
        help="Print available route-improver files and exit.",
    )
    parser.add_argument(
        "--list-keys",
        metavar="RI_FILE",
        help="Print available keys for the given improver file and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying any files.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file output; only print the final count.",
    )


def _run_ms_from_namespace(args: argparse.Namespace) -> int:
    """Execute the update_ms action from parsed *args*.

    Args:
        args: Parsed namespace produced by :func:`add_ms_update_args`.

    Returns:
        Exit code (0 = success, 1 = error or nothing matched).
    """
    if getattr(args, "list_strategies", False):
        strategies = list_available_ms_strategies()
        print("Available mandatory-selection strategies:")
        for s in strategies:
            print(f"  {s}")
        return 0

    list_keys_for = getattr(args, "list_keys", None)
    if list_keys_for:
        keys = list_strategy_keys(list_keys_for)
        if not keys:
            print(
                f"[ERROR] No keys found for '{list_keys_for}'. Does the file exist?",
                file=sys.stderr,
            )
            return 1
        print(f"Available keys in {list_keys_for}:")
        for k in keys:
            print(f"  {k}")
        return 0

    if not args.file or not args.keys:
        print(
            "[ERROR] --file and --keys are required unless --list-strategies or "
            "--list-keys is used.\nRun with --help for usage.",
            file=sys.stderr,
        )
        return 1

    try:
        modified = update_mandatory_selection(
            constructors=args.constructors,
            ms_yaml=args.file,
            keys=args.keys,
            dry_run=args.dry_run,
            verbose=not args.quiet,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if args.quiet:
        action = "Would update" if args.dry_run else "Updated"
        print(f"{action} {len(modified)} policy file(s).")

    return 0 if modified else 1


def _run_ri_from_namespace(args: argparse.Namespace) -> int:
    """Execute the update_ri action from parsed *args*.

    Args:
        args: Parsed namespace produced by :func:`add_ri_update_args`.

    Returns:
        Exit code (0 = success, 1 = error or nothing matched).
    """
    if getattr(args, "list_improvers", False):
        improvers = list_available_ri_improvers()
        print("Available route improvers:")
        for i in improvers:
            print(f"  {i}")
        return 0

    list_keys_for = getattr(args, "list_keys", None)
    if list_keys_for:
        keys = list_improver_keys(list_keys_for)
        if not keys:
            print(
                f"[ERROR] No keys found for '{list_keys_for}'. Does the file exist?",
                file=sys.stderr,
            )
            return 1
        print(f"Available keys in {list_keys_for}:")
        for k in keys:
            print(f"  {k}")
        return 0

    if not args.file or not args.keys:
        print(
            "[ERROR] --file and --keys are required unless --list-improvers or "
            "--list-keys is used.\nRun with --help for usage.",
            file=sys.stderr,
        )
        return 1

    try:
        modified = update_route_improvement(
            constructors=args.constructors,
            ri_yaml=args.file,
            keys=args.keys,
            dry_run=args.dry_run,
            verbose=not args.quiet,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if args.quiet:
        action = "Would update" if args.dry_run else "Updated"
        print(f"{action} {len(modified)} policy file(s).")

    return 0 if modified else 1


def main(argv=None) -> int:
    """Standalone CLI entry point with ``ms`` / ``ri`` subcommands.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        prog="python -m logic.controllers.cli.tgt_parser",
        description=(
            "Batch-update policy YAML config overrides for WSmart+ Route.\n\n"
            "  ms  — update mandatory-selection strategy overrides\n"
            "  ri  — update route-improver overrides"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    ms_parser = subparsers.add_parser(
        "ms",
        help="Update mandatory-selection strategy overrides",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_ms_update_args(ms_parser)

    ri_parser = subparsers.add_parser(
        "ri",
        help="Update route-improver overrides",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_ri_update_args(ri_parser)

    args = parser.parse_args(argv)

    if args.subcommand == "ms":
        return _run_ms_from_namespace(args)
    return _run_ri_from_namespace(args)


if __name__ == "__main__":
    sys.exit(main())
