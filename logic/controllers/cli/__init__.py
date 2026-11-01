"""
Unified entry point for the WSmart Route+ modular CLI.

Attributes:
    parse_params: Parses arguments, determines the command, and performs necessary validation.

Example:
    >>> from logic.controllers.cli import parse_params
    >>> command, opts = parse_params()
    >>> command
    'test_suite'
    >>> opts
    {'verbose': True, ...}
"""

import argparse
from typing import Any, Dict, Tuple, Union

from logic.controllers.cli.base.registry import get_parser
from logic.controllers.cli.bench_parser import validate_benchmark_args
from logic.controllers.cli.fs_parser import add_files_args, validate_file_system_args
from logic.controllers.cli.out_parser import add_excel_summary_args, add_output_args
from logic.controllers.cli.tgt_parser import add_ms_update_args, add_ri_update_args
from logic.controllers.cli.ts_parser import add_test_suite_args, validate_test_suite_args

from .base import (
    ConfigsParser,
    LowercaseAction,
    StoreDictKeyPair,
    UpdateFunctionMapActionFactory,
)


def parse_params() -> Tuple[Union[str, Tuple[str, str]], Dict[str, Any]]:
    """
    Parses arguments, determines the command, and performs necessary validation.

    Args:
        None

    Returns:
        (command, validated_opts) where 'command' might be a tuple (comm, inner_comm).
    """
    parser = get_parser()

    try:
        # Parse arguments into a dictionary using the custom handler
        command, opts = parser.parse_process_args()
        assert command is not None

        # --- COMMAND-SPECIFIC VALIDATION AND route improvement ---
        # "gen_data", "eval", "test_sim", "train" are now fully handled by Hydra or removed from legacy flow
        if command == "file_system":
            # This returns a tuple: (fs_command, validated_opts)
            inner_comm, opts = validate_file_system_args(opts)
            return ("file_system", inner_comm), opts
        elif command == "test_suite":
            opts = validate_test_suite_args(opts)
        elif command == "benchmark":
            opts = validate_benchmark_args(opts)
        return command, opts
    except (argparse.ArgumentError, AssertionError) as e:
        parser.error_message(f"Error: {e}", print_help=True)
    except Exception as e:
        parser.error_message(f"An unexpected error occurred: {e}", print_help=False)


__all__ = [
    "parse_params",
    "add_files_args",
    "add_test_suite_args",
    "add_ms_update_args",
    "add_ri_update_args",
    "add_output_args",
    "add_excel_summary_args",
    "ConfigsParser",
    "LowercaseAction",
    "StoreDictKeyPair",
    "UpdateFunctionMapActionFactory",
]
