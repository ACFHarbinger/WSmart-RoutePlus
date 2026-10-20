"""Argument-parser helpers for file-system CLI commands.

Moved from ``logic.controllers.cli.fs_parser``.
"""

import argparse
from typing import Any, Dict, Tuple

from logic.controllers.cli.base import UpdateFunctionMapActionFactory
from logic.src.constants import FS_COMMANDS


def add_files_args(parser: Any) -> Any:
    """
    Adds all arguments related to file system operations (as subparsers).

    Args:
        parser: The argparse parser or subparser.

    Returns:
        The parser with added file system arguments.
    """
    files_subparsers = parser.add_subparsers(help="file system command", dest="fs_command", required=True)

    # Update file system entries
    update_parser = files_subparsers.add_parser("update", help="Update file system entries")
    add_files_update_args(update_parser)

    # Delete file system entries
    delete_parser = files_subparsers.add_parser("delete", help="Delete file system entries")
    add_files_delete_args(delete_parser)

    # Cryptography
    crypto_parser = files_subparsers.add_parser(
        "cryptography", help="Perform cryptographic operations on file system entries"
    )
    add_files_crypto_args(crypto_parser)
    return parser


def add_files_update_args(parser: Any) -> Any:
    """
    Adds file system 'update' sub-command arguments.

    Args:
        parser: The argparse parser or subparser.

    Returns:
        The parser with added update arguments.
    """
    parser.add_argument(
        "--target_entry",
        type=str,
        help="Path to the file to the file system entry we want to update",
    )
    parser.add_argument(
        "--output_key",
        type=str,
        default=None,
        help="Key of the values we want to update in the files",
    )
    parser.add_argument(
        "--filename_pattern",
        type=str,
        default=None,
        help="Pattern to match names of files to update (target_entry must be directory)",
    )
    parser.add_argument(
        "--update_operation",
        type=str,
        default=None,
        action=UpdateFunctionMapActionFactory(inplace=True),
        help="Operation to update the file values",
    )
    parser.add_argument("--update_value", type=float, default=0.0, help="Value for the update operation")
    parser.add_argument(
        "--update_preview",
        action="store_true",
        help="Preview how files/directories will look like after the update",
    )
    parser.add_argument(
        "--stats_function",
        type=str,
        default=None,
        action=UpdateFunctionMapActionFactory(),
        help="Function to perform over the file values",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        help="Name of the file we want to save the output values to",
    )
    parser.add_argument(
        "--input_keys",
        type=str,
        default=(None, None),
        nargs="*",
        help="Key(s) of the values we want to use as input to update the other key in the files",
    )
    return parser


def add_files_delete_args(parser: Any) -> Any:
    """
    Adds file system 'delete' sub-command arguments.

    Args:
        parser: The argparse parser or subparser.

    Returns:
        The parser with added delete arguments.
    """
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Flag to delete the train wandb log directory",
    )
    parser.add_argument("--log_dir", default="logs", help="Directory of train logs")
    parser.add_argument("--log", action="store_true", help="Flag to delete the train log directory")
    parser.add_argument(
        "--output_dir",
        default="model_weights",
        help="Directory to write output models to",
    )
    parser.add_argument(
        "--output",
        action="store_true",
        help="Flag to delete the train output models directory",
    )
    parser.add_argument("--data_dir", default="datasets", help="Directory of generated datasets")
    parser.add_argument("--data", action="store_true", help="Flag to delete the datasets directory")
    parser.add_argument("--eval_dir", default="results", help="Name of the evaluation results directory")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Flag to delete the evaluation results directory",
    )
    parser.add_argument(
        "--test_dir",
        default="output",
        help="Name of the WSR simulator test output directory",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Flag to delete the WSR simulator test output directory",
    )
    parser.add_argument(
        "--test_checkpoint_dir",
        default="temp",
        help="Name of WSR simulator test runs checkpoint directory",
    )
    parser.add_argument(
        "--test_checkpoint",
        action="store_true",
        help="Flag to delete the WSR simulator test runs checkpoint directory",
    )
    parser.add_argument("--cache", action="store_true", help="Flag to delete the cache directories")
    parser.add_argument(
        "--delete_preview",
        action="store_true",
        help="Preview which files/directories will be removed",
    )
    return parser


def add_files_crypto_args(parser: Any) -> Any:
    """
    Adds file system 'cryptography' sub-command arguments.

    Args:
        parser: The argparse parser or subparser.

    Returns:
        The parser with added cryptography arguments.
    """
    parser.add_argument(
        "--symkey_name",
        type=str,
        default=None,
        help="Name of the key for the files to save the salt and key/hash parameters to",
    )
    parser.add_argument(
        "--env_file",
        type=str,
        default="vars.env",
        help="Name of the file that contains the environment variables",
    )
    parser.add_argument("--salt_size", type=int, default=16)
    parser.add_argument("--key_length", type=int, default=32)
    parser.add_argument("--hash_iterations", type=int, default=100_000)
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    return parser


def validate_file_system_args(args: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Validates and post-processes arguments for file_system.
    Returns a tuple: (fs_command, validated_args)

    Args:
        args: Dictionary of file system arguments

    Returns:
        Tuple of (fs_command, validated_args)
    """
    args = args.copy()
    fs_comm = args.pop("fs_command", None)
    if fs_comm not in FS_COMMANDS:
        raise argparse.ArgumentError(None, "ERROR: unknown File System (inner) command " + str(fs_comm))

    assert not ("stats_function" in args and args["stats_function"] is not None) or not (
        "update_operation" in args and args["update_operation"] is not None
    ), "'update_operation' and 'stats_function' arguments are mutually exclusive"

    return fs_comm, args
