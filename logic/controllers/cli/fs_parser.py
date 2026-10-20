"""

Parser helpers live in ``fs_parser_args``; this module keeps the operations and re-exports those names.
Parser arguments and execution logic for File System operations.

Attributes:
    add_files_args: Adds all arguments related to file system operations (as subparsers)
    add_files_update_args: Adds file system 'update' sub-command arguments
    add_files_delete_args: Adds file system 'delete' sub-command arguments
    add_files_crypto_args: Adds file system 'cryptography' sub-command arguments
    validate_file_system_args: Validates and post-processes arguments for file_system
    perform_cryptographic_operations: Performs cryptographic tasks.
    update_file_system_entries: Applies content updates or statistical processing to files/directories.
    delete_file_system_entries: Batch deletes project-related directories.

Example:
    >>> from logic.controllers.cli.fs_parser import add_files_args, validate_file_system_args
    >>> parser = argparse.ArgumentParser()
    >>> add_files_args(parser)
    >>> validate_file_system_args(parser.parse_args())
    ('update', {'target_entry': None, 'output_key': None, 'filename_pattern': None, 'update_operation': None, 'update_value': 0.0, 'update_preview': False, 'stats_function': None, 'output_filename': None, 'input_keys': (None, None)})

    python -m logic.controllers.cli.fs_parser delete --delete_all
"""

import argparse
import os
import shutil
import sys
import traceback
from typing import Any, Dict

from logic.controllers.cli.base import ConfigsParser
from logic.src.constants import ROOT_DIR
from logic.src.utils.input.file_processing import process_file, process_pattern_files
from logic.src.utils.input.files import confirm_proceed
from logic.src.utils.input.preview import (
    preview_changes,
    preview_file_changes,
    preview_file_statistics,
    preview_pattern_files_statistics,
)
from logic.src.utils.input.statistics import (
    process_file_statistics,
    process_pattern_files_statistics,
)
from logic.src.utils.security import (
    decrypt_file_data,
    encrypt_file_data,
    generate_key,
    load_key,
)

from .fs_parser_args import (
    add_files_args as add_files_args,
)
from .fs_parser_args import (
    add_files_crypto_args as add_files_crypto_args,
)
from .fs_parser_args import (
    add_files_delete_args as add_files_delete_args,
)
from .fs_parser_args import (
    add_files_update_args as add_files_update_args,
)
from .fs_parser_args import (
    validate_file_system_args as validate_file_system_args,
)


def perform_cryptographic_operations(opts: Dict[str, Any]) -> None:
    """
    Perform cryptographic tasks based on the provided options.

    Tasks include:
    - Key generation: If no input path is provided, generates a new symmetric key.
    - Encryption: Encrypts a file using a stored key and verifies the result.

    Args:
        opts (Dict[str, Any]): Configuration options containing paths, key names,
            and encryption parameters.

    Raises:
        Exception: If any cryptographic operation fails.
    """
    try:
        if "input_path" not in opts or opts["input_path"] is None:
            _, _ = generate_key(
                opts["salt_size"],
                opts["key_length"],
                opts["hash_iterations"],
                opts["symkey_name"],
                opts["env_file"],
            )
        else:
            key = load_key(opts["symkey_name"], opts["env_file"])
            inpath = os.path.join(ROOT_DIR, opts["input_path"])
            outpath = (
                os.path.join(ROOT_DIR, opts["output_path"])
                if "output_path" in opts and opts["output_path"]
                else os.path.join(ROOT_DIR, opts["input_path"] + ".enc")
            )
            _ = encrypt_file_data(key, inpath, outpath)
            dec_data = decrypt_file_data(key, outpath)
            with open(inpath, "r") as gp_file:
                data = gp_file.read()
            assert dec_data == data
    except Exception as e:
        raise Exception(f"failed to perform cryptographic operations due to {repr(e)}") from e


def update_file_system_entries(opts: Dict[str, Any]) -> int:
    """
    Apply content updates or statistical processing to files/directories.

    Logic:
    1. Identifies the target entry (file or directory).
    2. Optional preview: Shows changes/statistics and asks for confirmation.
    3. Execution: Processes files matching a pattern or a specific target path.

    Args:
        opts (Dict[str, Any]): Configuration options containing target paths,
            filename patterns, update operations, and preview flags.

    Returns:
        int: 1 if successful, 0 if cancelled by user.

    Raises:
        ValueError: If target path does not exist.
        Exception: If file processing fails.
    """
    try:
        target_path = os.path.join(ROOT_DIR, opts["target_entry"])
        process_stats = "stats_function" in opts and opts["stats_function"] is not None
        if os.path.isdir(target_path):
            if opts["update_preview"]:
                if process_stats:
                    preview_pattern_files_statistics(
                        target_path,
                        opts["filename_pattern"],
                        opts["output_filename"],
                        opts["output_key"],
                        opts["stats_function"],
                    )
                else:
                    preview_changes(
                        target_path,
                        opts["output_key"],
                        opts["filename_pattern"],
                        opts["update_operation"],
                        opts["update_value"],
                        opts["input_keys"],
                    )
                if not confirm_proceed(operation_name="directory update"):
                    print("Operation cancelled by user.")
                    return 0
            if process_stats:
                process_pattern_files_statistics(
                    target_path,
                    opts["filename_pattern"],
                    opts["output_filename"],
                    opts["output_key"],
                    opts["stats_function"],
                )
            else:
                process_pattern_files(
                    target_path,
                    opts["filename_pattern"],
                    opts["output_key"],
                    opts["update_operation"],
                    opts["update_value"],
                    opts["input_keys"],
                )
        elif os.path.isfile(target_path):
            if opts["update_preview"]:
                if process_stats:
                    preview_file_statistics(
                        target_path,
                        opts["output_filename"],
                        opts["output_key"],
                        opts["stats_function"],
                    )
                else:
                    preview_file_changes(
                        target_path,
                        opts["output_key"],
                        opts["update_operation"],
                        opts["update_value"],
                        opts["input_keys"],
                    )
                if not confirm_proceed(operation_name="file update"):
                    print("Operation cancelled by user.")
                    return 0
            if process_stats:
                process_file_statistics(
                    target_path,
                    opts["output_filename"],
                    opts["output_key"],
                    opts["stats_function"],
                )
            else:
                process_file(
                    target_path,
                    opts["output_key"],
                    opts["update_operation"],
                    opts["update_value"],
                    opts["input_keys"],
                )
        else:
            raise ValueError(f"no file or directory found with path '{target_path}'")
        return 1
    except Exception as e:
        raise Exception(f"failed to update file system entries due to {repr(e)}") from e


def delete_file_system_entries(opts: Dict[str, Any]) -> int:
    """
    Batch delete project-related directories based on provided flags.

    Supported categories:
    - WandB logs, training logs, model outputs.
    - Datasets, evaluation results, test sim outputs/checkpoints.
    - Local cache directories.

    Args:
        opts (Dict[str, Any]): Configuration options containing boolean flags for
            each directory category and directory names.

    Returns:
        int: 0 after completion or cancellation.

    Raises:
        Exception: If listing or deletion operations fail.
    """
    try:
        directories_to_delete = _collect_directories_to_delete(opts)

        if not directories_to_delete:
            print("No directories exist for deletion based on the provided options.")
            return 0

        if opts.get("delete_preview"):
            print("\nThe following directories exist and will be deleted:")
            print("-" * 60)
            for i, (description, _) in enumerate(directories_to_delete, 1):
                print(f"{i}. {description}:")

            if not confirm_proceed(operation_name="deletion"):
                print("Deletion cancelled by user.")
                return 0

        print("\nDeleting directories...")
        success_count = 0
        for description, path in directories_to_delete:
            try:
                shutil.rmtree(path, ignore_errors=True)
                if not os.path.exists(path):
                    print(f"- Successfully deleted: {description}")
                    success_count += 1
                else:
                    print(f"- Failed to delete: {description}")
            except Exception as e:
                print(f"- Error deleting {description}: {e}")

        print(f"\nDeletion completed: {success_count}/{len(directories_to_delete)} directories removed successfully.")
        return 0
    except Exception as e:
        raise Exception(f"failed to delete file system entries due to {repr(e)}") from e


def _collect_directories_to_delete(opts: Dict[str, Any]):
    """Helper to gather existing directories for deletion.

    Args:
        opts: Configuration options with boolean flags per directory category.

    Returns:
        List of (description, absolute_path) tuples for directories that exist.
    """
    to_delete = []

    # Standard logs and outputs
    mappings = {
        "wandb": ("wandb", "wandb logs"),
        "log": (opts.get("log_dir"), "train logs"),
        "output": (opts.get("output_dir"), "model outputs"),
        "data": (opts.get("data_dir"), "datasets"),
        "eval": (opts.get("eval_dir"), "evaluation results"),
    }

    for key, (dname, desc) in mappings.items():
        if opts.get(key) and dname:
            path = os.path.join(ROOT_DIR, dname)
            if os.path.exists(path):
                to_delete.append((desc, path))

    # Assets-based paths
    assets_mappings = {
        "test_sim": (opts.get("test_sim_dir"), "test sim outputs"),
        "test_checkpoint": (opts.get("test_sim_checkpoint_dir"), "test sim checkpoints"),
    }
    for key, (dname, desc) in assets_mappings.items():
        if opts.get(key) and dname:
            path = os.path.join(ROOT_DIR, "assets", dname)
            if os.path.exists(path):
                to_delete.append((desc, path))

    # Cache handling
    if opts.get("cache"):
        for cdir, desc in [("cache", "main cache"), (os.path.join("notebooks", "cache"), "notebooks cache")]:
            path = os.path.join(ROOT_DIR, cdir)
            if os.path.exists(path):
                to_delete.append((desc, path))

    return to_delete


if __name__ == "__main__":
    exit_code = 0
    _parser = ConfigsParser(
        description="File System Utility Runner (update/delete/cryptography)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    add_files_args(_parser)
    try:
        parsed_args = _parser.parse_process_args(sys.argv[1:], "file_system")
        comm, opts = validate_file_system_args(parsed_args) # pyrefly: ignore [bad-argument-type]
        if comm == "update":
            update_file_system_entries(opts)
        elif comm == "delete":
            delete_file_system_entries(opts)
        else:
            assert comm == "cryptography"
            perform_cryptographic_operations(opts)

    except (argparse.ArgumentError, AssertionError) as e:
        exit_code = 1
        _parser.print_help()
        print(f"Error: {e}", file=sys.stderr)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(str(e), file=sys.stderr)
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(exit_code)
