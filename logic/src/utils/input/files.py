"""
Safe file operation utilities (read, write, zip).

Attributes:
    read_json: Reads a JSON file safely with an optional lock.
    zip_directory: Creates a zip archive of a directory.
    extract_zip: Extracts a zip archive to a directory.
    confirm_proceed: Asks user for confirmation with timeout and default response.
    compose_dirpath: Decorator to compose directory paths based on home_dir, ndays, num_bins, etc.

Example:
    >>> from logic.src.utils.io.files import read_json, zip_directory, extract_zip, confirm_proceed, compose_dirpath
    >>> json_data = read_json("path/to/file.json")
    >>> zip_directory("path/to/directory", "path/to/archive.zip")
    >>> extract_zip("path/to/archive.zip", "path/to/output")
    >>> proceed = confirm_proceed()
    >>> @compose_dirpath
    ... def my_function(dir_path):
    ...     # Process files in dir_path
    ...     pass
"""

from __future__ import annotations

import json
import os
import signal
import threading
import zipfile
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

import logic.src.constants as udef

T = TypeVar("T", bound=Callable[..., Any])


def read_json(json_path: str, lock: Optional[threading.Lock] = None) -> Union[Dict[str, Any], List[Any]]:
    """
    Reads a JSON file safely with an optional lock.

    Args:
        json_path: Path to the JSON file.
        lock: Thread lock for safe access.

    Returns:
        The parsed JSON data.
    """
    if lock is not None:
        lock.acquire(timeout=udef.LOCK_TIMEOUT)
    try:
        with open(json_path, "r", encoding="utf-8") as json_file:
            json_data: Union[Dict[str, Any], List[Any]] = json.load(json_file)
    finally:
        if lock is not None:
            lock.release()
    return json_data


def zip_directory(input_dir: str, output_zip: str) -> None:
    """
    Create a zip archive of a directory.

    Args:
        input_dir: Path to the directory to zip.
        output_zip: Path to save the zip archive to.
    """
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(input_dir):
            for file in files:
                file_path: str = os.path.join(root, file)
                arcname: str = os.path.relpath(file_path, input_dir)
                zipf.write(file_path, arcname)
    print(f"Directory '{input_dir}' zipped to '{output_zip}'")


def extract_zip(input_zip: str, output_dir: str) -> None:
    """
    Extracts a zip archive to a directory.

    Args:
        input_zip: Path to the zip archive.
        output_dir: Directory where contents will be extracted.
    """
    with zipfile.ZipFile(input_zip, "r") as zipf:
        zipf.extractall(output_dir)
    print(f"Zip archive '{input_zip}' was extracted to directory '{output_dir}'")


def confirm_proceed(default_no: bool = True, operation_name: str = "update") -> bool:
    """
    Ask user for confirmation with timeout and default response
    Returns True if user confirms, False if user cancels or timeout

    Args:
        default_no: Default response if user doesn't provide one (default: True).
        operation_name: Name of the operation to confirm (default: "update").

    Returns:
        True if user confirms, False otherwise.
    """

    def timeout_handler(signum: int, frame: Any) -> None:
        """
        Raises TimeoutError on signal.

        Args:
            signum: Signal number.
            frame: Current stack frame.
        """
        raise TimeoutError()

    try:
        # Check if signal.SIGALRM is available (Unix systems)
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(udef.CONFIRM_TIMEOUT)
            try:
                prompt: str = f"\nProceed with {operation_name}? (y/n) [{'n' if default_no else 'y'}] (timeout {udef.CONFIRM_TIMEOUT}s): "
                response: str = input(prompt).strip().lower()
                signal.alarm(0)
                if response in ["y", "yes"]:
                    return True
                elif response in ["n", "no"]:
                    return False
                else:
                    return not default_no
            except TimeoutError:
                print(f"\nTimeout reached. Defaulting to {'no' if default_no else 'yes'}.")
                return not default_no
            except KeyboardInterrupt:
                print(f"\nOperation interrupted by user. Defaulting to {'no' if default_no else 'yes'}.")
                return not default_no
        else:
            # Fallback for Windows
            prompt = f"\nProceed with {operation_name}? (y/n) [{'n' if default_no else 'y'}]: "
            response = input(prompt).strip().lower()
            if response in ["y", "yes"]:
                return True
            elif response in ["n", "no"]:
                return False
            return not default_no
    except Exception:
        return not default_no


def compose_dirpath(fun: T) -> T:
    """
    Decorator to compose directory paths based on home_dir, ndays, num_bins, etc.

    The full path produced is::

        <home_dir>/<output_dir>/<ndays>days/<area><nbins>_<waste_type>[/<data_distribution>][/<run_name>]

    The ``data_distribution`` and ``run_name`` components are appended only when
    they are non-empty strings.

    Args:
        fun: Function to decorate.

    Returns:
        Decorated function.
    """

    def inner(
        home_dir: str,
        ndays: int,
        nbins: Union[int, List[int]],
        output_dir: str,
        area: str,
        waste_type: str = "",
        data_distribution: str = "",
        run_name: str = "",
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Inner wrapper function.

        Args:
            home_dir: Home directory.
            ndays: Number of days.
            nbins: Number of bins (int or list of ints).
            output_dir: Output directory, relative to the repository root (or absolute).
            area: Geographic area identifier.
            waste_type: Waste type suffix for directory naming. Defaults to "".
            data_distribution: Data distribution sub-directory (e.g. ``"gamma3"``). Defaults to "".
            run_name: Run name sub-directory for experiment isolation. Defaults to "".
            *args: Additional positional arguments forwarded to *fun*.
            **kwargs: Additional keyword arguments forwarded to *fun*.

        Returns:
            Result of the decorated function.
        """

        def _build(gs: Union[int, str]) -> str:
            base = os.path.join(home_dir, output_dir, f"{ndays}days", f"{area}{gs}_{waste_type}")
            extras = [d for d in (data_distribution, run_name) if d]
            return os.path.join(base, *extras) if extras else base

        if not isinstance(nbins, Iterable):
            return fun(_build(nbins), *args, **kwargs)

        return fun([_build(gs) for gs in nbins], *args, **kwargs)

    return cast(T, inner)
