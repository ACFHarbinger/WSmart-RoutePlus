"""
Directory and Zip encryption utilities.

This module provides functions to encrypt and decrypt directories and zip files.

Attributes:
    encrypt_directory: Encrypt all files in a directory recursively.
    decrypt_directory: Decrypt all .enc files in a directory recursively.
    encrypt_zip_directory: Zip a directory and then encrypt the resulting zip file.
    decrypt_zip: Decrypt a zip file and extract its contents.

Example:
    >>> import directories
    >>> directories.encrypt_directory(key, "test")
    >>> directories.decrypt_directory(key, "test")
    >>> directories.encrypt_zip_directory(key, "test")
    >>> directories.decrypt_zip(key, "test")
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union

from logic.src.utils.input.files import extract_zip, zip_directory

from .data import decrypt_file_data, encrypt_file_data


def _validate_input_directory(input_dir: Union[str, os.PathLike]) -> Path:
    """Resolve an existing source directory before any output is created."""
    source = Path(input_dir).resolve(strict=True)
    if not source.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    return source


def encrypt_directory(
    key: bytes, input_dir: Union[str, os.PathLike], output_dir: Optional[Union[str, os.PathLike]] = None
) -> List[bytes]:
    """
    Encrypt all files in a directory recursively.

    Args:
        key (bytes): The encryption key.
        input_dir (Union[str, os.PathLike]): Directory to encrypt.
        output_dir (Union[str, os.PathLike], optional): Output directory. Defaults to input_dir.

    Returns:
        list: List of encrypted data bytes for each file.

    Raises:
        Exception: If directory creation fails.
        ValueError: If paths escape the roots or output is nested inside input.
        FileNotFoundError: If the input directory does not exist.
        NotADirectoryError: If the input path is not a directory.
    """
    if output_dir is None:
        output_dir = input_dir

    # Resolve paths to prevent symlink attacks
    input_dir_resolved = _validate_input_directory(input_dir)
    output_dir_resolved = Path(output_dir).resolve()
    if input_dir_resolved in output_dir_resolved.parents:
        raise ValueError("Output directory must not be inside input directory")

    try:
        os.makedirs(output_dir_resolved, exist_ok=True)
    except OSError as e:
        raise Exception(f"Failed to create output directory: {e}") from e

    # Recursively process all files in the input directory
    encdata_ls = []
    for root, _, files in os.walk(str(input_dir_resolved), followlinks=False):
        for file in files:
            input_file = Path(root) / file
            input_file_resolved = input_file.resolve()

            # Check if the file is a symlink pointing outside the input directory
            if input_file.is_symlink() and input_dir_resolved not in input_file_resolved.parents:
                raise ValueError(
                    f"Symlink {input_file} points outside input directory: {input_file_resolved}"
                )

            relative_path = input_file_resolved.relative_to(input_dir_resolved)
            output_file = (output_dir_resolved / (str(relative_path) + ".enc")).resolve()
            if output_dir_resolved not in output_file.parents:
                raise ValueError(f"Output path {output_file} is outside output directory: {output_dir_resolved}")

            # Create subdirectories in the output directory if they don't exist
            try:
                os.makedirs(output_file.parent, exist_ok=True)
            except OSError as e:
                raise Exception(f"Failed to create subdirectory: {e}") from e
            encdata_ls.append(encrypt_file_data(key, str(input_file_resolved), str(output_file)))
    return encdata_ls


def decrypt_directory(
    key: bytes, input_dir: Union[str, os.PathLike], output_dir: Optional[Union[str, os.PathLike]] = None
) -> List[Union[str, bytes]]:
    """
    Decrypt all .enc files in a directory recursively.

    Args:
        key (bytes): The encryption key.
        input_dir (Union[str, os.PathLike]): Directory to decrypt.
        output_dir (Union[str, os.PathLike], optional): Output directory. Defaults to input_dir.

    Returns:
        list: Decrypted strings for valid UTF-8 payloads, otherwise bytes.

    Raises:
        Exception: If directory creation fails.
        ValueError: If paths escape the roots or output is nested inside input.
        FileNotFoundError: If the input directory does not exist.
        NotADirectoryError: If the input path is not a directory.
    """
    if output_dir is None:
        output_dir = input_dir

    # Resolve paths to prevent symlink and path traversal attacks
    input_dir_resolved = _validate_input_directory(input_dir)
    output_dir_resolved = Path(output_dir).resolve()
    if input_dir_resolved in output_dir_resolved.parents:
        raise ValueError("Output directory must not be inside input directory")

    try:
        os.makedirs(output_dir_resolved, exist_ok=True)
    except OSError as e:
        raise Exception(f"Failed to create output directory: {e}") from e

    # Recursively process all files in the input directory
    decdata_ls = []
    for root, _, files in os.walk(str(input_dir_resolved), followlinks=False):
        for file in files:
            input_file = Path(root) / file
            input_file_resolved = input_file.resolve()

            # Check if the file is a symlink pointing outside the input directory
            if input_file.is_symlink() and input_dir_resolved not in input_file_resolved.parents:
                raise ValueError(
                    f"Symlink {input_file} points outside input directory: {input_file_resolved}"
                )

            file_path, file_ext = os.path.splitext(str(input_file_resolved))
            if file_ext == ".enc":
                relative_path = Path(file_path).relative_to(input_dir_resolved)
                output_file = output_dir_resolved / relative_path

                # Validate that output file is within output directory (prevent path traversal)
                output_file_resolved = output_file.resolve()
                if output_dir_resolved not in output_file_resolved.parents:
                    raise ValueError(
                        f"Output path {output_file} is outside output directory: {output_dir_resolved}"
                    )

                # Create subdirectories in the output directory if they don't exist
                try:
                    os.makedirs(output_file_resolved.parent, exist_ok=True)
                except OSError as e:
                    raise Exception(f"Failed to create subdirectory: {e}") from e
                decdata_ls.append(decrypt_file_data(key, str(input_file_resolved), str(output_file_resolved)))
    return decdata_ls


def encrypt_zip_directory(
    key: bytes, input_dir: Union[str, os.PathLike], output_enczip: Optional[Union[str, os.PathLike]] = None
) -> bytes:
    """
    Zip a directory and then encrypt the resulting zip file.

    Args:
        key (bytes): The encryption key.
        input_dir (Union[str, os.PathLike]): Directory to zip and encrypt.
        output_enczip (Union[str, os.PathLike], optional): Output path for the encrypted zip.

    Returns:
        bytes: The encrypted zip data.

    Raises:
        FileNotFoundError: If the input directory does not exist.
        NotADirectoryError: If the input path is not a directory.
    """
    _validate_input_directory(input_dir)
    input_dir_str = str(input_dir)
    if output_enczip is None:
        norm_path = os.path.normpath(input_dir_str)
        output_enczip = os.path.join(os.path.dirname(norm_path), f"{os.path.basename(norm_path)}.zip")

    output_enczip_str = str(output_enczip)

    # Use secure temporary file
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip_file:
        tmp_zip = tmp_zip_file.name

    try:
        zip_directory(input_dir_str, tmp_zip)
        encrypted_data = encrypt_file_data(key, tmp_zip, output_enczip_str)
    finally:
        # Ensure cleanup even if encryption fails
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)

    return encrypted_data


def decrypt_zip(
    key: bytes, input_enczip: Union[str, os.PathLike], output_dir: Optional[Union[str, os.PathLike]] = None
) -> Union[str, bytes]:
    """
    Decrypt a zip file and extract its contents.

    Args:
        key (bytes): The encryption key.
        input_enczip (Union[str, os.PathLike]): Path to the encrypted zip file.
        output_dir (Union[str, os.PathLike], optional): Directory to extract to.

    Returns:
        Union[str, bytes]: Decrypted ZIP payload (string if valid UTF-8, otherwise bytes).
    """

    input_enczip_str = str(input_enczip)
    if output_dir is None:
        norm_path = os.path.normpath(input_enczip_str)
        output_dir, _ = os.path.splitext(norm_path)

    output_dir_str = str(output_dir)

    # Use secure temporary file
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip_file:
        tmp_zip = tmp_zip_file.name

    try:
        decrypted_data = decrypt_file_data(key, input_enczip_str, tmp_zip)
        extract_zip(tmp_zip, output_dir_str)
    finally:
        # Ensure cleanup even if extraction fails
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)

    return decrypted_data
