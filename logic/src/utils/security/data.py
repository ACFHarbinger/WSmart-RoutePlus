"""
Data encryption/decryption utilities.

This module provides functions to encode, encrypt, and decrypt data.

Attributes:
    encode_data: Encodes various data types into bytes for encryption.
    encrypt_file_data: Encrypt a file or data object using Fernet symmetric encryption.
    decrypt_file_data: Decrypt a file or data bytes using Fernet symmetric encryption.

Example:
    >>> import data
    >>> data.encode_data("test")
    >>> data.encrypt_file_data(key, "test")
    >>> data.decrypt_file_data(key, "test")
"""

import os
import pickle
import struct
from pathlib import Path
from typing import Any, Optional, Union

from cryptography.fernet import Fernet


def encode_data(data: Any) -> bytes:
    """
    Encodes various data types into bytes for encryption.

    Args:
        data (Any): Data to encode (str, int, float, bytes, list, dict, etc.).

    Returns:
        bytes: Encoded byte representation.

    Note:
        This function encodes data for encryption. decrypt_file_data() returns a string
        if the decrypted bytes are valid UTF-8, otherwise returns raw bytes. This allows
        recovery of the encoded bytes; it does not reconstruct the original Python
        type. Single-precision float encoding may lose precision.
    """
    if isinstance(data, str):
        return data.encode("utf-8")
    elif isinstance(data, bytes):
        # Raw bytes are returned as-is
        return data
    elif isinstance(data, bool):
        # Handle bool before int (bool is a subclass of int in Python)
        return b"\x01" if data else b"\x00"
    elif isinstance(data, int):
        # Handle zero and negative integers
        if data == 0:
            return b"\x00"
        elif data > 0:
            # Positive integer: use unsigned representation
            return data.to_bytes((data.bit_length() + 7) // 8, byteorder="big", signed=False)
        else:
            # Negative integer: use signed representation
            # Calculate the number of bytes needed for signed representation
            byte_length = (data.bit_length() + 8) // 8  # +8 for sign bit
            return data.to_bytes(byte_length, byteorder="big", signed=True)
    elif isinstance(data, float):
        # Use single precision (4 bytes) for backward compatibility
        # Note: This loses precision for double-precision floats
        return struct.pack("!f", data)
    else:  # elif isinstance(data, list) or isinstance(data, ITraversable):
        return pickle.dumps(data)


def encrypt_file_data(
    key: bytes, input: Union[str, os.PathLike, Any], output_file: Optional[Union[str, os.PathLike]] = None
) -> bytes:
    """
    Encrypt a file or data object using Fernet symmetric encryption.

    Args:
        key (bytes): The encryption key.
        input (Union[str, os.PathLike, Any]): Path to file OR data object to encrypt.
        output_file (Union[str, os.PathLike], optional): Path to save encrypted data.

    Returns:
        bytes: The encrypted data.
    """
    fernet = Fernet(key)
    # Check if input looks like a path and exists
    if isinstance(input, (str, Path)) and os.path.isfile(input):
        with open(input, "rb") as f:
            original_data = f.read()
    else:
        original_data = encode_data(input)

    encrypted_data = fernet.encrypt(original_data)
    if output_file:
        with open(output_file, "wb") as f:
            f.write(encrypted_data)
    return encrypted_data


def decrypt_file_data(
    key: bytes, input: Union[str, os.PathLike, Any], output_file: Optional[Union[str, os.PathLike]] = None
) -> Union[str, bytes]:
    """
    Decrypt a file or data bytes using Fernet symmetric encryption.

    Args:
        key (bytes): The encryption key.
        input (Union[str, os.PathLike, Any]): Path to encrypted file OR encrypted bytes.
        output_file (Union[str, os.PathLike], optional): Path to save decrypted content.

    Returns:
        Union[str, bytes]: The decrypted data. Returns a string if the decrypted bytes
            are valid UTF-8, otherwise returns raw bytes. This allows round-trip
            encryption/decryption of payloads, not original Python object types.
            If output_file is supplied, its contents are the exact plaintext bytes.
    """
    fernet = Fernet(key)
    if isinstance(input, (str, Path)) and os.path.isfile(input):
        with open(input, "rb") as f:
            encrypted_data = f.read()
    else:
        # Cast input to bytes if it's not a file path
        if not isinstance(input, bytes):
            raise TypeError(f"Expected file path or bytes for decryption, got {type(input)}")
        encrypted_data = input

    decrypted_bytes = fernet.decrypt(encrypted_data)

    # Try to decode as UTF-8 for backward compatibility with text data
    # If decoding fails, return raw bytes for binary data
    try:
        decrypted_data = decrypted_bytes.decode("utf-8")
    except UnicodeDecodeError:
        decrypted_data = decrypted_bytes

    if output_file:
        # Preserve the authenticated plaintext bytes regardless of locale/newlines.
        with open(output_file, "wb") as f:
            f.write(decrypted_bytes)
    return decrypted_data
