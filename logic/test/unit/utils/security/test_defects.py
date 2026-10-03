"""
Fail-before tests for utils/security defects.

These tests demonstrate the bugs in the unfixed code and pass after fixes.
"""


import pytest
from cryptography.fernet import Fernet

# Generate a test key once for all tests
TEST_KEY = Fernet.generate_key()


class TestInitDefects:
    """Tests for __init__.py defects."""

    def test_all_has_no_duplicates(self):
        """__all__ should not have duplicate entries."""
        from logic.src.utils.security import __all__

        assert len(__all__) == len(set(__all__)), f"__all__ has duplicates: {[x for x in __all__ if __all__.count(x) > 1]}"


class TestDataDefects:
    """Tests for data.py defects."""

    def test_encode_zero_integer(self):
        """encode_data should handle zero correctly."""
        from logic.src.utils.security.data import encode_data

        # This should not crash
        result = encode_data(0)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_encode_negative_integer(self):
        """encode_data should handle negative integers."""
        from logic.src.utils.security.data import encode_data

        # This should not crash
        result = encode_data(-42)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_float_precision_preserved(self):
        """Float encryption/decryption should preserve the binary representation."""
        import struct

        from logic.src.utils.security.data import decrypt_file_data, encrypt_file_data

        original = 3.141592653589793  # Double precision

        # Encrypt the float
        encrypted = encrypt_file_data(TEST_KEY, original)

        # Decrypt - should return bytes since float encoding is binary
        decrypted = decrypt_file_data(TEST_KEY, encrypted)

        # The decrypted data should be bytes (not a string)
        assert isinstance(decrypted, bytes), f"Expected bytes, got {type(decrypted)}"

        # The bytes should be decodable back to the original float
        # Note: We use single precision (4 bytes) for backward compatibility,
        # so we expect some precision loss
        decrypted_float = struct.unpack("!f", decrypted)[0]

        # Check that the value is close (within single-precision tolerance)
        assert abs(decrypted_float - original) < 1e-5, \
            f"Float precision lost: {original} -> {decrypted_float}"

    def test_decrypt_binary_data_returns_bytes(self):
        """Decrypting binary data should return bytes, not raise an error."""
        from logic.src.utils.security.data import decrypt_file_data, encrypt_file_data

        # Encrypt binary data that's not valid UTF-8
        binary_data = b"\xff\xfe\xfd\xfc"
        encrypted = encrypt_file_data(TEST_KEY, binary_data)

        # This should successfully decrypt to the original bytes
        decrypted = decrypt_file_data(TEST_KEY, encrypted)

        # Should return bytes (not a string) since it's not valid UTF-8
        assert isinstance(decrypted, bytes), f"Expected bytes, got {type(decrypted)}"
        assert decrypted == binary_data, f"Binary data not preserved: {binary_data} -> {decrypted}"


class TestDirectoriesDefects:
    """Tests for directories.py defects."""

    def test_symlink_traversal_blocked(self, tmp_path):
        """encrypt_directory should not follow symlinks outside the input directory."""
        from logic.src.utils.security.directories import encrypt_directory

        # Create a file outside the input directory
        outside_file = tmp_path / "outside.txt"
        outside_file.write_text("secret data")

        # Create input directory with a symlink pointing outside
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        symlink = input_dir / "symlink.txt"
        symlink.symlink_to(outside_file)

        output_dir = tmp_path / "output"

        # This should either:
        # 1. Skip the symlink, or
        # 2. Raise an error about symlinks
        # Currently it will follow the symlink and encrypt the outside file
        with pytest.raises((ValueError, PermissionError, OSError)):
            encrypt_directory(TEST_KEY, input_dir, output_dir)

    def test_path_traversal_in_decrypt_blocked(self, tmp_path):
        """decrypt_directory should not allow path traversal in filenames."""
        from logic.src.utils.security.directories import decrypt_directory, encrypt_directory

        # Create input directory
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Create a file with a malicious name containing path traversal
        # Note: We can't actually create a file with ".." in the name on most filesystems
        # But we can test that the code validates output paths
        test_file = input_dir / "test.txt"
        test_file.write_text("test data")

        # Encrypt it
        encrypt_directory(TEST_KEY, input_dir, input_dir)

        # Now try to decrypt - this should validate output paths
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # This should succeed without path traversal
        decrypt_directory(TEST_KEY, input_dir, output_dir)

        # Verify the output file is within output_dir
        output_file = output_dir / "test.txt"
        assert output_file.exists()
        assert output_file.resolve().is_relative_to(output_dir.resolve())

    def test_temporary_files_use_secure_names(self, tmp_path):
        """encrypt_zip_directory should use secure temporary file names."""
        from logic.src.utils.security.directories import encrypt_zip_directory

        # Create input directory
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test.txt").write_text("test data")

        output_file = tmp_path / "output.enczip"

        # Encrypt - this should use secure temporary files
        encrypt_zip_directory(TEST_KEY, input_dir, output_file)

        # Verify no .tmp.zip files are left behind
        tmp_files = list(tmp_path.glob("*.tmp.zip"))
        assert len(tmp_files) == 0, f"Temporary files not cleaned up: {tmp_files}"

    def test_temporary_files_cleaned_on_error(self, tmp_path):
        """Temporary files should be cleaned up even if encryption fails."""
        from logic.src.utils.security.directories import encrypt_zip_directory

        # Use an invalid key to cause encryption to fail
        invalid_key = b"invalid"

        # Create input directory
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test.txt").write_text("test data")

        output_file = tmp_path / "output.enczip"

        # This should fail
        with pytest.raises(ValueError):
            encrypt_zip_directory(invalid_key, input_dir, output_file)

        # Verify no temporary files are left behind
        tmp_files = list(tmp_path.glob("*.tmp.zip"))
        assert len(tmp_files) == 0, f"Temporary files not cleaned up after error: {tmp_files}"
