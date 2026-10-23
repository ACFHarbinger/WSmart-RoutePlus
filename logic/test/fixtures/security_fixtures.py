"""
Fixtures for security and cryptography tests.
"""

import base64
import os
from unittest.mock import patch

import pytest


@pytest.fixture
def mock_crypto_env():
    """Mock environment variables for crypto tests."""
    with patch.dict(os.environ, {
        "KEY_PASSWORD": "test_password",
        "SALT_STRING": base64.urlsafe_b64encode(os.urandom(16)).decode("utf-8"),
        "KEY_LENGTH": "32",
        "HASH_ITERATIONS": "1000",
        "ROOT_DIR": "/tmp/root",
    }):
        yield

@pytest.fixture
def mock_crypto_dotenv():
    """Mock dotenv_values for crypto tests."""
    with patch("logic.src.utils.security.keys.dotenv_values") as mock:
        mock.return_value = {
            "KEY_PASSWORD": "test_password",
            "SALT_SIZE": "16",
            "KEY_LENGTH": "32",
            "HASH_ITERATIONS": "1000",
        }
        yield mock


@pytest.fixture
def make_symlink():
    """Create a symlink, or skip the test where the OS refuses to create one.

    Windows lets a standard user create symlinks only with Developer Mode on (WinError 1314).
    """

    def _make(link, target, target_is_directory=False):
        try:
            link.symlink_to(target, target_is_directory=target_is_directory)
        except OSError as error:
            if getattr(error, "winerror", None) == 1314:
                pytest.skip("creating symlinks needs Developer Mode or an elevated shell on Windows")
            raise

    return _make
