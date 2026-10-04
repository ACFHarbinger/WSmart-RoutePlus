"""Directory operations must reject invalid sources before creating outputs."""

import pytest
from cryptography.fernet import Fernet
from logic.src.utils.security import directories


@pytest.mark.parametrize("operation", ["encrypt_directory", "decrypt_directory", "encrypt_zip_directory"])
@pytest.mark.parametrize("source_kind", ["missing", "file"])
def test_invalid_source_leaves_destination_untouched(tmp_path, operation, source_kind):
    source = tmp_path / "source"
    destination = tmp_path / "output"
    if source_kind == "file":
        source.write_bytes(b"original")
    error = FileNotFoundError if source_kind == "missing" else NotADirectoryError
    with pytest.raises(error):
        getattr(directories, operation)(Fernet.generate_key(), source, destination)
    assert not destination.exists()
    if source_kind == "missing":
        assert not source.exists()
    else:
        assert source.read_bytes() == b"original"


@pytest.mark.parametrize("operation", ["encrypt_directory", "decrypt_directory"])
def test_missing_inplace_source_is_not_created(tmp_path, operation):
    source = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        getattr(directories, operation)(Fernet.generate_key(), source)
    assert not source.exists()


@pytest.mark.parametrize("operation", ["encrypt_directory", "decrypt_directory"])
def test_existing_empty_source_is_valid(tmp_path, operation):
    source = tmp_path / "empty"
    source.mkdir()
    destination = tmp_path / "output"
    assert getattr(directories, operation)(Fernet.generate_key(), source, destination) == []
    assert destination.is_dir()
