"""Review regressions for byte-preserving files and bounded directory traversal."""

import builtins
from pathlib import Path
from unittest.mock import Mock

import pytest
from cryptography.fernet import Fernet
from logic.src.utils.security import data, directories


def test_utf8_payload_is_written_as_original_bytes(tmp_path, monkeypatch):
    payload = "café\r\n".encode("utf-8")
    key = Fernet.generate_key()
    encrypted = data.encrypt_file_data(key, payload)
    output = tmp_path / "restored"
    modes = []

    def destination_open(path, mode, *args, **kwargs):
        modes.append(mode)
        # Exercise a non-UTF-8 platform default when text mode is requested.
        if "b" not in mode:
            kwargs["encoding"] = "ascii"
        return builtins.open(path, mode, *args, **kwargs)

    monkeypatch.setattr(data, "open", destination_open, raising=False)
    assert data.decrypt_file_data(key, encrypted, output) == payload.decode("utf-8")
    assert output.read_bytes() == payload
    assert modes == ["wb"]


@pytest.mark.parametrize("operation", ["encrypt", "decrypt"])
def test_nested_output_rejected_before_any_work(tmp_path, monkeypatch, operation):
    source = tmp_path / "input"
    source.mkdir()
    # An empty tree makes the fail-before test safe: no recursive file creation.
    walk = Mock(return_value=[])
    monkeypatch.setattr(directories.os, "walk", walk)
    output = source / "nested-output"
    with pytest.raises(ValueError, match="inside input"):
        getattr(directories, operation + "_directory")(Fernet.generate_key(), source, output)
    walk.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize("payload", [b"\xff\x00\r\n", b"ASCII\r\n", "café".encode("utf-8")])
def test_binary_zip_roundtrip(tmp_path, payload):
    source = tmp_path / "input"
    source.mkdir()
    (source / "payload").write_bytes(payload)
    key = Fernet.generate_key()
    encrypted = tmp_path / "archive.enc"
    directories.encrypt_zip_directory(key, source, encrypted)
    output = tmp_path / "restored"
    directories.decrypt_zip(key, encrypted, output)
    restored = list(output.rglob("payload"))
    assert len(restored) == 1
    assert restored[0].read_bytes() == payload


def test_zip_temporary_file_removed_on_encrypt_error(tmp_path, monkeypatch):
    created = []
    original = directories.tempfile.NamedTemporaryFile

    def tracked_tempfile(*args, **kwargs):
        kwargs["dir"] = tmp_path
        result = original(*args, **kwargs)
        created.append(Path(result.name))
        return result

    monkeypatch.setattr(directories.tempfile, "NamedTemporaryFile", tracked_tempfile)
    monkeypatch.setattr(directories, "zip_directory", lambda *args: None)
    monkeypatch.setattr(directories, "encrypt_file_data", Mock(side_effect=RuntimeError("injected")))
    with pytest.raises(RuntimeError, match="injected"):
        directories.encrypt_zip_directory(Fernet.generate_key(), tmp_path, tmp_path / "archive.enc")
    assert len(created) == 1
    assert not created[0].exists()
