"""Drive sync: copy-only rclone wrapper, checked end to end with rclone's local backend."""

import shutil

import pytest
from logic.cloud import remote as remote_module
from logic.cloud.downloader.sync import download
from logic.cloud.remote import join_remote, local_directory
from logic.cloud.uploader.sync import upload

pytestmark = pytest.mark.skipif(shutil.which("rclone") is None, reason="rclone is not installed")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "data" / "sub").mkdir(parents=True)
    (root / "data" / "sub" / "f.txt").write_text("waste")
    (root / "data" / "run.log").write_text("log")
    (root / "data" / "linked.txt").symlink_to(root / "data" / "sub" / "f.txt")
    monkeypatch.setattr(remote_module, "repository_root", lambda: root)
    return root


@pytest.fixture
def drive(tmp_path):
    folder = tmp_path / "drive"
    folder.mkdir()
    return f":local:{folder}"


def test_join_remote_adds_one_separator():
    assert join_remote("gdrive:WSmart-Route/", "data") == "gdrive:WSmart-Route/data"


def test_upload_copies_files_and_skips_symlinks(repo, drive, tmp_path):
    upload("data", drive, config="/dev/null")
    copied = tmp_path / "drive" / "data"
    assert (copied / "sub" / "f.txt").read_text() == "waste"
    assert not (copied / "linked.txt").exists()


def test_upload_dry_run_changes_nothing(repo, drive, tmp_path):
    upload("data", drive, dry_run=True, config="/dev/null")
    assert not (tmp_path / "drive" / "data").exists()


def test_upload_honours_excludes(repo, drive, tmp_path):
    upload("data", drive, excludes=["*.log"], config="/dev/null")
    assert not (tmp_path / "drive" / "data" / "run.log").exists()
    assert (tmp_path / "drive" / "data" / "sub" / "f.txt").exists()


def test_download_fills_the_local_directory(repo, drive, tmp_path):
    remote_outputs = tmp_path / "drive" / "outputs" / "30days"
    remote_outputs.mkdir(parents=True)
    (remote_outputs / "result.json").write_text("{}")
    download("outputs", drive, config="/dev/null")
    assert (repo / "assets" / "output" / "30days" / "result.json").read_text() == "{}"


def test_nothing_is_deleted_on_either_side(repo, drive, tmp_path):
    (tmp_path / "drive" / "data").mkdir()
    (tmp_path / "drive" / "data" / "remote_only.txt").write_text("keep")
    upload("data", drive, config="/dev/null")
    assert (tmp_path / "drive" / "data" / "remote_only.txt").read_text() == "keep"


def test_symlinked_source_directory_is_refused(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "elsewhere").mkdir(parents=True)
    root.mkdir(exist_ok=True)
    (root / "data").symlink_to(root / "elsewhere")
    monkeypatch.setattr(remote_module, "repository_root", lambda: root)
    with pytest.raises(ValueError, match="symlink"):
        local_directory("data")


def test_symlinked_download_destination_is_refused(repo, drive, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (repo / "assets" / "output").parent.mkdir(parents=True, exist_ok=True)
    (repo / "assets" / "output").symlink_to(other)
    with pytest.raises(ValueError, match="symlink"):
        download("outputs", drive, config="/dev/null")


def test_missing_rclone_is_reported(repo, drive, monkeypatch):
    monkeypatch.setattr(remote_module.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="rclone is not installed"):
        upload("data", drive, config="/dev/null")
