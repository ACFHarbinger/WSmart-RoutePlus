"""Tests for utils/input/files.py."""

import json
import os

import pytest
from logic.src.utils.input.files import compose_dirpath, extract_zip, read_json, zip_directory

pytestmark = [pytest.mark.unit, pytest.mark.fast]





def test_read_json(tmp_path):
    """Verify safe JSON reading."""
    data = {"test": 123}
    path = tmp_path / "test.json"
    path.write_text(json.dumps(data))

    assert read_json(str(path)) == data


def test_zip_unzip_directory(tmp_path):
    """Verify zipping and extraction of directories."""
    # Setup
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "file.txt").write_text("hello")

    zip_path = tmp_path / "test.zip"
    output_dir = tmp_path / "output"

    # Zip
    zip_directory(str(input_dir), str(zip_path))
    assert os.path.exists(zip_path)

    # Extract
    extract_zip(str(zip_path), str(output_dir))
    assert (output_dir / "file.txt").read_text() == "hello"


def test_compose_dirpath():
    """Verify directory path composition decorator."""

    @compose_dirpath
    def mock_fun(path, *args, **kwargs):
        return path

    home = "/test"
    ndays = 30
    nbins = 20
    out = "res"
    area = "madrid"

    path = mock_fun(home, ndays, nbins, out, area)
    # Expected: /test/res/30days/madrid20_ (output_dir is relative to the repository root)
    assert path.startswith("/test/res/30days")
    assert "30days" in path
    assert "madrid20_" in path
