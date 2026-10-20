"""
Tests for the input file splitting and preview utilities.

The ``TestSplitting`` and ``TestPreview`` groups, moved unchanged from ``test_input.py``.
"""

import json
import os
import zipfile
from unittest.mock import mock_open, patch

import pandas as pd
import pytest
from logic.src.utils.input.preview import (
    preview_changes,
    preview_file_changes,
    preview_file_statistics,
    preview_pattern_files_statistics,
)
from logic.src.utils.input.splitting import chunk_zip_content, reassemble_files, split_file

pytestmark = [pytest.mark.unit, pytest.mark.fast]


# ============================================================================
# File Splitting Tests
# ============================================================================


class TestSplitting:
    """Tests for file splitting and reassembly."""

    def test_split_and_reassemble_csv(self, input_temp_dir):
        """Test CSV file splitting and reassembly."""
        data_dir = os.path.join(input_temp_dir, "data")
        os.makedirs(data_dir)

        # Create a dummy CSV file
        df = pd.DataFrame({"col1": range(100), "col2": range(100)})
        file_path = os.path.join(data_dir, "test.csv")
        df.to_csv(file_path, index=False)

        # Split
        chunks = split_file(file_path, max_part_size=100, output_dir=data_dir)
        assert len(chunks) > 1, "File should have been split into multiple chunks"

        # Cleanup original file to verify reassembly
        os.remove(file_path)

        # Reassemble
        reassembled = reassemble_files(data_dir)
        assert "test.csv" in reassembled
        assert os.path.exists(file_path)

        # Verify content
        df_new = pd.read_csv(file_path)
        pd.testing.assert_frame_equal(df, df_new)

    def test_chunk_zip_content(self, input_temp_dir):
        """Test chunking zip file contents."""
        data_dir = os.path.join(input_temp_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        # Create dummy file inside zip
        csv_path = os.path.join(data_dir, "inside.csv")
        df = pd.DataFrame({"x": range(100)})
        df.to_csv(csv_path, index=False)

        zip_path = os.path.join(input_temp_dir, "test.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            zipf.write(csv_path, arcname="inside.csv")

        os.remove(csv_path)

        # Chunk zip content
        data_out = os.path.join(input_temp_dir, "extracted")
        os.makedirs(data_out)

        files_list = chunk_zip_content(zip_path, max_part_size=50, data_dir=data_out)
        assert len(files_list) > 1
        assert any("inside_part" in f for f in files_list)

        # Reassemble from the extracted dir
        reassemble_files(data_out)
        assert os.path.exists(os.path.join(data_out, "inside.csv"))


# ============================================================================
# Preview Utilities Tests
# ============================================================================


class TestPreview:
    """Tests for preview utilities."""

    def test_preview_changes(self, input_temp_dir):
        """Test previewing changes to directory."""
        json_path = os.path.join(input_temp_dir, "log_1.json")
        with open(json_path, "w") as f:
            json.dump({"entry": {"km": 100}}, f)

        def stats(x: float, keys: list) -> float:
            """Identity stats function."""
            return x

        preview_changes(input_temp_dir, output_key="km", update_val=0.5, process_func=stats)

    def test_preview_file_changes(self, input_temp_dir):
        """Test previewing changes to single file."""
        json_path = os.path.join(input_temp_dir, "log_1.json")
        if not os.path.exists(json_path):
            with open(json_path, "w") as f:
                json.dump({"entry": {"km": 100}}, f)

        def stats(x: float, keys: list) -> float:
            """Identity stats function for file preview."""
            return x

        preview_file_changes(json_path, output_key="km", update_val=0.5, process_func=stats)

    def test_preview_statistics(self, input_temp_dir):
        """Test previewing file statistics."""
        json_path = os.path.join(input_temp_dir, "log_1.json")
        if not os.path.exists(json_path):
            with open(json_path, "w") as f:
                json.dump({"entry": {"km": 100}}, f)

        def stats(x: list) -> float:
            """Sum the input list as stats."""
            return sum(x)

        preview_file_statistics(json_path, output_key="km", process_func=stats)

    @patch("logic.src.utils.input.preview.find_single_input_values")
    @patch("logic.src.utils.input.preview.os.walk")
    def test_preview_changes_mocked(self, mock_walk, mock_find):
        """Test previewing changes with mocked search."""
        mock_find.return_value = [("entry.km", 10)]
        # preview_changes uses glob, not os.walk.
        # But wait, preview_changes implementation uses glob.glob

        with patch("logic.src.utils.input.preview.glob.glob") as mock_glob:
            mock_glob.return_value = ["file.json"]

            with patch("builtins.print"):
                # We need to mock open/json load too
                with patch("builtins.open", new_callable=mock_open, read_data='{"a": 1}'):
                    preview_changes(
                        "root",
                        output_key="km",
                        update_val=0,
                        process_func=lambda x, u: x,
                    )

                assert mock_glob.called
                assert mock_find.called

    def test_preview_pattern_files_statistics(self):
        """Test previewing stats for pattern files."""
        with patch("logic.src.utils.input.preview.glob.glob") as mock_glob:
            mock_glob.return_value = ["f1.json"]

            # We need to mock open/json load
            with (
                patch("builtins.open", new_callable=mock_open, read_data='{"a": 1}'),
                patch("logic.src.utils.input.preview.find_single_input_values") as mock_find,
            ):
                mock_find.return_value = [("entry.val", 10)]

                with patch("builtins.print"):
                    preview_pattern_files_statistics("root", process_func=lambda x: sum(x))

                    assert mock_find.called

    @patch("logic.src.utils.input.preview.os.path.isfile")
    @patch("builtins.open", new_callable=mock_open, read_data='{"a": 10}')
    def test_preview_file_statistics_mocked(self, mock_file, mock_isfile):
        """Test file statistics preview."""
        mock_isfile.return_value = True

        with patch("builtins.print") as mock_print:
            preview_file_statistics("f.json", output_key="a", process_func=lambda x: x * 2)
            # Should print "old: ... new: ..."
            assert mock_print.called
