"""Unit tests for Evidently-based data drift detection module."""

import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from logic.src.pipeline.features.eval.drift_detection import (
    _build_arg_parser,
    _check_evidently,
    _npz_to_dataframe,
    load_and_flatten,
    run_column_drift_suite,
    run_drift_detection,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]



@pytest.fixture
def temp_files(tmp_path):
    """Generates various files (CSV, JSON, NPZ, PKL) for testing data loader."""
    # CSV
    csv_file = tmp_path / "data.csv"
    df = pd.DataFrame({"col1": [1.0, 2.0, 3.0], "col2": [4, 5, 6]})
    df.to_csv(csv_file, index=False)

    # JSON
    json_file = tmp_path / "data.json"
    df.to_json(json_file)

    # JSONL
    jsonl_file = tmp_path / "data.jsonl"
    df.to_json(jsonl_file, orient="records", lines=True)

    # PKL DataFrame
    pkl_df_file = tmp_path / "data_df.pkl"
    with open(pkl_df_file, "wb") as f:
        pickle.dump(df, f)

    # PKL dict
    pkl_dict_file = tmp_path / "data_dict.pkl"
    with open(pkl_dict_file, "wb") as f:
        pickle.dump({"col1": [1.0, 2.0], "col2": [3, 4]}, f)

    # PKL invalid
    pkl_invalid_file = tmp_path / "data_invalid.pkl"
    with open(pkl_invalid_file, "wb") as f:
        pickle.dump("just a string", f)

    # NPZ
    npz_file = tmp_path / "data.npz"
    np.savez(
        npz_file,
        arr1d=np.array([10.0, 20.0, 30.0]),
        arr2d=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        arr3d=np.array([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]]),
        scalar=np.array(42)
    )

    return {
        "csv": str(csv_file.resolve()),
        "json": str(json_file.resolve()),
        "jsonl": str(jsonl_file.resolve()),
        "pkl_df": str(pkl_df_file.resolve()),
        "pkl_dict": str(pkl_dict_file.resolve()),
        "pkl_invalid": str(pkl_invalid_file.resolve()),
        "npz": str(npz_file.resolve()),
    }


@pytest.mark.unit
def test_check_evidently():
    # Should not raise unless evidently is None (tested via patch if needed)
    with patch("logic.src.pipeline.features.eval.drift_detection.evidently", None), pytest.raises(ImportError):
        _check_evidently()


def test_parser_accepts_ttop_problem_tag():
    """The drift-report CLI accepts every temporal routing problem tag."""
    args = _build_arg_parser().parse_args(["--reference", "reference.csv", "--current", "current.csv", "--problem", "ttop"])
    assert args.problem == "ttop"


@pytest.mark.unit
def test_npz_to_dataframe(temp_files):
    df = _npz_to_dataframe(temp_files["npz"])
    assert "arr1d" in df.columns
    assert "arr2d_mean" in df.columns
    assert "arr2d_std" in df.columns
    assert "arr2d_min" in df.columns
    assert "arr2d_max" in df.columns
    assert "arr3d_mean" in df.columns
    assert "arr3d_std" in df.columns
    assert "scalar" not in df.columns
    assert len(df) == 3


@pytest.mark.unit
def test_load_and_flatten(temp_files, tmp_path):
    # CSV
    df = load_and_flatten(temp_files["csv"])
    assert list(df.columns) == ["col1", "col2"]

    # JSON
    df = load_and_flatten(temp_files["json"])
    assert "col1" in df.columns

    # JSONL
    df = load_and_flatten(temp_files["jsonl"])
    assert "col1" in df.columns

    # PKL df
    df = load_and_flatten(temp_files["pkl_df"])
    assert "col1" in df.columns

    # PKL dict
    df = load_and_flatten(temp_files["pkl_dict"])
    assert "col1" in df.columns

    # PKL invalid
    with pytest.raises(ValueError):
        load_and_flatten(temp_files["pkl_invalid"])

    # NPZ
    df = load_and_flatten(temp_files["npz"])
    assert "arr1d" in df.columns

    # Non-existent
    with pytest.raises(FileNotFoundError):
        load_and_flatten("non_existent_file.csv")

    # Unsupported extension (file exists but has wrong extension)
    bad_ext_file = tmp_path / "bad.txt"
    bad_ext_file.write_text("Hello", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_and_flatten(str(bad_ext_file.resolve()))


@pytest.mark.unit
def test_run_drift_detection(temp_files, tmp_path):
    mock_evidently = MagicMock()
    mock_column_mapping = MagicMock()
    mock_report_instance = MagicMock()
    mock_report_instance.as_dict.return_value = {
        "metrics": [{"result": {"drift_detected": True}}, {"result": {"drift_detected": False}}]
    }

    patches = {
        "evidently": mock_evidently,
        "ColumnMapping": mock_column_mapping,
        "Report": MagicMock(return_value=mock_report_instance),
        "DataDriftPreset": MagicMock(),
        "DataQualityPreset": MagicMock(),
        "DatasetDriftMetric": MagicMock(),
        "DatasetMissingValuesMetric": MagicMock(),
        "ColumnDriftMetric": MagicMock(),
    }

    with patch.multiple("logic.src.pipeline.features.eval.drift_detection", **patches):
        out_dir = tmp_path / "drift_reports"
        report_path = run_drift_detection(
            reference_path=temp_files["csv"],
            current_path=temp_files["csv"],
            output_dir=str(out_dir.resolve()),
            report_filename="my_report.html",
            target_column="col2",
            feature_columns=["col1"],
        )

        assert "my_report.html" in report_path
        mock_report_instance.run.assert_called_once()
        mock_report_instance.save_html.assert_called_once()


@pytest.mark.unit
def test_run_column_drift_suite(temp_files, tmp_path):
    mock_evidently = MagicMock()
    mock_report_instance = MagicMock()

    patches = {
        "evidently": mock_evidently,
        "ColumnMapping": MagicMock(),
        "Report": MagicMock(return_value=mock_report_instance),
        "ColumnDriftMetric": MagicMock(),
        "ColumnSummaryMetric": MagicMock(),
    }

    with patch.multiple("logic.src.pipeline.features.eval.drift_detection", **patches):
        out_dir = tmp_path / "drift_reports"
        report_path = run_column_drift_suite(
            reference_path=temp_files["csv"],
            current_path=temp_files["csv"],
            columns=["col1"],
            output_dir=str(out_dir.resolve()),
        )

        assert "column_drift_" in report_path
        mock_report_instance.run.assert_called_once()
        mock_report_instance.save_html.assert_called_once()
