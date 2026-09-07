"""Unit tests for simulation loader and context."""

import json
import os
from unittest.mock import patch

import pandas as pd
import pytest
from logic.src.pipeline.simulations.day_context import SimulationDayContext
from logic.src.pipeline.simulations.repository import (
    FileSystemRepository,
    load_depot,
    load_indices,
    load_simulator_data,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]



def test_simulation_day_context_mapping():
    """Test SimulationDayContext dataclass and Mapping interface."""
    ctx = SimulationDayContext(
        graph_size=10,
        full_policy="test_policy",
        day=5
    )

    # Access via attributes
    assert ctx.graph_size == 10

    # Access via __getitem__ (Mapping)
    assert ctx["full_policy"] == "test_policy"
    assert ctx["day"] == 5

    # Test __setitem__
    ctx["cost"] = 15.0
    assert ctx.cost == 15.0

    # Test get()
    assert ctx.get("graph_size") == 10
    assert ctx.get("missing", "default") == "default"

    # Test Mapping properties
    assert "graph_size" in ctx
    assert len(ctx) > 30 # many fields
    assert "graph_size" in list(iter(ctx))

def test_file_system_repository_init(tmp_path):
    """Test repository initialization and path resolution."""
    repo = FileSystemRepository(str(tmp_path))
    expected_dir = os.path.join(str(tmp_path), "data", "simulator")
    assert repo.default_data_dir == expected_dir
    assert repo._get_data_dir() == expected_dir
    assert repo._get_data_dir("override") == "override"

def test_get_indices_from_file(tmp_path):
    """Test loading indices from an existing JSON file."""
    repo = FileSystemRepository(str(tmp_path))
    os.makedirs(os.path.join(repo.default_data_dir, "bins_selection"), exist_ok=True)

    file_path = os.path.join(repo.default_data_dir, "bins_selection", "test.json")
    indices = [[1, 2], [3, 4]]
    with open(file_path, "w") as f:
        json.dump(indices, f)

    loaded = repo.get_indices("test.json", n_samples=2, n_nodes=2, data_size=10)
    assert loaded == indices

@patch("pandas.Series.sample")
def test_get_indices_generate(mock_sample, tmp_path):
    """Test generating indices when file is missing."""
    repo = FileSystemRepository(str(tmp_path))
    os.makedirs(os.path.join(repo.default_data_dir, "bins_selection"), exist_ok=True)

    # Mock sample to return predictable list
    mock_sample.return_value = pd.Series([5, 6])

    indices = repo.get_indices("new.json", n_samples=1, n_nodes=2, data_size=10)
    assert indices == [[5, 6]]

    # Verify file was created
    file_path = os.path.join(repo.default_data_dir, "bins_selection", "new.json")
    assert os.path.exists(file_path)

@patch("pandas.read_csv")
def test_get_depot(mock_read_csv, tmp_path):
    """Test loading depot coordinates."""
    repo = FileSystemRepository(str(tmp_path))

    # Mock Facilities.csv content
    # Rio Maior sigla is CTEASO
    df = pd.DataFrame({
        "Sigla": ["CTEASO", "CITVRSU"],
        "Lat": [39.0, 40.0],
        "Lng": [-8.0, -9.0]
    })
    mock_read_csv.return_value = df

    # Rio Maior (RM)
    depot = repo.get_depot("Rio Maior")
    assert depot.iloc[0]["Lat"] == 39.0
    assert depot.iloc[0]["ID"] == 0
    assert "Stock" in depot.columns

@patch("pandas.read_excel")
@patch("pandas.read_csv")
def test_get_simulator_data_mixrmbac(mock_read_csv, mock_read_excel, tmp_path):
    """Test loading simulator data for Mix RM/BAC (Excel branch)."""
    repo = FileSystemRepository(str(tmp_path))

    df = pd.DataFrame({
        "ID": [1, 2],
        "Lat": [39.1, 39.2],
        "Lng": [-8.1, -8.2],
        "Stock": [0.1, 0.2],
        "Accum_Rate": [0.3, 0.4]
    })
    mock_read_excel.return_value = df

    stats, coords = repo.get_simulator_data(20, area="mixrmbac")

    assert len(stats) == 2
    assert "Stock" in stats.columns

@patch("pandas.read_csv")
def test_get_simulator_data_riomaior(mock_read_csv, tmp_path):
    """Test loading simulator data for Rio Maior."""
    repo = FileSystemRepository(str(tmp_path))

    # Mock waste stats
    waste_df = pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02"],
        "101": [50.0, 60.0],
        "102": [10.0, 20.0]
    })

    # Mock coordinates
    coords_df = pd.DataFrame({
        "ID": [101, 102],
        "Lat": [39.1, 39.2],
        "Lng": [-8.1, -8.2],
        "Tipo de Residuos": ["Indiferenciados", "Indiferenciados"]
    })

    # read_csv will be called twice: once for waste, once for coords
    mock_read_csv.side_effect = [waste_df, coords_df]

    stats, coords = repo.get_simulator_data(2, area="Rio Maior")

    assert len(stats) == 2
    assert len(coords) == 2
    assert stats.iloc[0]["ID"] == 101

@patch("pandas.read_csv")
def test_get_simulator_data_figueira(mock_read_csv, tmp_path):
    """Test loading simulator data for Figueira da Foz."""
    repo = FileSystemRepository(str(tmp_path))

    # Mock data for Figueira
    mock_read_csv.side_effect = [
        pd.DataFrame({"Date": ["2024-01-01"], "1": [0.5]}), # waste
        pd.DataFrame({"ID": [1], "Lat": [40.0], "Lng": [-8.0], "Tipo de Residuos": ["Indiferenciados"]}) # coords
    ]

    stats, coords = repo.get_simulator_data(1, area="figueiradafoz")
    assert len(stats) == 1
    assert coords.iloc[0]["ID"] == 1

@patch("pandas.read_excel")
@patch("pandas.read_csv")
def test_get_simulator_data_both(mock_read_csv, mock_read_excel, tmp_path):
    """Test loading simulator data for 'both' area (merged datasets)."""
    repo = FileSystemRepository(str(tmp_path))

    # Needs wsrs_data (Excel) and wsba_data (CSV)
    # Plus intersection/merged/union CSVs
    mock_read_excel.return_value = pd.DataFrame({"ID": [1], "Stock": [0], "Accum_Rate": [0]})

    mock_read_csv.side_effect = [
        pd.DataFrame({"Date": ["2024-01-01"], "101": [0]}), # wsba_data
        pd.DataFrame({"ID225": [1], "ID317": [101], "Lat": [39], "Lng": [-8]}) # intersection.csv
    ]

    stats, coords = repo.get_simulator_data(57, area="both")
    assert len(stats) == 1
    assert coords.iloc[0]["ID"] == 1

def test_wrapper_functions(mocker):
    """Test top-level wrapper functions call the repository."""
    mock_repo = mocker.patch("logic.src.pipeline.simulations.repository._REPOSITORY")

    load_indices("test.json", 1, 1, 1)
    mock_repo.get_indices.assert_called()

    load_depot("dir", "area")
    mock_repo.get_depot.assert_called()

    mock_repo.get_simulator_data.return_value = (mocker.Mock(), mocker.Mock())
    load_simulator_data("dir", 5)
    mock_repo.get_simulator_data.assert_called()
