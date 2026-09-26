"""
Fixtures relating to data generation, datasets, and file system structures for data.
"""

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def gen_data_opts():
    """Returns a basic set of mock arguments (opts) for generate_datasets."""
    return {
        "name": "test_suite",
        "problem": "vrpp",
        "n_samples": 100,
        "graph_sizes": [20],
        "data_distributions": ["all"],
        "area": "SomeArea",
        "waste_type": "SomeWaste",
        "focus_graph": ["graph.csv"],
        "focus_size": 0,
        "vertex_method": "Uniform",
        "is_gaussian": False,
        "sigma": 0.1,
        "seed": 42,
        "dataset_type": "train",
        "env.graph.n_days": 1,
        "env.graph.start_day": 0,
        "data_dir": "datasets",
        "filename": "test",
        "f": True,
        "penalty_factor": 10,
    }


@pytest.fixture
def sample_dataset_file(temp_data_dir):
    """Create a sample dataset file"""
    dataset_path = Path(temp_data_dir) / "sample_dataset.pkl"
    # Create an empty file for testing
    dataset_path.touch()
    return str(dataset_path)


@pytest.fixture
def sample_model_file(temp_model_dir):
    """Create a sample model file"""
    model_path = Path(temp_model_dir) / "sample_model.pt"
    # Create an empty file for testing
    model_path.touch()
    return str(model_path)


@pytest.fixture
def sample_config_file(temp_data_dir):
    """Create a sample configuration file"""
    config_path = Path(temp_data_dir) / "sample_config.json"
    # Create an empty file for testing
    config_path.touch()
    return str(config_path)


@pytest.fixture(autouse=True)
def mock_dependencies(mocker):
    """Mocks all external data generation and saving dependencies."""
    # Mock the save utilities
    mocker.patch("logic.src.data.generators.datasets.save_td_dataset", return_value=None)
    mocker.patch("logic.src.data.generators.datasets.save_simulation_dataset", return_value=None)


@pytest.fixture
def mock_data_dir(tmp_path):
    """Creates a mock data directory structure."""
    data_dir = tmp_path / "data" / "simulator"

    # For loader.py (load_depot)
    coord_dir = data_dir / "coordinates"
    coord_dir.mkdir(parents=True, exist_ok=True)
    facilities_data = {
        "Sigla": ["RM", "OTHER"],
        "Lat": [40.0, 41.0],
        "Lng": [-8.0, -9.0],
        "ID": [1000, 1001],
    }
    pd.DataFrame(facilities_data).to_csv(coord_dir / "Facilities.csv", index=False)

    # For loader.py (load_simulator_data)
    pd.DataFrame({"ID": [1, 2, 3], "Lat": [40.1, 40.2, 40.3], "Lng": [-8.1, -8.2, -8.3]}).to_csv(
        coord_dir / "old_out_info[riomaior].csv", index=False
    )

    # For loader.py (load_simulator_data)
    data_file = data_dir / "daily_c[riomaior_paper].csv"
    sim_data = {
        "Date": ["2023-01-01", "2023-01-02"],
        "1": [10, 20],
        "2": [15, 25],
        "3": [20, 30],
    }
    pd.DataFrame(sim_data).to_csv(data_file, index=False)

    return data_dir


@pytest.fixture
def mock_coords_df():
    """Returns a mock coordinates DataFrame."""
    depot = pd.DataFrame({"ID": [0], "Lat": [40.0], "Lng": [-8.0], "Stock": [0], "Accum_Rate": [0]})
    depot = depot.set_index(pd.Index([0]))

    coords = pd.DataFrame({"ID": [1, 2], "Lat": [40.1, 40.2], "Lng": [-8.1, -8.2]})
    return depot, coords


@pytest.fixture
def mock_data_df():
    """Returns a mock data DataFrame."""
    data = pd.DataFrame({"ID": [1, 2], "Stock": [10, 20], "Accum_Rate": [5, 3]})
    return data
