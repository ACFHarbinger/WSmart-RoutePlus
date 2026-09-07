

from unittest.mock import MagicMock, patch

import logic.src.pipeline.simulations.repository as repo_module
import pandas as pd
import pytest
from logic.src.pipeline.simulations.repository.base import SimulationRepository
from logic.src.pipeline.simulations.repository.filesystem import FileSystemRepository

pytestmark = [pytest.mark.unit, pytest.mark.fast]



def test_set_repository():
    mock_repo = MagicMock()
    original_repo = repo_module._REPOSITORY
    try:
        repo_module.set_repository(mock_repo)
        assert repo_module._REPOSITORY is mock_repo
    finally:
        repo_module._REPOSITORY = original_repo


def test_set_repository_from_path_directory(tmp_path):
    dir_path = tmp_path / "dummy_dir"
    dir_path.mkdir()
    original_repo = repo_module._REPOSITORY
    try:
        res = repo_module.set_repository_from_path(str(dir_path))
        assert res is True
        assert isinstance(repo_module._REPOSITORY, FileSystemRepository)
    finally:
        repo_module._REPOSITORY = original_repo


def test_set_repository_from_path_nonexistent():
    res = repo_module.set_repository_from_path("non_existent_file_path_12345.npz")
    assert res is False


def test_set_repository_from_path_unsupported(tmp_path):
    file_path = tmp_path / "dummy.txt"
    file_path.write_text("hello")
    res = repo_module.set_repository_from_path(str(file_path))
    assert res is False


def test_load_indices():
    mock_repo = MagicMock()
    mock_repo.get_indices.return_value = [[1, 2], [3, 4]]
    original_repo = repo_module._REPOSITORY
    try:
        repo_module._REPOSITORY = mock_repo
        res = repo_module.load_indices("test.json", 2, 2, 10)
        assert res == [[1, 2], [3, 4]]
        mock_repo.get_indices.assert_called_once_with("test.json", 2, 2, 10, None)
    finally:
        repo_module._REPOSITORY = original_repo


def test_get_area_params():
    # Test paper / riomaior
    cap, rev, dens, exp, vol = SimulationRepository.get_area_params("Rio Maior", "paper")
    assert rev == pytest.approx(0.65 * 250 / 1000)
    assert dens == 21.0
    assert exp == 1.0
    assert vol == 2.5

    # Test paper / figueiradafoz
    cap, rev, dens, exp, vol = SimulationRepository.get_area_params("Figueira da Foz", "paper")
    assert dens == 32.0

    # Test plastic / riomaior
    cap, rev, dens, exp, vol = SimulationRepository.get_area_params("Rio Maior", "plastic")
    assert rev == pytest.approx(0.65 * 898 / 1000)
    assert dens == 19.0

    # Test plastic / figueiradafoz
    cap, rev, dens, exp, vol = SimulationRepository.get_area_params("Figueira da Foz", "plastic")
    assert dens == 20.0

    # Test glass / riomaior
    cap, rev, dens, exp, vol = SimulationRepository.get_area_params("Rio Maior", "glass")
    assert dens == 190.0

    # Test glass / figueiradafoz
    cap, rev, dens, exp, vol = SimulationRepository.get_area_params("Figueira da Foz", "glass")
    assert dens == 200.0

    # Test invalid waste type
    with pytest.raises(AssertionError):
        SimulationRepository.get_area_params("Rio Maior", "invalid_type")

    # Test invalid area
    with pytest.raises(AssertionError):
        SimulationRepository.get_area_params("invalid_area", "paper")


def test_filesystem_repository_init_and_dir():
    repo = FileSystemRepository("/dummy/root")
    assert repo.default_data_dir == "/dummy/root/data/simulator"
    assert repo._get_data_dir() == "/dummy/root/data/simulator"
    assert repo._get_data_dir("/override/path") == "/override/path"


def test_filesystem_repository_get_indices_generate(tmp_path):
    repo = FileSystemRepository(str(tmp_path))
    lock = MagicMock()
    # Check when file doesn't exist, it generates and writes indices
    filename = "indices_test.json"
    indices = repo.get_indices(filename, n_samples=3, n_nodes=2, data_size=5, lock=lock)
    assert len(indices) == 3
    for idxs in indices:
        assert len(idxs) == 2
    assert lock.acquire.called
    assert lock.release.called

    # If we call again, it should read from the existing file
    lock.reset_mock()
    indices_read = repo.get_indices(filename, n_samples=3, n_nodes=2, data_size=5, lock=lock)
    assert indices_read == indices
    assert lock.acquire.called
    assert lock.release.called


def test_filesystem_repository_get_depot(tmp_path):
    repo = FileSystemRepository(str(tmp_path))
    coord_dir = tmp_path / "data" / "simulator" / "coordinates"
    coord_dir.mkdir(parents=True)
    pd.DataFrame({
        "Sigla": ["CTEASO"],
        "Lat": [40.0],
        "Lng": [-8.0],
        "ID": [1000]
    }).to_csv(coord_dir / "Facilities.csv", index=False)

    depot = repo.get_depot("riomaior")
    assert len(depot) == 1
    assert depot.iloc[0]["ID"] == 0
    assert depot.iloc[0]["Lat"] == 40.0
    assert depot.iloc[0]["Lng"] == -8.0
    assert depot.iloc[0]["Stock"] == 0
    assert depot.iloc[0]["Accum_Rate"] == 0


def test_get_mixrmbac_data(tmp_path):
    repo = FileSystemRepository(str(tmp_path))
    with patch("pandas.read_excel") as mock_read:
        df_waste = pd.DataFrame({"ID": [1, 2], "Stock": [0.5, 0.6]})
        df_coords = pd.DataFrame({"ID": [1, 2], "Lat": [40.0, 41.0], "Lng": [-8.0, -9.0]})
        mock_read.side_effect = [df_waste, df_coords]

        # bins <= 20
        w, c = repo._get_mixrmbac_data("d_dir", 10, "mixrmbac")
        assert w is df_waste
        assert c is df_coords
        mock_read.assert_any_call("d_dir/bins_waste/StockAndAccumulationRate - small.xlsx")
        mock_read.assert_any_call("d_dir/coordinates/Coordinates - small.xlsx")

        # bins <= 50
        mock_read.reset_mock()
        mock_read.side_effect = [df_waste, df_coords]
        w, c = repo._get_mixrmbac_data("d_dir", 30, "mixrmbac")
        mock_read.assert_any_call("d_dir/bins_waste/StockAndAccumulationRate - 50bins.xlsx")

        # bins <= 225
        mock_read.reset_mock()
        mock_read.side_effect = [df_waste, df_coords]
        w, c = repo._get_mixrmbac_data("d_dir", 100, "mixrmbac")
        mock_read.assert_any_call("d_dir/bins_waste/StockAndAccumulationRate.xlsx")

        # bins > 225 should assert/raise
        with pytest.raises(AssertionError):
            repo._get_mixrmbac_data("d_dir", 300, "mixrmbac")


def test_get_riomaior_data(tmp_path):
    repo = FileSystemRepository(str(tmp_path))
    with patch("pandas.read_csv") as mock_read:
        # number_of_bins == 104
        df_sensors = pd.DataFrame({
            "Data da leitura": ["2023-01-01", "2023-01-02"],
            "idcontentor": [1, 2],
            "Enchimento": [10, 20]
        })
        df_coords = pd.DataFrame({
            "Lon": [-8.0, -8.1],
            "Latitude": [40.0, 40.1],
            "ID": [1, 2]
        })
        mock_read.side_effect = [df_sensors, df_coords]

        w, c = repo._get_riomaior_data("d_dir", 104, "riomaior", None)
        assert "Lng" in c.columns
        assert "Lat" in c.columns
        mock_read.assert_any_call("d_dir/bins_waste/Rio_Maior_Sensores_2021_2024_cleaned_104.csv")
        mock_read.assert_any_call("d_dir/coordinates/coordinates104.csv")

        # number_of_bins != 104, wtype is specified
        mock_read.reset_mock()
        df_crude = pd.DataFrame({
            "Date": ["2023-01-01", "2023-01-02"],
            "1": [1.0, 2.0],
            "2": [3.0, 4.0]
        })
        df_info = pd.DataFrame({
            "ID": [1, 2],
            "Latitude": [40.0, 40.1],
            "Longitude": [-8.0, -8.1],
            "Tipo de Residuos": ["paper", "plastic"]
        })
        mock_read.side_effect = [df_crude, df_info]
        w, c = repo._get_riomaior_data("d_dir", 50, "riomaior", "paper")
        assert len(c) == 1
        assert c.iloc[0]["ID"] == 1
        mock_read.assert_any_call("d_dir/bins_waste/old_out_crude_rate[riomaior].csv")
        mock_read.assert_any_call("d_dir/coordinates/old_out_info[riomaior].csv")

        # number_of_bins > 317 should assert/raise
        with pytest.raises(AssertionError):
            repo._get_riomaior_data("d_dir", 350, "riomaior", None)


def test_get_figueiradafoz_data(tmp_path):
    repo = FileSystemRepository(str(tmp_path))
    with patch("pandas.read_csv") as mock_read:
        df_crude = pd.DataFrame({
            "Date": ["2023-01-01", "2023-01-02"],
            "1": [1.0, 2.0]
        })
        df_info = pd.DataFrame({
            "ID": [1],
            "Latitude": [40.0],
            "Longitude": [-8.0],
            "description": ["paper"]
        })
        mock_read.side_effect = [df_crude, df_info]
        w, c = repo._get_figueiradafoz_data("d_dir", 10, "figueiradafoz", "paper")
        assert len(c) == 1
        mock_read.assert_any_call("d_dir/bins_waste/out_rate_crude[figdafoz].csv")
        mock_read.assert_any_call("d_dir/coordinates/out_info[figdafoz].csv")

        # bins > 1094 should assert/raise
        mock_read.reset_mock()
        mock_read.side_effect = [df_crude, df_info]
        with pytest.raises(AssertionError):
            repo._get_figueiradafoz_data("d_dir", 1100, "figueiradafoz", None)


def test_dataset_repository():
    import numpy as np
    from logic.src.pipeline.simulations.repository.dataset import DatasetRepository



    fake_sample_0 = {
        "node_ids": [1, 2],
        "depot": [40.0, -8.0],
        "depot_id": 0,
        "locs": np.array([[40.1, -8.1], [40.2, -8.2]])
    }
    fake_sample_1 = {
        "node_ids": [3, 4],
        "depot": [41.0, -9.0],
        "depot_id": 1,
        "locs": np.array([[41.1, -9.1], [41.2, -9.2]])
    }
    fake_dataset = [fake_sample_0, fake_sample_1]

    # pyrefly: ignore [bad-argument-type]
    repo = DatasetRepository(fake_dataset, sample_id=0)
    assert repo._sample is fake_sample_0

    # Test set_sample
    repo.set_sample(1)
    assert repo._sample is fake_sample_1

    # Test get_indices
    indices = repo.get_indices(filename=None, n_samples=3, n_nodes=2, data_size=2)
    assert indices == [[3, 4], [3, 4], [3, 4]]

    # Test get_depot
    depot_df = repo.get_depot(area="dummy")
    assert depot_df.iloc[0]["ID"] == 1
    assert depot_df.iloc[0]["Lat"] == 41.0
    assert depot_df.iloc[0]["Lng"] == -9.0

    # Test get_simulator_data
    data_df, coords_df = repo.get_simulator_data(number_of_bins=2, area="dummy")
    assert list(coords_df["ID"]) == [3, 4]
    assert coords_df.iloc[0]["Lat"] == 41.1
    assert data_df.iloc[0]["Stock"] == 0.0

    # Test get_simulator_data error case (number_of_bins mismatch)
    with pytest.raises(AssertionError):
        repo.get_simulator_data(number_of_bins=3)
