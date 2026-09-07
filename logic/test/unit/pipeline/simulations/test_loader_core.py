
import logic.src.pipeline.simulations.repository as repo_module
import pandas as pd
import pytest
from logic.src.pipeline.simulations.repository import (
    FileSystemRepository,
    load_area_and_waste_type_params,
    load_depot,
    load_indices,
    load_simulator_data,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]



class TestLoader:
    """Class for data loading tests."""

    @pytest.fixture(autouse=True)
    def setup_repo(self, tmp_path):
        original_repo = repo_module._REPOSITORY
        repo_module.set_repository(FileSystemRepository(str(tmp_path)))
        yield
        repo_module._REPOSITORY = original_repo



    @pytest.mark.unit
    def test_load_params(self, mock_load_dependencies):
        """Test loading area and waste type parameters."""
        # load_area_and_waste_type_params doesn't use read_excel, it uses hardcoded values.
        res = load_area_and_waste_type_params("riomaior", "paper")
        assert len(res) == 5
        assert res[2] == 21.0 # density for paper in riomaior

    @pytest.mark.unit
    def test_load_depot(self, mock_load_dependencies):
        """Test loading depot data."""
        mock_read_csv, _, _ = mock_load_dependencies
        # get_depot calls pd.read_csv and then reset_index
        mock_read_csv.return_value = pd.DataFrame({
            "Sigla": ["RM"],
            "Lat": [39.0],
            "Lng": [-8.0]
        })
        load_depot(data_dir="data/wsr_simulator", area="Rio Maior")
        assert mock_read_csv.called

    @pytest.mark.unit
    def test_load_indices(self, mock_load_dependencies):
        """Test loading bin indices."""
        mock_read_csv, _, _ = mock_load_dependencies
        indices = load_indices("test_graph.json", 1, 10, 100)
        assert isinstance(indices, list)

    @pytest.mark.unit
    def test_load_simulator_data(self, mock_load_dependencies):
        """Test loading main simulator data (bins and coordinates)."""
        mock_read_csv, mock_read_excel, _ = mock_load_dependencies

        # side_effect for multiple read_csv calls
        # 1. Time series data
        # 2. Coordinates data
        mock_read_csv.side_effect = [
            pd.DataFrame({
                "1": [10, 20],
                "2": [30, 40],
                "Date": ["2024-01-01", "2024-01-02"]
            }),
            pd.DataFrame({
                "ID": [1, 2],
                "Lat": [39.1, 39.2],
                "Lng": [-8.1, -8.2],
                "Tipo de Residuos": ["Embalagens de papel e cartão", "Embalagens de papel e cartão"]
            })
        ]

        load_simulator_data("data/wsr_simulator", 5, area="riomaior", waste_type="paper")
        assert mock_read_csv.call_count >= 1
