

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from logic.src.data.processor import SimulationDataMapper, format_coordinates

pytestmark = [pytest.mark.unit, pytest.mark.fast]





class TestSimulationDataMapper:
    @pytest.fixture
    def mapper(self):
        return SimulationDataMapper()

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"ID": [2, 1, 3], "val": [20, 10, 30], "obj": ["b", "a", "c"]})

    def test_sort_dataframe(self, mapper, sample_df):
        sorted_df = mapper.sort_dataframe(sample_df, "val", ascending_order=True)
        assert sorted_df.iloc[0]["val"] == 10
        assert sorted_df.columns[0] == "val"

        sorted_df_desc = mapper.sort_dataframe(sample_df, "val", ascending_order=False)
        assert sorted_df_desc.iloc[0]["val"] == 30

    def test_get_df_types(self, mapper, sample_df):
        types = mapper.get_df_types(sample_df)
        assert types["ID"] == "int32"
        assert types["val"] == "int32"
        assert types["obj"] == "string"

    def test_setup_df(self, mapper):
        depot = pd.DataFrame({"ID": [0], "Lat": [10.0], "Lng": [20.0]})
        df = pd.DataFrame({"ID": [1, 2], "Lat": [11.0, 12.0], "Lng": [21.0, 22.0]})
        col_names = ["ID", "Lat", "Lng"]

        result = mapper.setup_df(depot, df, col_names)

        # Verify depot row inserted and sorting
        assert len(result) == 3
        # ID 0 should be first after sort, but setup_df logic:
        # 1. df.loc[-1] = depot
        # 2. df.index = df.index + 1
        # 3. sort_index
        # 4. sort_values by ID

        assert result.iloc[0]["ID"] == 0
        assert result.iloc[1]["ID"] == 1
        assert result.iloc[2]["ID"] == 2

    def test_format_coordinates_pandas(self, mapper):
        # Setup data
        coords = pd.DataFrame({"Lat": [0.0, 10.0, 20.0, 30.0], "Lng": [0.0, 10.0, 20.0, 30.0]})

        # 1. MMN (Min-Max)
        depot, loc = format_coordinates(coords, "mmn")
        assert depot.shape == (2,)
        assert loc.shape == (3, 2)
        # 0 is min, 30 is max.
        # depot (0,0) -> (0,0)
        # loc[2] (30,30) -> (1,1)
        assert np.allclose(depot, [0, 0])
        assert np.allclose(loc[-1], [1, 1])

        # 2. MUN (Mean)
        depot, loc = format_coordinates(coords, "mun")
        # Mean is 15. Min 0, Max 30.
        # (0 - 15) / 30 = -0.5
        assert np.allclose(depot, [-0.5, -0.5])

        # 3. SMSD (Standardization)
        # std of [0, 10, 20, 30] is ~11.18
        depot, loc = format_coordinates(coords, "smsd")
        assert depot.shape == (2,)

        # 4. WMP (Web Mercator)
        depot, loc = format_coordinates(coords, "wmp")
        # Should transform lat/lng. loc[0] was (10,10), should be large number.
        assert not np.allclose(loc[0], [10.0, 10.0])

        # 5. C3D (3D Cartesian)
        depot, loc = format_coordinates(coords, "c3d")
        assert depot.shape == (3,)  # x, y, z
        assert loc.shape == (3, 3)

        # 6. S4D (4D Spherical)
        depot, loc = format_coordinates(coords, "s4d")
        assert depot.shape == (4,)
        assert loc.shape == (3, 4)

    def test_format_coordinates_numpy(self, mapper):
        # Setup data (Batch, Nodes, 2)
        # Note: logic seems to expect coords as (B, N, 2) but format_coordinates implementation for numpy:
        # coords[:, :, 0] -> lat
        coords = np.zeros((1, 4, 2))
        coords[0, :, 0] = [0.0, 10.0, 20.0, 30.0]
        coords[0, :, 1] = [0.0, 10.0, 20.0, 30.0]

        # MMN
        depot, loc = format_coordinates(coords, "mmn")
        # Output for numpy path: depot (B, 2), loc (B, N-1, 2) assuming node 0 is depot
        assert depot.shape == (1, 2)
        assert loc.shape == (1, 3, 2)

    def test_process_model_input(self, mapper):
        coords = pd.DataFrame({"Lat": [0.0, 10.0], "Lng": [0.0, 10.0]})
        dist_matrix = np.zeros((2, 2))
        device = torch.device("cpu")
        configs = {"problem": "ptp", "model": "tam", "graph_size": 2, "temporal_horizon": 5}

        with patch("logic.src.pipeline.simulations.repository.load_area_and_waste_type_params") as mock_load:
            # ret: CAPACITY, REVENUE, DENSITY, COST, VOLUME
            mock_load.return_value = (1000, 1.0, 0.5, 1.0, 100)

            data, (edges, dm), profit = mapper.process_model_input(
                coordinates=coords,
                dist_matrix=dist_matrix,
                device=device,
                method="mmn",
                configs=configs,
                edge_threshold=0,  # No edges
                edge_method="dist",
                area="riomaior",
                waste_type="plastic",
            )

            assert "locs" in data
            assert "depot" in data
            assert "max_waste" in data
            assert "fill_history" in data  # from 'tam' model logic

            assert dm.shape == (2, 2)
            assert edges is None

            assert profit["bin_capacity"] == 50.0  # 100 * 0.5

    def test_process_model_input_with_edges(self, mapper):
        coords = pd.DataFrame({"Lat": [0, 1], "Lng": [0, 1]})
        dist_matrix = np.array([[0, 1], [1, 0]])
        configs = {"problem": "ptp"}
        device = torch.device("cpu")

        with (
            patch(
                "logic.src.pipeline.simulations.repository.load_area_and_waste_type_params",
                return_value=(100, 1, 1, 1, 1),
            ),
            patch("logic.src.data.processor.mapper.get_adj_knn", return_value=np.ones((1, 1))),
        ):
            data, (edges, dm), profit = mapper.process_model_input(
                coords,
                dist_matrix,
                device,
                "mmn",
                configs,
                edge_threshold=0.5,  # > 0
                edge_method="knn",
                area="a",
                waste_type="w",
                adj_matrix=None,
            )
            assert edges is not None
