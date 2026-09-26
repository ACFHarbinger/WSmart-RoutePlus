"""
Data Mapper for transformation between raw data and simulation representations.

Attributes:
    SimulationDataMapper: Instance of SimulationDataMapper.

Example:
    from logic.src.data.processor import SimulationDataMapper
    mapper = SimulationDataMapper()
    depot_sample = pd.DataFrame([{'ID': 0, 'Lat': 40.7128, 'Lng': -74.0060}])
    bin_samples = pd.DataFrame([{'ID': 1, 'Lat': 40.7128, 'Lng': -74.0060}, {'ID': 2, 'Lat': 40.7128, 'Lng': -74.0060}])
    coords_sample = pd.DataFrame([{'ID': 0, 'Lat': 40.7128, 'Lng': -74.0060}, {'ID': 1, 'Lat': 40.7128, 'Lng': -74.0060}, {'ID': 2, 'Lat': 40.7128, 'Lng': -74.0060}])
    mapper.process_raw_data(bin_samples, coords_sample, depot_sample)
"""

import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import torch

from logic.src.constants import MAX_WASTE

if TYPE_CHECKING:
    pass

from logic.src.utils.graph import (
    adj_to_idx,
    get_adj_knn,
    get_edge_idx_dist,
)

from .formatting import format_coordinates


class SimulationDataMapper:
    """
    Data Mapper for the WSmart+ Route simulator.
    Attributes:
        None
    """

    def sort_dataframe(self, df: pd.DataFrame, metric_tosort: str, ascending_order: bool = True) -> pd.DataFrame:
        """Sorts a DataFrame by a metric.

        Args:
            df: DataFrame to sort.
            metric_tosort: Metric to sort by.
            ascending_order: Whether to sort in ascending order.

        Returns:
            Sorted DataFrame.
        """
        df = df.sort_values(by=metric_tosort, ascending=ascending_order)
        columns = [metric_tosort] + [col for col in df.columns if col != metric_tosort]
        return cast(pd.DataFrame, df[columns])

    def get_df_types(self, df: pd.DataFrame, prec: str = "32") -> Dict[str, str]:
        """Infers and maps column data types to specific precisions.

        Args:
            df: DataFrame to get types from.
            prec: Precision to use for floating point and integer types.

        Returns:
            Dictionary of column names to data types.
        """
        df_types = dict(df.dtypes)
        for key, val in df_types.items():
            if key == "ID":
                new_type = f"int{prec}"
            elif "obj" in str(val):
                new_type = "string"
            elif "float" in str(val) or "int" in str(val):
                new_type = str(val)[:-2] + prec
            else:
                new_type = str(val)
            df_types[key] = new_type
        return df_types

    def setup_df(
        self,
        depot: pd.DataFrame,
        df: pd.DataFrame,
        col_names: List[str],
        index_name: Optional[str] = "#bin",
    ) -> pd.DataFrame:
        """Merges depot and bin data.

        Args:
            depot: DataFrame containing depot data.
            df: DataFrame containing bin data.
            col_names: List of column names to include.
            index_name: Name of the index column.

        Returns:
            DataFrame containing merged and sorted data.
        """
        df = df.loc[:, col_names].copy()
        df.loc[-1] = depot.loc[0, col_names].values
        df.index = df.index + 1
        df = df.sort_index()
        if index_name is None:
            df = df.sort_values(by="ID").reset_index(drop=True).astype(self.get_df_types(df))
        else:
            df = df.sort_values(by="ID").reset_index().astype(self.get_df_types(df))
            df = df.rename(columns={"index": index_name})
            df[index_name] = df[index_name].astype(df["ID"].dtype)
        return df

    def sample_df(
        self,
        df: pd.DataFrame,
        n_elems: int,
        depot: Optional[pd.DataFrame] = None,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Samples a subset of bins.

        Args:
            df: DataFrame to sample from.
            n_elems: Number of elements to sample.
            depot: Optional depot DataFrame to add to the sample.
            output_path: Optional path to save the sample indices.

        Returns:
            Sampled DataFrame.
        """
        df = df.sample(n=n_elems)
        df_types = self.get_df_types(df)
        if depot is not None:
            df.loc[0] = depot
        if output_path is not None:
            if os.path.isfile(output_path):
                with open(output_path) as fp:
                    data = json.load(fp)
                data.append(df.sort_index().index.tolist())
            else:
                data = [df.sort_index().index.tolist()]
            with open(output_path, "w") as fp:
                json.dump(data, fp)
        return df.sort_values(by="ID").reset_index(drop=True).astype(df_types)

    def process_indices(self, df: pd.DataFrame, indices: Optional[List[int]]) -> pd.DataFrame:
        """Extracts subset based on indices.

        Args:
            df: DataFrame to extract subset from.
            indices: Indices to extract.

        Returns:
            Subset of DataFrame.
        """
        if indices is None:
            return df.copy()
        if "index" in df.columns or "ID" in df.columns:
            sampled_df = df.iloc[indices]
            if isinstance(sampled_df, pd.Series):
                sampled_df = sampled_df.to_frame().T
            return sampled_df.sort_values(by="ID").reset_index(drop=True).astype(self.get_df_types(sampled_df))
        sampled_df = df.iloc[:, indices]
        if isinstance(sampled_df, pd.Series):
            sampled_df = sampled_df.to_frame().T
        return sampled_df

    def process_raw_data(
        self,
        data: pd.DataFrame,
        bins_coordinates: pd.DataFrame,
        depot: pd.DataFrame,
        indices: Optional[List[int]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepares coordinate and statistic data.

        Args:
            data: DataFrame containing bin data.
            bins_coordinates: DataFrame containing bin coordinates.
            depot: DataFrame containing depot data.
            indices: Optional indices to extract.

        Returns:
            Tuple of (DataFrame, DataFrame) containing processed data and coordinates.
        """
        new_data = self.process_indices(data, indices)
        coords = self.process_indices(bins_coordinates, indices)
        coords = self.setup_df(depot, coords, ["ID", "Lat", "Lng"])
        new_data = self.setup_df(depot, new_data, ["ID", "Stock", "Accum_Rate"])
        return new_data, coords

    def _prepare_model_data(
        self, coordinates: Any, method: str, configs: Dict[str, Any], problem_size: int
    ) -> Dict[str, torch.Tensor]:
        """Prepare the base model data dictionary.

        Args:
            coordinates: Coordinates of bins.
            method: Normalization method.
            configs: Configuration dictionary.
            problem_size: Size of the problem.

        Returns:
            Dictionary of model data.
        """
        depot, loc = format_coordinates(coordinates, method)
        model_data = {
            "locs": torch.as_tensor(loc, dtype=torch.float32),
            "depot": torch.as_tensor(depot, dtype=torch.float32),
            "waste": torch.zeros(problem_size),
        }

        if configs.get("problem") in ["vrpp", "cvrpp", "ctop", "wcvrp", "cwcvrp", "sdwcvrp"]:
            model_data["max_waste"] = torch.as_tensor(MAX_WASTE, dtype=torch.float32)
        elif "problem" in configs:
            raise ValueError(f"Unknown problem: {configs['problem']}")

        if configs.get("model") == "tam":
            model_data["fill_history"] = torch.zeros((1, configs["graph_size"], configs["temporal_horizon"]))

        return model_data

    def _prepare_edges(
        self,
        dist_matrix: np.ndarray,
        configs: Dict[str, Any],
        device: torch.device,
        edge_threshold: float,
        edge_method: str,
        adj_matrix: Optional[np.ndarray],
    ) -> Optional[torch.Tensor]:
        """Prepare edge indices and adjacency matrix.

        Args:
            dist_matrix: Distance matrix.
            configs: Configuration dictionary.
            device: Device to use.
            edge_threshold: Threshold for edge detection.
            edge_method: Method to use for edge detection.
            adj_matrix: Adjacency matrix.

        Returns:
            Edge indices and adjacency matrix.
        """
        if not (0 < edge_threshold < 1):
            return None

        if edge_method == "dist":
            edges = (
                torch.tensor(adj_to_idx(adj_matrix, negative=False))
                if adj_matrix is not None
                else torch.tensor(get_edge_idx_dist(dist_matrix[1:, 1:], edge_threshold))
            )
        else:
            assert edge_method == "knn"
            edges = (
                torch.from_numpy(adj_matrix)
                if adj_matrix is not None
                else torch.from_numpy(get_adj_knn(dist_matrix[1:, 1:], edge_threshold, negative=False))
            )

        dtype = torch.float32 if configs.get("encoder") in ["gac", "tgc"] else torch.bool
        return edges.unsqueeze(0).to(device, dtype=dtype)

    def _load_profit_vars(self, area: str, waste_type: str) -> Dict[str, float]:
        """Load profit-related variables for the simulation.

        Args:
            area: Area of the simulation.
            waste_type: Type of waste.

        Returns:
            Dictionary of profit-related variables.
        """
        from logic.src.pipeline.simulations.repository import load_area_and_waste_type_params

        (
            VEHICLE_CAPACITY,
            REVENUE_KG,
            DENSITY,
            COST_KM,
            VOLUME,
        ) = load_area_and_waste_type_params(area, waste_type)

        return {
            "cost_km": COST_KM,
            "revenue_kg": REVENUE_KG,
            "bin_capacity": VOLUME * DENSITY,
            "vehicle_capacity": VEHICLE_CAPACITY / 100,
        }

    def process_model_input(
        self,
        coordinates: Any,
        dist_matrix: np.ndarray,
        device: torch.device,
        method: str,
        configs: Dict[str, Any],
        edge_threshold: float,
        edge_method: str,
        area: str,
        waste_type: str,
        adj_matrix: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Tuple[Optional[torch.Tensor], torch.Tensor], Dict[str, float]]:
        """Prepares data for neural models.

        Args:
            coordinates: Coordinates of bins.
            dist_matrix: Distance matrix.
            device: Device to use.
            method: Normalization method.
            configs: Configuration dictionary.
            edge_threshold: Threshold for edge detection.
            edge_method: Method to use for edge detection.
            area: Area of the simulation.
            waste_type: Type of waste.
            adj_matrix: Adjacency matrix.

        Returns:
            Tuple of (model data, edges and distance matrix, profit variables).
        """
        problem_size = len(dist_matrix) - 1

        # 1. Prepare Model Data
        model_data = self._prepare_model_data(coordinates, method, configs, problem_size)

        # 2. Prepare Edges
        edges = self._prepare_edges(dist_matrix, configs, device, edge_threshold, edge_method, adj_matrix)

        # 3. Load Profit Variables
        profit_vars = self._load_profit_vars(area, waste_type)

        # 4. Final Formatting
        dm_tensor = torch.as_tensor(dist_matrix, dtype=torch.float32, device=device)

        return (
            {k: v.unsqueeze(0) for k, v in model_data.items()},
            (edges, dm_tensor),
            profit_vars,
        )

    def save_results(self, matrix: List[List[float]], results_dir: str, seed: int, policy: str, sample_id: int) -> None:
        """Exports fill history to Excel.

        Args:
            matrix: Fill history matrix.
            results_dir: Directory to save results.
            seed: Random seed.
            policy: Policy used.
            sample_id: Sample ID.
        """
        path = os.path.join(results_dir, "fill_history")
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, f"{policy}{seed}_sample{sample_id}.xlsx")
        if not os.path.exists(filepath):
            pd.DataFrame(matrix).transpose().to_excel(filepath, index=False, header=False)
