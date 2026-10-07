"""
Builders for Vehicle Routing Problem (VRP) instances.

This module provides builder classes to construct VRP instances from raw data,
handling coordinate normalization, waste scaling, and feature extraction.

Attributes:
    VRPInstanceBuilder: Builder pattern for creating VRP instances.

Example:
    builder = VRPInstanceBuilder()
    builder.set_problem_size(10)
    builder.set_waste_type("plastic")
    builder.set_distribution("gamma1")
    builder.set_area("Rio Maior")
    builder.set_focus_graph("graph.pkl", 20)
    builder.set_method("random")
    builder.set_num_days(1)
    builder.set_problem_name("vrpp")
    builder.set_noise(0.0, 0.0)
    builder.build()
"""

from typing import Optional

import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict

from logic.src.constants import MAX_CAPACITY_PERCENT, MAX_WASTE
from logic.src.data.generators.waste import generate_waste
from logic.src.data.processor import process_coordinates
from logic.src.utils.data.loader import load_focus_coords, load_grid_base
from logic.src.utils.functions.path import get_path_until_string


class VRPInstanceBuilder:
    """
    Builder pattern for creating Vehicle Routing Problem (VRP) instances.

    This class provides a fluent interface to configure and generate VRP datasets
    with various parameters such as problem size, distribution, area, and waste type.
    Attributes:
        dataset_size (int): Number of instances to generate.
        problem_size (int): Number of nodes (graph size).
        waste_type (Optional[str]): Type of waste.
        distribution (str): Data distribution.
        area (str): Geographical area.
        focus_graph (Optional[str]): Path to focus graph file.
        focus_size (int): Size of focus graph.
        method (Optional[str]): Method for vertex generation/selection.
        num_days (int): Number of simulation days.
        problem_name (Optional[str]): Name of the problem.
        noise_mean (float): Mean for noise injection.
        noise_variance (float): Variance for noise injection.
        device (torch.device): Device for random number generation.
        np_rng (np.random.Generator): Random number generator.
        generator (torch.Generator): PyTorch generator.
    """

    def __init__(
        self,
        data=None,
        depot_idx=0,
        vehicle_cap=100.0,
        customers=None,
        dimension=0,
        coords=None,
        device="cpu",
    ):
        """
        Initialize the VRPInstanceBuilder.

        Args:
            data: Raw problem data dictionary or dataframe.
            depot_idx (int): Index of the depot node.
            vehicle_cap (float): Capacity of the vehicles.
            customers: List of customer node indices.
            dimension (int): Total number of nodes (including depot).
            coords: Coordinate data (list or array).
            device (str): Device for random number generation.
        """
        self.data_dict = data
        self.depot_idx = depot_idx
        self.vehicle_cap = vehicle_cap
        self.customers = customers if customers is not None else []
        self.dimension = dimension
        self.coords = coords

        self._dataset_size = 10
        self._problem_size = 20
        self._waste_type = None
        self._distribution = "gamma1"
        self._area = "Rio Maior"
        self._focus_graph = None
        self._focus_size = 0
        self._method = None
        self._num_days = 1
        self._problem_name = None
        self._noise_mean = 0.0
        self._noise_variance = 0.0
        self._device = torch.device(device)
        self._seed = None
        self.np_rng = np.random.default_rng()
        self.generator = torch.Generator(device=self._device)

    def set_seed(self, seed: int):
        """Sets the random seed for reproducibility.

        Args:
            seed: Description of seed.

        Returns:
            Description of return value.
        """
        self._seed = seed
        self.np_rng = np.random.default_rng(seed)
        self.generator.manual_seed(seed)
        return self

    def set_dataset_size(self, size: int):
        """Sets the number of instances to generate.

        Args:
            size: Description of size.

        Returns:
            Description of return value.
        """
        self._dataset_size = size
        return self

    def set_problem_size(self, size: int):
        """Sets the number of nodes (graph size) for the problem.

        Args:
            size: Description of size.

        Returns:
            Description of return value.
        """
        self._problem_size = size
        return self

    def set_waste_type(self, waste_type: str):
        """Sets the type of waste (e.g., 'plastic', 'paper').

        Args:
            waste_type: Description of waste_type.

        Returns:
            Description of return value.
        """
        self._waste_type = waste_type
        return self

    def set_distribution(self, distribution: str):
        """Sets the data distribution for generating waste levels.

        Args:
            distribution: Description of distribution.

        Returns:
            Description of return value.
        """
        self._distribution = distribution
        return self

    def set_area(self, area: str):
        """Sets the geographical area for the problem instance.

        Args:
            area: Description of area.

        Returns:
            Description of return value.
        """
        self._area = area
        return self

    def set_focus_graph(self, focus_graph: Optional[str] = None, focus_size: int = 0):
        """Sets parameters for focusing on a specific subgraph.

        Args:
            focus_graph: Description of focus_graph.
            focus_size: Description of focus_size.

        Returns:
            Description of return value.
        """
        self._focus_graph = focus_graph
        self._focus_size = focus_size
        return self

    def set_method(self, method: Optional[str]):
        """Sets the method used for vertex generation/selection.

        Args:
            method: Description of method.

        Returns:
            Description of return value.
        """
        self._method = method
        return self

    def set_num_days(self, num_days: int):
        """Sets the number of simulation days.

        Args:
            num_days: Description of num_days.

        Returns:
            Description of return value.
        """
        self._num_days = num_days
        return self

    def set_problem_name(self, problem_name: str):
        """Sets the name of the problem (e.g., 'vrpp', 'cvrpp').

        Args:
            problem_name: Description of problem_name.

        Returns:
            Description of return value.
        """
        self._problem_name = problem_name
        return self

    def set_noise(self, mean: float, variance: float):
        """Sets the mean and variance for noise injection.

        Args:
            mean: Description of mean.
            variance: Description of variance.

        Returns:
            Description of return value.
        """
        self._noise_mean = mean
        self._noise_variance = variance
        return self

    def build(self):
        """
        Generates the simulation dataset as a dict of numpy arrays.

        Returns:
            dict: A dictionary of numpy arrays:
                - 'depot': np.ndarray of shape (dataset_size, coord_dim)
                - 'locs': np.ndarray of shape (dataset_size, problem_size, coord_dim)
                - 'waste': np.ndarray of shape (dataset_size, num_days, problem_size)
                - 'noisy_waste': np.ndarray of same shape as waste (equals waste)
                - 'max_waste': np.ndarray of shape (dataset_size,)
        """
        depot, loc, grid, idx, node_ids = self._prepare_coordinates()

        # Generate waste/fill values over days
        fill_values = []
        coords = (depot, loc)

        for _ in range(self._num_days):
            waste = generate_waste(
                self._problem_size, self._distribution, coords, self._dataset_size, grid, rng=self.np_rng
            )
            if self._dataset_size == 1 and len(waste.shape) == 1:
                waste = waste[None, :]
            fill_values.append(waste)

        # Shape: (dataset_size, num_days, problem_size)
        fill_arr = np.transpose(np.array(fill_values), (1, 0, 2))

        # generate_waste returns values in [0, 1] normalized scale;
        # simulation datasets store in [0, 100] percentage scale to match Bins.
        fill_arr = fill_arr * MAX_CAPACITY_PERCENT

        noisy_fill_arr = fill_arr.copy()

        # Tile node_ids to match dataset_size (bs, n_nodes) - exclude depot (index 0)
        node_ids_val = node_ids.values if hasattr(node_ids, "values") else node_ids
        tiled_node_ids = np.tile(node_ids_val[1:], (self._dataset_size, 1))

        return {
            "depot": np.flip(depot, axis=-1) if depot.shape[-1] == 2 else depot,
            "locs": np.flip(loc, axis=-1) if loc.shape[-1] == 2 else loc,
            "node_ids": tiled_node_ids,
            "waste": fill_arr,
            "noisy_waste": noisy_fill_arr,
            "max_waste": np.full(self._dataset_size, MAX_CAPACITY_PERCENT),
        }

    def build_td(self) -> TensorDict:
        """
        Generates the dataset as a batched TensorDict.

        Returns:
            TensorDict: A TensorDict with keys matching environment expectations.
        """
        depot, loc, grid, idx, node_ids = self._prepare_coordinates()

        fill_values = []
        coords = (depot, loc)
        for _ in range(self._num_days):
            waste = generate_waste(
                self._problem_size, self._distribution, coords, self._dataset_size, grid, rng=self.generator
            )
            if self._dataset_size == 1 and len(waste.shape) == 1:
                waste = waste[None, :]
            fill_values.append(waste)

        fill_vals = np.array(fill_values)  # (num_days, dataset_size, num_loc)
        fill_vals = np.transpose(fill_vals, (1, 0, 2))  # (dataset_size, num_days, num_loc)

        bs = self._dataset_size
        device = "cpu"  # Generate on CPU

        depot_tensor = torch.tensor(depot, dtype=torch.float32, device=device)
        locs_tensor = torch.tensor(loc, dtype=torch.float32, device=device)

        waste = torch.tensor(fill_vals, dtype=torch.float32, device=device)
        if self._num_days == 1:
            waste = waste.squeeze(1)

        td_data = {
            "depot": depot_tensor,
            "locs": locs_tensor,
            "waste": torch.clamp(waste, 0, MAX_WASTE),
        }

        # Common attributes - exclude depot (index 0) from node_ids to match locs
        node_ids_val = node_ids.values if hasattr(node_ids, "values") else node_ids
        node_ids_tensor = torch.tensor(node_ids_val[1:], dtype=torch.float32, device=device).repeat(bs, 1)

        td_data.update(
            {
                "node_ids": node_ids_tensor,
                "capacity": torch.full((bs,), self.vehicle_cap, device=device),
                "max_waste": torch.full((bs,), float(MAX_WASTE), device=device),
            }
        )

        return TensorDict(td_data, batch_size=[bs], device=device)

    def _prepare_coordinates(self):
        """Internal helper to prepare depot and location coordinates.

        Returns:
            Description of return value.
        """
        if self._focus_graph is not None:
            if self._focus_size <= 0:
                self._focus_size = self._dataset_size
            depot, loc, mm_arr, idx, node_ids = load_focus_coords(
                self._problem_size,
                self._method,
                self._area,
                self._waste_type,  # type: ignore[arg-type]
                self._focus_graph,
                self._focus_size,
            )
            remaining_coords_size = self._dataset_size - self._focus_size
            if remaining_coords_size > 0:
                random_coords = self.np_rng.uniform(
                    mm_arr[0],  # type: ignore[index]
                    mm_arr[1],  # type: ignore[index]
                    size=(remaining_coords_size, self._problem_size + 1, mm_arr.shape[-1]),  # type: ignore[union-attr]
                )
                depots, locs = process_coordinates(random_coords, self._method, col_names=None)
                depot = np.concatenate((depot, depots))  # type: ignore[assignment]
                loc = np.concatenate((loc, locs))  # type: ignore[assignment]
            grid = None
            if self._distribution == "emp":
                data_dir = get_path_until_string(self._focus_graph, "simulator")
                # node_ids row 0 is the depot; the grid must follow the focus graph's bins.
                grid = load_grid_base(idx, self._area, data_dir, ids=list(node_ids.iloc[1:]))
        else:
            grid = None
            idx = np.array(range(self._problem_size))
            node_ids = pd.Series(idx)
            coord_size = 2 if self._method != "triple" else 3
            depot = self.np_rng.uniform(size=(self._dataset_size, coord_size))
            loc = self.np_rng.uniform(size=(self._dataset_size, self._problem_size, coord_size))
        return depot, loc, grid, idx, node_ids
