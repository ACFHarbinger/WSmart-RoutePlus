"""
Bin State Management and Waste Accumulation Simulation.

This module provides the Bins class, which manages the state, daily waste
accumulation, and collection processes for a population of bins.

Attributes:
    Bins: Manager class for bin state and dynamics.

Example:
    >>> # bins = Bins(n=100, data_dir="data/ptp")
    >>> # bins.collect(tour=[0, 1, 2, 0], cost=15.5)

The waste sampling, filling, collection and fill-distribution methods of :class:`Bins`
live in the :class:`~logic.src.pipeline.simulations.bins.filling.BinsFillingMixin` mixin
(``filling.py``).
"""

import contextlib
import os
from typing import Any, List, Optional, Union

import numpy as np
import pandas

from logic.src.constants import ROOT_DIR
from logic.src.data.datasets import (
    GenerativeDataset,
    NumpyDictDataset,
    NumpyPickleDataset,
    PandasCsvDataset,
    PandasExcelDataset,
    SimulationDataset,
)
from logic.src.pipeline.simulations.repository import load_area_and_waste_type_params
from logic.src.utils.data.loader import load_grid_base

try:
    from logic.src.tracking.core.run import get_active_run
except ImportError:
    get_active_run = None  # type: ignore[assignment]

from logic.src.pipeline.simulations.bins.filling import BinsFillingMixin
from logic.src.pipeline.simulations.bins.prediction import predict_days_to_overflow
from logic.submodules.wsmart_bin_analysis import GridBase


class Bins(BinsFillingMixin):
    """
    Manages the state and dynamics of a population of waste collection bins.

    This class simulates the temporal evolution of bin fill levels through:
    - Daily waste deposition (stochastic or pre-recorded)
    - Collection events that empty bins
    - Overflow detection and waste loss calculation
    - Online statistical learning (mean, variance) via Welford's algorithm

    The Bins object maintains two parallel state vectors:
        - real_c: Ground truth fill levels (0-100%)
        - c: Observed fill levels (may include sensor noise)

    Attributes:
        n (int): Number of bins.
        data_dir (str): Path to data directory.
        sample_dist (str, optional): Sampling distribution. Defaults to "gamma".
        grid (Optional[GridBase], optional): Grid object. Defaults to None.
        area (Optional[str], optional): Area name. Defaults to None.
        waste_type (Optional[str], optional): Waste type. Defaults to None.
        indices (Optional[Union[np.ndarray, List[int]]], optional): List of bin indices. Defaults to None.
        waste_file (Optional[str], optional): Path to waste file. Defaults to None.
        noise_mean (float, optional): Noise mean. Defaults to 0.0.
        noise_variance (float, optional): Noise variance. Defaults to 0.0.
        n_days (int, optional): Number of days to simulate. Defaults to 31.
        n_samples (int, optional): Number of samples to generate. Defaults to 1.
        seed (Optional[int], optional): Random seed. Defaults to None.
    """

    def __init__(
        self,
        n: int,
        data_dir: str,
        sample_dist: str = "gamma",
        grid: Optional[GridBase] = None,
        area: Optional[str] = None,
        waste_type: Optional[str] = None,
        indices: Optional[Union[np.ndarray, List[int]]] = None,
        waste_file: Optional[str] = None,
        noise_mean: float = 0.0,
        noise_variance: float = 0.0,
        n_days: int = 31,
        n_samples: int = 1,
        seed: Optional[int] = None,
    ):
        """
        Initializes the bin population with area-specific parameters.
        Args:
            n: Number of bins.
            data_dir: Path to the data directory.
            sample_dist: Sampling distribution (e.g., "gamma", "emp").
            grid: GridBase object for empirical data.
            area: Name of the area for parameter loading.
            waste_type: Type of waste (e.g., "all", "glass").
            indices: Subset of bin indices to use.
            waste_file: Path to a file containing pre-recorded waste data.
            noise_mean: Mean of the sensor noise.
            noise_variance: Variance of the sensor noise.
            n_days: Total number of simulation days.
            n_samples: Number of samples to generate.
            seed: Random seed for reproducibility.
        """
        assert sample_dist == "emp" or "gamma" in sample_dist
        self.n = n
        self.noise_mean = noise_mean
        self.noise_variance = noise_variance
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        _, revenue, density, expenses, bin_volume = load_area_and_waste_type_params(area, waste_type)
        self.revenue = revenue
        self.density = density
        self.expenses = expenses
        self.volume = bin_volume

        self.c = np.zeros((n))
        self.real_c = np.zeros((n))
        self.means = np.zeros((n))
        self.std = np.zeros((n))
        self.day_count = 0
        self.square_diff = np.zeros((n))
        self.start_with_fill = False

        self.lost = np.zeros((n))
        self.distribution = sample_dist
        self.inoverflow = np.zeros((n))
        self.collected = np.zeros((n))
        self.ncollections = np.zeros((n))
        self.history: List[np.ndarray] = []
        self.level_history: List[np.ndarray] = []
        self.travel: float = 0.0
        self.profit: float = 0.0
        self.ndays: int = 0
        self.collectdays = np.ones((n)) * 5
        self.collectlevl = np.ones((n)) * 80
        self.data_dir = data_dir
        self.indices = np.array(indices) if indices is not None else np.arange(n)
        if grid is not None:
            self.grid = grid
        else:
            try:
                self.grid = load_grid_base(self.indices, area, data_dir)
            except FileNotFoundError:
                if sample_dist == "emp":
                    raise  # GridBase already provides the specific file path in the message
                self.grid = None  # type: ignore[assignment]

        if sample_dist == "emp":
            self.dist_param1 = self.grid.get_mean_rate()
            self.dist_param2 = self.grid.get_var_rate()
        else:
            self.dist_param1 = np.ones((n)) * 10
            self.dist_param2 = np.ones((n)) * 10

        self.waste_dataset: Optional[SimulationDataset] = None
        if waste_file is not None:
            if os.path.isabs(waste_file):
                path = waste_file
            elif waste_file.startswith("data/"):
                path = os.path.join(ROOT_DIR, waste_file)
            else:
                path = os.path.join(data_dir, waste_file)

            print(f"\n[INFO] Loading data from '{path}'...")
            if waste_file.endswith(".pkl"):
                self.waste_dataset = NumpyPickleDataset.load(path, area=area, waste_type=waste_type)
            elif waste_file.endswith(".xlsx"):
                self.waste_dataset = PandasExcelDataset.load(path, area=area, waste_type=waste_type)
            elif waste_file.endswith(".csv"):
                self.waste_dataset = PandasCsvDataset.load(path, area=area, waste_type=waste_type)
            elif waste_file.endswith(".html") or waste_file.endswith(".htm"):
                from logic.src.data.datasets import HtmlSimulationDataset

                self.waste_dataset = HtmlSimulationDataset.load(path, area=area, waste_type=waste_type, n_days=n_days)
            else:
                self.waste_dataset = NumpyDictDataset.load(path, area=area, waste_type=waste_type)
        else:
            print("\n[INFO] Generating data...")
            self.waste_dataset = GenerativeDataset(
                data_dir=data_dir,
                n_samples=n_samples,
                n_days=n_days,
                n_bins=n,
                distribution=sample_dist,
                noise_mean=noise_mean,
                noise_variance=noise_variance,
                grid=self.grid,
                seed=seed,
            )

        with contextlib.suppress(Exception):
            run = get_active_run()
            if run is not None:
                run.log_params(
                    {
                        "sim.n_bins": n,
                        "sim.distribution": sample_dist,
                        "sim.area": str(area) if area is not None else "",
                        "sim.waste_type": str(waste_type) if waste_type is not None else "all",
                        "sim.noise_mean": noise_mean,
                        "sim.noise_variance": noise_variance,
                        "sim.is_stochastic": isinstance(self.waste_dataset, GenerativeDataset),
                        "sim.has_waste_file": waste_file is not None,
                    }
                )

    def set_statistics(self, stats_file: str) -> None:
        """Loads pre-computed fill statistics.

        Args:
            stats_file: Path to the statistics CSV file.
        """
        if os.path.isabs(stats_file):
            path = stats_file
        elif stats_file.startswith("data/"):
            path = os.path.join(ROOT_DIR, stats_file)
        else:
            path = os.path.join(self.data_dir, stats_file)

        data = pandas.read_csv(path)
        if "ID" in data.columns:
            data = data[data["ID"] != 0].reset_index(drop=True)

        self.means = np.maximum(data["Mean"].values.astype(np.float64), 0)
        self.std = np.maximum(data["StD"].values.astype(np.float64), 0)
        self.day_count = np.maximum(data.at[0, "Count"].astype(np.int64), 0)
        self.square_diff = (self.std**2) * (self.day_count - 1)
        self.start_with_fill = True
        with contextlib.suppress(Exception):
            run = get_active_run()
            if run is not None:
                run.log_params(
                    {
                        "sim.stats_file": str(stats_file),
                        "sim.stats_n_bins": len(data),
                        "sim.stats_mean_fill": float(self.means.mean()),
                        "sim.stats_mean_std": float(self.std.mean()),
                        "sim.stats_day_count": int(self.day_count),
                    }
                )
                run.log_dataset_event(
                    "load",
                    file_path=str(os.path.join(self.data_dir, stats_file)),
                    shape=data.shape,
                    metadata={
                        "event": "fill_stats_load",
                        "variable_name": "self.means/self.std",
                        "source_file": "bins/base.py",
                        "source_line": 182,
                    },
                )

    def _require_external_stats_rows(self, horizon: int) -> None:
        """Reject a file-backed stats-file sample that cannot cover the horizon.

        Args:
            horizon: Number of simulated days.

        Raises:
            ValueError: The sample has fewer than ``horizon + 1`` rows.
        """
        rows = len(self.waste_fills)
        noisy_rows = len(self.noisy_waste_fills)
        need = horizon + 1
        if rows < need or noisy_rows < need:
            raise ValueError(
                f"stats-file waste sample has {rows} rows; a {horizon}-day horizon needs {need} "
                "because row 0 is the opening level and is not deposited again"
            )

    def is_stochastic(self) -> bool:
        """Checks if using stochastic filling.

        Returns:
            True if the waste dataset is generative/stochastic.
        """
        return isinstance(self.waste_dataset, GenerativeDataset)

    def get_fill_history(self, device: Optional[Any] = None) -> Any:
        """Retrieves history of daily fill increments.

        Args:
            device: Optional torch device to move the tensor to. NumPy is returned
                when no device is requested, so the classical path does not import torch.

        Returns:
            Numpy array or Torch tensor of fill history.
        """
        if device is not None:
            import torch

            return torch.tensor(np.array(self.history), dtype=torch.float, device=device) / 100.0
        return np.array(self.history)

    def get_level_history(self, device: Optional[Any] = None) -> Any:
        """Retrieves history of daily absolute fill levels.

        Args:
            device: Optional torch device to move the tensor to.

        Returns:
            Numpy array or Torch tensor of level history.
        """
        if device is not None:
            import torch

            return torch.tensor(np.array(self.level_history), dtype=torch.float, device=device) / 100.0
        return np.array(self.level_history)

    def predict_days_to_overflow(self, cl: float) -> np.ndarray:
        """Predicts days to overflow based on statistics.

        Args:
            cl: Confidence level for prediction.

        Returns:
            Numpy array of predicted days to overflow.
        """
        # predict_days_to_overflow expects the fill-rate VARIANCE (see its
        # docstring and the gamma method-of-moments mapping k = ui^2/vi);
        # self.std is a standard deviation, so square it here.
        return predict_days_to_overflow(self.means, self.std**2, self.c, cl)

    def set_indices(self, indices: Optional[Union[List[int], np.ndarray]] = None) -> None:
        """Sets subset of bin indices.

        Args:
            indices: Optional list or array of bin indices.
        """
        if indices is not None:
            self.indices = np.array(indices)
        else:
            self.indices = np.array(range(self.n))
