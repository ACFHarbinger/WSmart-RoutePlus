"""
Bin State Management and Waste Accumulation Simulation.

This module provides the Bins class, which manages the state, daily waste
accumulation, and collection processes for a population of bins.

Attributes:
    Bins: Manager class for bin state and dynamics.

Example:
    >>> # bins = Bins(n=100, data_dir="data/vrpp")
    >>> # bins.collect(tour=[0, 1, 2, 0], cost=15.5)
"""

import contextlib
import math
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas
import torch

from logic.src.constants import MAX_CAPACITY_PERCENT, ROOT_DIR
from logic.src.data.datasets import (
    GenerativeDataset,
    HtmlSimulationDataset,
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

from logic.src.pipeline.simulations.bins.prediction import calculate_frequency_and_level, predict_days_to_overflow
from logic.src.pipeline.simulations.wsmart_bin_analysis import GridBase


class Bins:
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
            n_samples: Number of waste samples in the dataset.
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

    def __get_stdev(self):
        """Computes current standard deviation.

        Returns:
            Numpy array of standard deviations for each bin.
        """
        if self.day_count > 1:
            variance = self.square_diff / (self.day_count - 1)
            return np.sqrt(variance)
        return np.zeros(self.n)

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

    def get_fill_history(self, device: Optional[torch.device] = None) -> Union[np.ndarray, torch.Tensor]:
        """Retrieves history of daily fill increments.

        Args:
            device: Optional torch device to move the tensor to.

        Returns:
            Numpy array or Torch tensor of fill history.
        """
        if device is not None:
            return torch.tensor(np.array(self.history), dtype=torch.float, device=device) / 100.0
        return np.array(self.history)

    def get_level_history(self, device: Optional[torch.device] = None) -> Union[np.ndarray, torch.Tensor]:
        """Retrieves history of daily absolute fill levels.

        Args:
            device: Optional torch device to move the tensor to.

        Returns:
            Numpy array or Torch tensor of level history.
        """
        if device is not None:
            return torch.tensor(np.array(self.level_history), dtype=torch.float, device=device) / 100.0
        return np.array(self.level_history)

    def predict_days_to_overflow(self, cl: float) -> np.ndarray:
        """Predicts days to overflow based on statistics.

        Args:
            cl: Confidence level for prediction.

        Returns:
            Numpy array of predicted days to overflow.
        """
        return predict_days_to_overflow(self.means, self.std, self.c, cl)

    def set_indices(self, indices: Optional[Union[List[int], np.ndarray]] = None) -> None:
        """Sets subset of bin indices.

        Args:
            indices: Optional list or array of bin indices.
        """
        if indices is not None:
            self.indices = np.array(indices)
        else:
            self.indices = np.array(range(self.n))

    def set_sample_waste(self, sample_id: int, horizon: Optional[int] = None) -> None:
        """Sets current waste profile from dataset.

        Args:
            sample_id: Index of the waste sample to load.
            horizon: Simulation length in days. When a stats file is in use and
                the sample comes from a file, the sample must hold ``horizon + 1``
                rows (row 0 is the opening level). Generated samples are sized
                by the caller and are not checked here.
        """
        assert self.waste_dataset is not None
        sample = self.waste_dataset[sample_id]
        self.waste_fills = sample["waste"]
        self.noisy_waste_fills = sample["noisy_waste"]
        if self.start_with_fill and horizon is not None and not self.is_stochastic():
            self._require_external_stats_rows(int(horizon))
        if self.start_with_fill:
            self.real_c = self.waste_fills[0].copy()
            self.c = self.noisy_waste_fills[0].copy()
            self.level_history.append(self.c.copy())

        with contextlib.suppress(Exception):
            run = get_active_run()
            if run is not None:
                n_days = int(self.waste_fills.shape[0]) if hasattr(self.waste_fills, "shape") else 0
                run.log_params(
                    {
                        "sim.waste_sample_id": int(sample_id),
                        "sim.waste_n_days": n_days,
                        "sim.waste_n_bins": int(self.waste_fills.shape[1]) if n_days > 0 else self.n,
                    }
                )
                run.log_dataset_event(
                    "load",
                    shape=self.waste_fills.shape if hasattr(self.waste_fills, "shape") else (n_days,),
                    metadata={
                        "event": "waste_sample_load",
                        "sample_id": int(sample_id),
                        "variable_name": "self.waste_fills",
                        "source_file": "bins/base.py",
                        "source_line": 241,
                    },
                )

    def collect(self, idsfull: List[int], cost: float = 0) -> Tuple[np.ndarray, float, int, float]:
        """Processes waste collection from bins in the tour.

        Args:
            idsfull: List of bin IDs in the tour (1-indexed, 0 is depot).
            cost: Total travel distance for the tour.

        Returns:
            Tuple containing (collected_mass_vector, total_mass, n_bins_collected, profit).
        """
        ids = set(idsfull)
        total_collected = np.zeros((self.n))
        if len(ids) < 2:
            return total_collected, 0, 0, 0

        ids.remove(0)
        self.ndays += 1
        bin_ids = np.array(list(ids)) - 1
        collected = (self.real_c[bin_ids] / 100) * self.volume * self.density
        self.collected[bin_ids] += collected
        self.ncollections[bin_ids] += 1
        total_collected[bin_ids] += collected
        self.real_c[bin_ids] = 0
        self.c[bin_ids] = 0
        self.travel += cost
        profit = np.sum(total_collected) * self.revenue - float(cost) * self.expenses
        self.profit += float(profit)
        return total_collected, float(np.sum(collected)), bin_ids.size, float(profit)

    def _process_filling(
        self, todaysfilling: np.ndarray, noisyfilling: Optional[np.ndarray] = None
    ) -> Tuple[int, np.ndarray, np.ndarray, float]:
        """Processes daily waste deposition.

        Args:
            todaysfilling: Ground truth waste increments.
            noisyfilling: Optional sensor-observed increments.

        Returns:
            Tuple containing (n_overflows, filling_array, level_array, lost_waste).
        """
        self.history.append(todaysfilling)

        todaysfilling_arr = np.array(todaysfilling)
        old_means = self.means.copy()

        self.day_count += 1
        delta = todaysfilling_arr - old_means
        self.means += delta / self.day_count
        self.square_diff += delta * (todaysfilling_arr - self.means)
        self.std = self.__get_stdev()

        todays_lost = (
            (np.maximum(self.real_c + todaysfilling - MAX_CAPACITY_PERCENT, 0) / MAX_CAPACITY_PERCENT)
            * self.volume
            * self.density
        )
        todaysfilling = np.minimum(todaysfilling, MAX_CAPACITY_PERCENT)
        self.lost += todays_lost
        self.real_c = np.minimum(self.real_c + todaysfilling, MAX_CAPACITY_PERCENT)

        if noisyfilling is not None:
            self.c = np.minimum(self.c + noisyfilling, MAX_CAPACITY_PERCENT)
        elif self.noise_variance > 0:
            noise = self.rng.normal(self.noise_mean, np.sqrt(self.noise_variance), self.n)
            self.c = np.clip(self.real_c + noise, 0, MAX_CAPACITY_PERCENT)
        else:
            self.c = self.real_c.copy()

        self.level_history.append(self.c.copy())
        self.real_c = np.maximum(self.real_c, 0)
        inoverflow = self.real_c == MAX_CAPACITY_PERCENT
        self.inoverflow += inoverflow
        return (
            int(np.sum(inoverflow)),
            np.array(todaysfilling),
            np.array(self.c),
            float(np.sum(todays_lost)),
        )

    def deterministic_filling(self, date):
        """Loads deterministic fill levels from historical data.

        Args:
            date: Date string or object to load values for.

        Returns:
            The processed filling tuple.
        """
        todaysfilling = self.grid.get_values_by_date(date, sample=True)
        return self._process_filling(todaysfilling)

    def load_filling(self, day: int) -> Tuple[int, np.ndarray, np.ndarray, float]:
        """Loads deterministic waste fills from pre-recorded data.

        Args:
            day: Current simulation day index.

        Returns:
            The processed filling tuple.
        """
        if self.start_with_fill:
            # Row 0 is the opening level copied by set_sample_waste. Day d
            # deposits row d, so a horizon of n_days needs n_days + 1 rows.
            # Indexing day - 1 here would deposit the opening observation again.
            n_rows = len(self.waste_fills)
            if day < 0 or day >= n_rows:
                raise IndexError(
                    f"stats-file waste sample has {n_rows} rows; day {day} needs row {day} "
                    "because row 0 is the opening level and is not deposited again "
                    f"(a {day}-day horizon needs {day + 1} rows)"
                )
            todaysfilling = self.waste_fills[day]
            noisyfilling = self.noisy_waste_fills[day]
        else:
            todaysfilling = self.waste_fills[day - 1]
            noisyfilling = self.noisy_waste_fills[day - 1]
        return self._process_filling(todaysfilling, noisyfilling)

    def __setDistribution(self, param1, param2):
        """Internal helper to set sampling distribution parameters.

        Args:
            param1: First parameter (e.g., shape k).
            param2: Second parameter (e.g., scale theta).
        """
        if len(param1) == 1:
            self.dist_param1 = np.ones((self.n)) * param1
            self.dist_param2 = np.ones((self.n)) * param2
        else:
            self.dist_param1 = param1
            self.dist_param2 = param2
        self.set_collection_level_and_freq()

    def set_gamma_distribution(self, option=0):
        """Configures bins with predefined Gamma distribution profiles.

        Args:
            option: The predefined configuration index (0-3).

        Returns:
            None.
        """

        def __set_param(param):
            """set param.

            Args:
                param: Parameter vector or scalar.

            Returns:
                Tiled parameter vector matching n_bins.
            """
            param_len = len(param)
            if self.n == param_len:
                return param
            tiled = param * math.ceil(self.n / param_len)
            return tiled[: self.n]

        self.distribution = "gamma"
        if option == 0:
            k = __set_param([5, 5, 5, 5, 5, 10, 10, 10, 10, 10])
            th = __set_param([5, 2])
        elif option == 1:
            k = __set_param([2, 2, 2, 2, 2, 6, 6, 6, 6, 6])
            th = __set_param([6, 4])
        elif option == 2:
            k = __set_param([1, 1, 1, 1, 1, 3, 3, 3, 3, 3])
            th = __set_param([8, 6])
        else:
            assert option == 3
            k = __set_param([5, 2])
            th = __set_param([10])
        self.__setDistribution(k, th)
        with contextlib.suppress(Exception):
            run = get_active_run()
            if run is not None:
                run.log_params(
                    {
                        "sim.gamma_option": int(option),
                        "sim.gamma_k_mean": float(np.mean(self.dist_param1)),
                        "sim.gamma_theta_mean": float(np.mean(self.dist_param2)),
                    }
                )

    def set_collection_level_and_freq(self, cf=0.9):
        """Sets visit frequency and level thresholds.

        Args:
            cf: Confidence factor for frequency calculation.
        """
        for ii in range(0, self.n):
            f2, lv2 = calculate_frequency_and_level(
                self.dist_param1[ii] * self.dist_param2[ii] if self.distribution == "gamma" else self.dist_param1[ii],
                self.dist_param1[ii] * self.dist_param2[ii] ** 2
                if self.distribution == "gamma"
                else self.dist_param2[ii],
                cf,
            )
            self.collectdays[ii] = f2
            self.collectlevl[ii] = lv2
