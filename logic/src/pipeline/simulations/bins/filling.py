"""Daily waste filling, collection and fill-distribution setters of the bins.

:class:`BinsFillingMixin` holds the :class:`~logic.src.pipeline.simulations.bins.base.Bins`
methods that sample a day's waste, apply the stochastic, deterministic or loaded filling
(overflow, noise and statistics bookkeeping), collect a tour, and set the gamma fill
distribution and the collection level and frequency. Moved unchanged from ``base.py``;
the name-mangled helpers ``__get_stdev`` and ``__setDistribution`` move with their only
callers, so each pair stays in one class.
"""

import contextlib
import math
from typing import List, Optional, Tuple

import numpy as np

from logic.src.constants import MAX_CAPACITY_PERCENT

try:
    from logic.src.tracking.core.run import get_active_run
except ImportError:
    get_active_run = None  # type: ignore[assignment]
from logic.src.pipeline.simulations.bins.prediction import calculate_frequency_and_level


class BinsFillingMixin:
    """Waste sampling, filling, collection and fill-distribution methods of :class:`Bins` (moved unchanged from ``base.py``)."""

    def __get_stdev(self):
        """Computes current standard deviation.

        Returns:
            Numpy array of standard deviations for each bin.
        """
        if self.day_count > 1:
            variance = self.square_diff / (self.day_count - 1)
            return np.sqrt(variance)
        return np.zeros(self.n)

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
