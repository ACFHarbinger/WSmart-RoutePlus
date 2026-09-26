"""
Abstract interface for simulation data access.

This module defines the base class for data repositories used by the
simulation pipeline to load areas, depots, and waste configurations.

Attributes:
    SimulationRepository: Abstract base class for data access.

Example:
    >>> # class MyRepo(SimulationRepository): ...
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class SimulationRepository(ABC):
    """
    Abstract interface for simulation data access.

    Defines the contract for loading geographic, waste, and configuration
    data required to initialize simulations. Implementations can source
    data from files, databases, APIs, or other backends.

    Attributes:
        None
    """

    @abstractmethod
    def get_indices(
        self, filename: Any, n_samples: int, n_nodes: int, data_size: int, lock: Optional[Any] = None
    ) -> List[List[int]]:
        """
        Loads or generates a list of bin indices for simulation samples.

        Args:
            filename: Path to the index file.
            n_samples: Number of samples to generate/load.
            n_nodes: Number of nodes per sample.
            data_size: Total size of the source data.
            lock: Optional file lock for concurrent access.

        Returns:
            A list of lists containing bin indices for each sample.
        """
        pass

    @abstractmethod
    def get_depot(self, area: Any, data_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieves the depot coordinates for a given area.

        Args:
            area: Name of the geographic area.
            data_dir: Path to the data directory.

        Returns:
            Pandas DataFrame containing depot coordinates.
        """
        pass

    @abstractmethod
    def get_simulator_data(
        self,
        number_of_bins: int,
        area: str = "Rio Maior",
        waste_type: Optional[str] = None,
        lock: Optional[Any] = None,
        data_dir: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads waste statistics and coordinate data for the simulator.

        Args:
            number_of_bins: Number of bins to load.
            area: Name of the geographic area.
            waste_type: Type of waste.
            lock: Optional file lock.
            data_dir: Path to the data directory.

        Returns:
            Tuple containing (stats_df, coordinates_df).
        """
        pass

    @staticmethod
    def get_area_params(area: Any, waste_type: Any) -> Tuple[float, float, float, float, float]:
        """
        Retrieves area and waste-type specific simulation parameters.

        Returns physical and economic parameters calibrated for specific
        geographic areas and waste streams. Values are based on real-world
        data from Portuguese waste management operations.

        Args:
            area: Geographic area ('Rio Maior', 'Figueira da Foz', etc.)
            waste_type: Waste stream ('paper', 'plastic', 'glass')

        Returns:
            Tuple containing:
                - vehicle_capacity: Max bin capacity units per vehicle (%)
                - revenue: Revenue per kg of collected waste (€/kg)
                - density: Waste density (kg/L)
                - expenses: Cost per km traveled (€/km)
                - bin_volume: Individual bin volume (L)

        Raises:
            AssertionError: If waste_type or area not recognized
        """
        expenses = 1.0
        bin_volume = 2.5
        src_area = area.translate(str.maketrans("", "", "-_ ")).lower() if area is not None else ""

        # Normalize waste_type: treat None or empty as "glass" (the default)
        waste_type = waste_type or "glass"

        if waste_type == "paper":
            revenue = 0.65 * 250 / 1000
            if src_area in ["riomaior", "mixrmbac"]:
                density = 21.0
                vehicle_capacity = 4000.0
            else:
                assert src_area == "figueiradafoz", f"Unknown waste collection area: {src_area}"
                density = 32.0
                vehicle_capacity = 3000.0

        elif waste_type == "plastic":
            revenue = 0.65 * 898 / 1000
            if src_area in ["riomaior", "mixrmbac"]:
                density = 19.0
                vehicle_capacity = 3500.0
            else:
                assert src_area == "figueiradafoz", f"Unknown waste collection area: {src_area}"
                density = 20.0
                vehicle_capacity = 2500.0

        else:
            assert waste_type == "glass", f"Unknown waste type: {waste_type}"
            revenue = 0.90 * 84 / 1000
            if src_area in ["riomaior", "mixrmbac"]:
                density = 190.0
                vehicle_capacity = 9000.0
            else:
                assert src_area == "figueiradafoz", f"Unknown waste collection area: {src_area}"
                density = 200.0
                vehicle_capacity = 8000.0

        # Calculate percentage capacity
        vehicle_capacity = (vehicle_capacity / (bin_volume * density)) * 100
        return (vehicle_capacity, revenue, density, expenses, bin_volume)

    @staticmethod
    def get_temporal_params(
        coords: Optional[pd.DataFrame] = None,
        distance_matrix: Optional[np.ndarray] = None,
        tm_filepath: Optional[str] = None,
        shift_hours: float = 7.0,
        avg_speed_kmh: float = 35.0,
        service_time_h: float = 1.5 / 60.0,
        time_unit: str = "seconds",
    ) -> Tuple[float, Union[float, np.ndarray], float]:
        """
        Retrieves the temporal resource parameters for time-constrained problems.

        Returns the working-shift budget and travel/service rates used by
        time-constrained problem variants alongside their capacity constraint.
        With coordinates and distances, the second value is a pairwise travel
        time matrix in hours, loaded from ``tm_filepath`` or computed using
        the uniform speed. Without matrix inputs the scalar-speed API remains
        available to generated routing environments.

        Returns:
            Tuple containing:
                - shift_hours: Total time budget per period/trip (h)
                - travel: Average speed (km/h), or travel-time matrix (h)
                - service_time_h: Time to visit and empty a single bin (h)
        """
        if not np.isfinite(shift_hours) or shift_hours <= 0:
            raise ValueError("shift_hours must be finite and positive")
        if not np.isfinite(avg_speed_kmh) or avg_speed_kmh <= 0:
            raise ValueError("avg_speed_kmh must be finite and positive")
        if not np.isfinite(service_time_h) or service_time_h < 0:
            raise ValueError("service_time_h must be finite and nonnegative")
        if coords is not None and distance_matrix is not None:
            from logic.src.data.time import compute_time_matrix

            return (
                shift_hours,
                compute_time_matrix(coords, distance_matrix, tm_filepath, avg_speed_kmh, time_unit),
                service_time_h,
            )
        if coords is not None or distance_matrix is not None or tm_filepath is not None:
            raise ValueError("Travel-time loading requires both coordinates and distances")
        return (shift_hours, avg_speed_kmh, service_time_h)
