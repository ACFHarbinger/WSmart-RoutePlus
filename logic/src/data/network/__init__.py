"""
Distance Calculation and Network Sparsification.

Attributes:
    haversine_distance: Stand-alone vectorized Haversine distance calculation.
    compute_distance_matrix: Computes or loads cached pairwise distance matrix for bin locations.
    DistanceStrategy: Base class for distance calculation strategies.
    IterativeDistanceStrategy: Base class for iterative distance calculation strategies.
    EuclideanStrategy: Computes distances using Euclidean metric.
    FileStrategy: Reads precomputed distances from a file.
    GeodesicStrategy: Computes great-circle distances on a sphere.
    GeoPandasStrategy: Computes distances using GeoPandas.
    GoogleMapsStrategy: Computes distances using Google Maps API.
    HaversineStrategy: Computes distances using Haversine formula.
    OSMStrategy: Computes distances using OpenStreetMap data.

Example:
    from logic.src.data.network import haversine_distance, compute_distance_matrix
    haversine_distance(lat1, lng1, lat2, lng2)
    compute_distance_matrix(coords, method="haversine")
"""

import contextlib
import os
from typing import Any, Union

import numpy as np
import pandas as pd

from logic.src.constants import EARTH_RADIUS, ROOT_DIR

try:
    from logic.src.tracking.core.run import get_active_run
except ImportError:
    get_active_run = None  # type: ignore[assignment]

from .base import DistanceStrategy, IterativeDistanceStrategy
from .euclidean import EuclideanStrategy
from .file import FileStrategy
from .geodesic import GeodesicStrategy
from .geopandas import GeoPandasStrategy
from .google import GoogleMapsStrategy
from .haversine import HaversineStrategy
from .osm import OSMStrategy


def haversine_distance(
    lat1: Union[float, np.ndarray, pd.Series],
    lng1: Union[float, np.ndarray, pd.Series],
    lat2: Union[float, np.ndarray, pd.Series],
    lng2: Union[float, np.ndarray, pd.Series],
) -> Union[float, np.ndarray]:
    """
    Stand-alone vectorized Haversine distance calculation.

    Args:
        lat1: Latitude(s) of the first point(s).
        lng1: Longitude(s) of the first point(s).
        lat2: Latitude(s) of the second point(s).
        lng2: Longitude(s) of the second point(s).

    Returns:
        Distance(s) between the first and second point(s).
    """
    lat1, lng1, lat2, lng2 = np.radians(lat1), np.radians(lng1), np.radians(lat2), np.radians(lng2)
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS * c


def compute_distance_matrix(coords: pd.DataFrame, method: str, **kwargs: Any) -> np.ndarray:
    """
    Computes or loads cached pairwise distance matrix for bin locations.

    Args:
        coords: DataFrame with coordinates (must contain 'ID', 'lat', 'lng' columns).
        method: Distance calculation method (e.g., 'haversine', 'gmaps', 'osm').
        kwargs: Additional arguments for the distance strategy.

    Returns:
        Distance matrix as a NumPy array.
    """
    STRATEGIES = {
        "gmaps": GoogleMapsStrategy,
        "gpd": GeoPandasStrategy,
        "osm": OSMStrategy,
        "gdsc": GeodesicStrategy,
        "hsd": HaversineStrategy,
        "ogd": EuclideanStrategy,
        "file": FileStrategy,
    }

    assert method in STRATEGIES, f"Method {method} not supported. usage: {list(STRATEGIES.keys())}"

    # Caching Logic
    to_save = False
    matrix_path = None

    if "dm_filepath" in kwargs and kwargs["dm_filepath"] is not None:
        dm_filepath = kwargs["dm_filepath"]
        filename_only = os.path.basename(dm_filepath) == dm_filepath and not os.path.isabs(dm_filepath)
        matrix_path = (
            os.path.join(
                ROOT_DIR,
                "data",
                "simulator",
                "distance_matrix",
                dm_filepath,
            )
            if filename_only
            else dm_filepath
        )

        if os.path.isfile(matrix_path):
            # Use FileStrategy for loading
            strategy = FileStrategy()
            distance_matrix = strategy.calculate(coords, **kwargs)
            with contextlib.suppress(Exception):
                run = get_active_run() if get_active_run is not None else None
                if run is not None:
                    run.log_params(
                        {
                            "data.dist_method": method,
                            "data.dist_matrix_source": "file",
                            "data.dist_matrix_n_nodes": distance_matrix.shape[0],
                        }
                    )
                    run.log_dataset_event(
                        "load",
                        file_path=str(matrix_path),
                        shape=distance_matrix.shape,
                        metadata={
                            "event": "dist_matrix_load",
                            "method": method,
                            "variable_name": "distance_matrix",
                            "source_file": "network/__init__.py",
                            "source_line": 90,
                        },
                    )
            return distance_matrix
        elif method == "file":
            raise FileNotFoundError(
                f"Distance matrix file not found: {matrix_path}\n"
                "Generate it first with:\n"
                "  uv run python logic/gen/gen_dist_matrix.py "
                f"--area <area> --waste-type <type> --method gmaps "
                f"--dm-filepath {os.path.basename(matrix_path)}"
            )
        else:
            # Prepare for saving after computation
            os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
            with open(matrix_path, mode="w", newline="") as matrix_f:
                matrix_f.write(",".join(map(str, coords["ID"].to_numpy())) + "\n")
            to_save = True

    # Strategy Execution
    strategy_cls = STRATEGIES[method]
    strategy = strategy_cls()  # type: ignore[abstract, assignment]

    kwargs["verbose"] = to_save or kwargs.get("verbose", False)

    distance_matrix = strategy.calculate(coords, **kwargs)

    # Saving Result
    if to_save:
        with open(matrix_path, mode="a", newline="") as matrix_f:  # type: ignore[arg-type]
            for row in distance_matrix:
                matrix_f.write(",".join(map(str, row)) + "\n")

    with contextlib.suppress(Exception):
        run = get_active_run() if get_active_run is not None else None
        if run is not None:
            n_nodes = distance_matrix.shape[0]
            off_diag = distance_matrix[~np.eye(n_nodes, dtype=bool)]
            positive = off_diag[off_diag > 0]
            run.log_params(
                {
                    "data.dist_method": method,
                    "data.dist_matrix_source": "computed",
                    "data.dist_matrix_n_nodes": n_nodes,
                    "data.dist_matrix_min_km": float(positive.min()) if len(positive) > 0 else 0.0,
                    "data.dist_matrix_max_km": float(distance_matrix.max()),
                    "data.dist_matrix_mean_km": float(positive.mean()) if len(positive) > 0 else 0.0,
                }
            )
            if matrix_path:
                run.log_dataset_event(
                    "generate",
                    file_path=str(matrix_path),
                    shape=distance_matrix.shape,
                    metadata={
                        "event": "dist_matrix_compute",
                        "method": method,
                        "variable_name": "distance_matrix",
                        "source_file": "network/__init__.py",
                        "source_line": 136,
                    },
                )

    return distance_matrix


__all__ = [
    "DistanceStrategy",
    "IterativeDistanceStrategy",
    "GoogleMapsStrategy",
    "GeoPandasStrategy",
    "OSMStrategy",
    "GeodesicStrategy",
    "HaversineStrategy",
    "EuclideanStrategy",
    "FileStrategy",
    "haversine_distance",
    "compute_distance_matrix",
]
