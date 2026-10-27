"""Data transformation functions for coordinates and model inputs.

Attributes:
    _mapper: Instance of SimulationDataMapper for processing.
    process_data: Process raw bin data and coordinates for simulation use.
    process_coordinates: Normalize and format coordinates using the specified method.
    process_model_data: Prepare data for neural model inputs.

Example:
    from logic.src.data.processor import process_data, process_coordinates, process_model_data
    processed_data, processed_coordinates = process_data(data, bins_coordinates, depot)
    depot, locations = process_coordinates(processed_coordinates, method)
    model_data, edges_and_dm, profit_vars = process_model_data(depot, loc, method)
"""

import contextlib

try:
    from logic.src.tracking.core.run import (  # pyrefly: ignore[missing-import]  # tracking is excluded from the minimal package
        get_active_run,
    )
except ImportError:
    get_active_run = None  # type: ignore[assignment]

from ._logging import _log_processor_event
from .formatting import format_coordinates
from .mapper import SimulationDataMapper

_mapper = SimulationDataMapper()


def process_data(data, bins_coordinates, depot, indices=None):
    """Process raw bin data and coordinates for simulation use.

    Args:
        data: Raw bin statistics DataFrame.
        bins_coordinates: Bin coordinates DataFrame.
        depot: Depot DataFrame.
        indices: Optional subset indices.

    Returns:
        Tuple of (processed_data, processed_coordinates) DataFrames.
    """
    result = _mapper.process_raw_data(data, bins_coordinates, depot, indices)
    with contextlib.suppress(Exception):
        run = get_active_run() if get_active_run is not None else None
        if run is not None:
            run.log_params({"data.n_bins_after_index_filter": len(result[0])})

    _log_processor_event("process_data", variable_name="data", n_bins=len(result[0]))
    return result


def process_coordinates(coords, method, col_names=None):
    """Normalize and format coordinates using the specified method.

    Args:
        coords: Raw coordinates (DataFrame or ndarray).
        method: Normalization method (e.g. 'mmn', 'mun', 'wmp').
        col_names: Column names for lat/lng.

    Returns:
        Tuple of (depot, locations).
    """
    if col_names is None:
        col_names = ["Lat", "Lng"]
    result = format_coordinates(coords, method, col_names)
    _log_processor_event("process_coordinates", variable_name="coordinates", method=method, col_names=col_names)
    return result


def process_model_data(
    coordinates,
    dist_matrix,
    device,
    method,
    configs,
    edge_threshold,
    edge_method,
    area,
    waste_type,
    adj_matrix=None,
):
    """Prepare data for neural model inputs.

    Args:
        coordinates: Coordinate data.
        dist_matrix: Distance matrix.
        device: Torch device.
        method: Coordinate normalization method.
        configs: Model configuration dict.
        edge_threshold: Edge filtering threshold.
        edge_method: Edge generation method ('dist' or 'knn').
        area: Geographic area name.
        waste_type: Waste type identifier.
        adj_matrix: Optional pre-computed adjacency matrix.

    Returns:
        Tuple of (model_data_dict, (edges, dm_tensor), profit_vars).
    """
    result = _mapper.process_model_input(
        coordinates,
        dist_matrix,
        device,
        method,
        configs,
        edge_threshold,
        edge_method,
        area,
        waste_type,
        adj_matrix,
    )
    with contextlib.suppress(Exception):
        run = get_active_run() if get_active_run is not None else None
        if run is not None:
            run.log_params(
                {
                    "data.problem_size": int(len(dist_matrix) - 1),
                    "data.coord_method": str(method),
                    "data.has_edge_filter": bool(0 < edge_threshold < 1),
                    "data.edge_threshold": float(edge_threshold),
                    "data.edge_method": str(edge_method),
                }
            )

    _log_processor_event("process_model_data", variable_name="model_input", problem_size=int(len(dist_matrix) - 1))
    return result
