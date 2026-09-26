"""Simulation setup functions for base data and distance computation.

Attributes:
    setup_basedata: Load depot, bin statistics and coordinates for a given area.
    setup_dist_path_tup: Compute distance matrix, edge structure and shortest paths.

Example:
    data, bins_coordinates, depot = setup_basedata(n_bins, data_dir, area, waste_type)
    (dist_matrix_edges, paths, dm_tensor, distC), adj_matrix = setup_dist_path_tup(bins_coordinates, size, dist_method, dm_filepath, env_filename, gapik_file, symkey_name, edge_thresh, edge_method, focus_idx)
"""

import contextlib
import os
import tempfile

import numpy as np
import pandas as pd
import torch

from logic.src.constants import ROOT_DIR
from logic.src.data.network import compute_distance_matrix

try:
    from logic.src.tracking.core.run import get_active_run
except ImportError:
    get_active_run = None  # type: ignore[assignment]
from logic.src.utils.graph.network_utils import apply_edges, get_paths_between_states

from ._logging import _log_processor_event


# simulations.repository imports this package, so a module-level import of it is
# circular in spawned simulation workers (they hung at start-up). These wrappers
# import lazily and stay patchable module attributes.
def load_depot(*args, **kwargs):
    """Lazy proxy for logic.src.pipeline.simulations.repository.load_depot."""
    from logic.src.pipeline.simulations.repository import load_depot as _load_depot

    return _load_depot(*args, **kwargs)


def load_simulator_data(*args, **kwargs):
    """Lazy proxy for logic.src.pipeline.simulations.repository.load_simulator_data."""
    from logic.src.pipeline.simulations.repository import load_simulator_data as _load_simulator_data

    return _load_simulator_data(*args, **kwargs)


def _save_distance_matrix_atomic(save_path, node_ids, distance_matrix):
    """Atomically save one complete matrix, even when simulation workers race.

    Every policy worker in a simulation run shares ``save_path``.  Writing the
    header and rows in separate append operations allowed those workers to
    concatenate or interleave matrices.  A temporary file in the destination
    directory followed by ``os.replace`` makes each worker publish a complete
    snapshot; concurrent workers may replace one another, but cannot combine
    their output.
    """
    matrix = np.asarray(distance_matrix)
    ids = np.asarray(node_ids)
    expected_shape = (len(ids), len(ids))
    if matrix.shape != expected_shape:
        raise ValueError(
            f"Refusing to save distance matrix with shape {matrix.shape}; "
            f"expected {expected_shape} for {len(ids)} node IDs"
        )

    destination = os.path.abspath(os.fspath(save_path))
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            dir=os.path.dirname(destination),
            prefix=f".{os.path.basename(destination)}.",
            suffix=".tmp",
            delete=False,
        ) as matrix_f:
            temp_path = matrix_f.name
            matrix_f.write(",".join(map(str, ids)) + "\n")
            pd.DataFrame(matrix).to_csv(matrix_f, index=False, header=False)
            matrix_f.flush()
            os.fsync(matrix_f.fileno())
        os.replace(temp_path, destination)
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)


def setup_basedata(n_bins, data_dir, area, waste_type):
    """Load depot, bin statistics and coordinates for a given area.

    Args:
        n_bins: Number of bins.
        data_dir: Base data directory.
        area: Geographic area name.
        waste_type: Waste type identifier.

    Returns:
        Tuple of (data, bins_coordinates, depot) DataFrames.
    """
    depot = load_depot(data_dir, area)
    data, bins_coordinates = load_simulator_data(data_dir, n_bins, area, waste_type)
    _log_processor_event("setup_basedata", variable_name="base_data", n_bins=n_bins, area=area, waste_type=waste_type)
    return data, bins_coordinates, depot


def setup_dist_path_tup(
    bins_coordinates,
    size,
    dist_method,
    dm_filepath,
    env_filename,
    gapik_file,
    symkey_name,
    edge_thresh,
    edge_method,
    area,
    waste_type,
    n_days,
    distribution=None,
    focus_idx=None,
    save_updated_dm=None,
):
    """Compute distance matrix, edge structure and shortest paths.

    Args:
        bins_coordinates: Coordinates DataFrame.
        size: Number of customer nodes (excludes depot).
        dist_method: Distance computation method.
        dm_filepath: Optional pre-computed distance matrix file.
        env_filename: Environment config file.
        gapik_file: Google API key file.
        symkey_name: Symmetric key name.
        edge_thresh: Edge filtering threshold.
        edge_method: Edge generation method.
        area: Geographic area name (used to construct the save path).
        n_days: Number of simulation days (used to construct the save path).
        focus_idx: Optional focus indices.
        save_updated_dm: Optional filename to save the updated distance matrix under
            ``assets/output/{n_days}days/{area}{size}_{area}/{distribution}/{run_name}/``.

    Returns:
        Tuple of ((dist_matrix_edges, paths, dm_tensor, distC), adj_matrix).
    """
    dist_matrix = compute_distance_matrix(
        bins_coordinates,
        dist_method,
        dm_filepath=dm_filepath,
        env_filename=env_filename,
        gapik_file=gapik_file,
        symkey_name=symkey_name,
        focus_idx=focus_idx,
    )
    dist_matrix_edges, shortest_paths, adj_matrix = apply_edges(dist_matrix, edge_thresh, edge_method)
    if save_updated_dm is not None and str(save_updated_dm).strip():
        save_path = os.path.join(
            ROOT_DIR,
            "assets",
            "output",
            f"{n_days}days",
            f"{area}{size}_{waste_type}",
            f"{distribution}",
            save_updated_dm,
        )
        _save_distance_matrix_atomic(
            save_path,
            bins_coordinates["ID"].to_numpy(),
            dist_matrix_edges,
        )
        print(f"[INFO] Saved {area} graph distance matrix ({size + 1} nodes) to: {save_path}")
    paths = get_paths_between_states(size + 1, shortest_paths)
    dm_tensor = torch.from_numpy(dist_matrix_edges / 100.0)
    distC = np.round(dist_matrix_edges * 10).astype("int32")

    with contextlib.suppress(Exception):
        run = get_active_run() if get_active_run is not None else None
        if run is not None:
            n_nonzero = np.count_nonzero(dist_matrix_edges)
            n_total = dist_matrix_edges.size
            run.log_params(
                {
                    "data.edge_threshold": float(edge_thresh),
                    "data.edge_method": str(edge_method),
                    "data.n_nodes": int(size + 1),
                }
            )
            run.log_metric("data/n_edges", float(n_nonzero))
            run.log_metric("data/edge_density", float(n_nonzero) / float(n_total) if n_total > 0 else 0.0)

    _log_processor_event(
        "setup_dist_path_tup",
        variable_name="distance_matrix",
        size=size,
        dist_method=dist_method,
        edge_thresh=edge_thresh,
    )
    return (dist_matrix_edges, paths, dm_tensor, distC), adj_matrix
