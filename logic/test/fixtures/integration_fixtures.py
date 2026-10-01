"""
Fixtures specifically for Integration Tests.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def train_opts(tmp_path):
    """Base options for training integration tests, derived from Config defaults."""
    # We populate a flat dict to match legacy expectations in integration tests
    opts = {
        "problem": "vrpp",
        "graph_size": 10,
        "batch_size": 2,
        "epoch_size": 10,
        "val_size": 10,
        "train.env.graph.n_days": 1,
        "env.graph.start_day": 0,
        "no_cuda": True,
        "device": "cpu",
        "no_tensorboard": True,
        "wandb_mode": "disabled",
        "log_dir": "logs",
        "save_dir": str(tmp_path / "checkpoints"),
        "final_dir": str(tmp_path / "final"),
        "embed_dim": 16,
        "hidden_dim": 16,
        "n_encode_layers": 1,
        "optimizer": "adam",
        "lr_model": 1e-4,
        "eval_batch_size": 2,
        "log_step": 5,
        "run_name": "test_run",
        "seed": 1234,
        "data_distribution": "const",
        "area": "riomaior",
        "waste_type": "paper",
        "w_waste": 1.0,
        "w_length": 1.0,
        "w_overflows": 100.0,
        "w_lost": 10.0,
        "w_penalty": 0.0,
        "mrl_method": "cb",
        "hpo_method": "gs",
    }
    return opts


@pytest.fixture
def sim_opts(tmp_path, setup_sim_data):
    """Base options for simulation integration tests."""
    return {
        "problem": "vrpp",
        "size": 10,  # Small graph
        "days": 2,
        "n_samples": 1,
        "area": "riomaior",
        "policies": ["regular_emp"],  # Format: policy_distribution (must be emp or gamma)
        "output_dir": str(tmp_path / "results"),
        "resume": False,
        "parallel": False,
        "cpu_cores": 1,
        "device": "cpu",
        # Policy specific args
        "threshold": 0.5,
        "look_ahead_days": 1,
        "alns_iterations": 10,
        "bcp_time_limit": 1,
        "hgs_time_limit": 1,
        # Neural agent specific
        "model": "am",
        "model_path": None,
        "decode_strategy": "greedy",
        # Gurobi/Solver
        "solver": "gurobi",
        "time_limit": 1,
        "waste_type": "paper",  # Required key
        "server_run": False,  # avoid server logic
        "gplic_file": None,
        "symkey_name": None,
        "env_file": None,
        "gapik_file": None,
        "distance_method": "hsd",  # euclidean not supported
        "dm_filepath": None,
        "edge_threshold": 0,
        "edge_method": "knn",
        "data_distribution": "emp",  # const not supported by Bins, use emp
        "seed": 42,
        # args expected by initial state/setup
        "stats_filepath": None,
        "cache_regular": False,
        "n_vehicles": 1,
        "checkpoint_dir": "checkpoints",
        "checkpoint_days": 1,
        "temperature": 1.0,
        "strategy": "greedy",
        "vertex_method": "mmn",
        # Explicit data_dir provided to override default hardcoded lookup if logic allows,
        # but logic often uses ROOT_DIR. setup_sim_data mocks ROOT_DIR.
        "data_dir": str(setup_sim_data / "data" / "simulator"),
    }


@pytest.fixture
def setup_sim_data(tmp_path, mocker):
    """
    Sets up a simulation data environment in tmp_path and patches ROOT_DIR.
    Creates necessary dummy CSV/Excel files for loader.
    """
    # Patch ROOT_DIR in key places
    mocker.patch("logic.src.pipeline.simulations.states.ROOT_DIR", str(tmp_path))
    mocker.patch("logic.src.pipeline.simulations.simulator.ROOT_DIR", str(tmp_path))
    mocker.patch("logic.src.pipeline.simulations.checkpoints.ROOT_DIR", str(tmp_path))
    mocker.patch("logic.src.constants.ROOT_DIR", str(tmp_path))
    mocker.patch("logic.src.pipeline.simulations.states.base.context.ROOT_DIR", str(tmp_path))
    mocker.patch("logic.src.pipeline.simulations.states.initializing.ROOT_DIR", str(tmp_path))
    mocker.patch("logic.src.pipeline.simulations.hpo.hpo_handler.ROOT_DIR", str(tmp_path))
    mocker.patch("logic.src.pipeline.simulations.actions.route_improvement.ROOT_DIR", str(tmp_path))
    mocker.patch("logic.src.pipeline.simulations.actions.node_selection.ROOT_DIR", str(tmp_path))
    mocker.patch("logic.src.pipeline.simulations.bins.base.ROOT_DIR", str(tmp_path))
    mocker.patch("logic.src.pipeline.simulations.repository.ROOT_DIR", str(tmp_path))
    # Mock fast_tsp to avoid solver crashes with dummy data
    mocker.patch(
        "logic.src.policies.route_construction.other_algorithms.travelling_salesman_problem.tsp.fast_tsp.find_tour",
        return_value=[0, 1, 0],
    )
    # Instantiate a real FileSystemRepository and patch the global singleton
    from logic.src.pipeline.simulations.repository import FileSystemRepository

    repo = FileSystemRepository(data_root_dir=str(tmp_path))
    # Patch the _REPOSITORY in the module
    mocker.patch("logic.src.pipeline.simulations.repository._REPOSITORY", repo)

    data_dir = tmp_path / "data" / "simulator"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create checkpoints directory mock path points to
    (tmp_path / "checkpoints").mkdir(exist_ok=True)

    coord_dir = data_dir / "coordinates"
    coord_dir.mkdir(parents=True, exist_ok=True)

    waste_dir = data_dir / "bins_waste"
    waste_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy Facilities.csv
    pd.DataFrame({"Sigla": ["CTEASO"], "Lat": [40.0], "Lng": [-8.0], "ID": [1000]}).to_csv(
        coord_dir / "Facilities.csv", index=False
    )

    # Create dummy node data
    # Filename format: old_out_info[riomaior].csv for coordinates
    # daily_c[riomaior_paper].csv for waste data
    # (Assuming loader uses these names - checking loader.py/FileSystemRepository)

    # 10 bins + 1 depot = 11 IDs
    ids = list(range(11))
    # Simulator usually treats 0 as depot.
    # Coordinates file - moved details below

    # Waste data
    # Loader expects bins_waste/old_out_crude_rate[riomaior].csv for Rio Maior
    # Variance is needed to avoid NaN during normalization (min=max)
    # Loader expects bins_waste/old_out_crude_rate[riomaior].csv for Rio Maior
    # Variance is needed to avoid NaN during normalization (min=max)
    waste_cols: dict = {str(i): [1.0 + float(i) * 0.1 + float(j) * 0.01 for j in range(365)] for i in ids if i != 0}
    # Must include 0 (Depot) with 0 waste
    waste_cols["0"] = [0.0] * 365
    waste_cols["Date"] = pd.date_range("2023-01-01", periods=365).tolist()

    # Create the file with the name expected by loader.py for 'riomaior' area
    df_waste = pd.DataFrame(waste_cols)
    # Create ALL possible aliases to be safe
    df_waste.to_csv(waste_dir / "old_out_crude_rate[riomaior].csv", index=False)
    df_waste.to_csv(waste_dir / "out_rate_crude[riomaior].csv", index=False)

    # Also fix coordinates file to include 'Tipo de Residuos' column
    df_coords = pd.DataFrame(
        {
            "ID": ids,
            "Lat": [40.0 + i * 0.01 for i in ids],
            "Lng": [-8.0 + i * 0.01 for i in ids],
            "Tipo de Residuos": ["Embalagens de papel e cartão"] * len(ids),
        }
    )
    # Create ALL possible aliases
    df_coords.to_csv(coord_dir / "out_info[riomaior].csv", index=False)
    df_coords.to_csv(coord_dir / "old_out_info[riomaior].csv", index=False)

    # Loader (small size) needs intersection.csv
    df_intersection = df_coords.copy()
    df_intersection["ID225"] = df_intersection["ID"]
    df_intersection["ID317"] = df_intersection["ID"]
    df_intersection.to_csv(coord_dir / "intersection.csv", index=False)

    return tmp_path


@pytest.fixture
def base_vrpp_data():
    """Generates a small valid VRPP instance for testing solvers."""
    bins = np.array([50.0, 95.0, 30.0, 85.0, 50.0])  # fill levels
    # 6x6 distance matrix (depot + 5 bins)
    dist_matrix = [
        [0, 10, 10, 10, 10, 10],
        [10, 0, 5, 10, 15, 20],
        [10, 5, 0, 5, 10, 15],
        [10, 10, 5, 0, 5, 10],
        [10, 15, 10, 5, 0, 5],
        [10, 20, 15, 10, 5, 0],
    ]
    values = {
        "Q": 400.0,
        "R": 1.0,
        "B": 1.0,
        "C": 0.1,
        "V": 1.0,
        "Omega": 0.1,
        "delta": 0.0,
        "psi": 0.9,  # Threshold to collect
    }
    binsids = [0, 1, 2, 3, 4, 5]
    mandatory_nodes = [2, 4]  # Bin IDs (index in dist_matrix)

    return {
        "bins": bins,
        "dist_matrix": dist_matrix,
        "values": values,
        "binsids": binsids,
        "mandatory_nodes": mandatory_nodes,
    }


@pytest.fixture
def parity_instance():
    """
    Small VRPP instance where all solvers should ideally match or be very close.
    5 Bins, 1 Vehicle, Capacity 100.
    """
    bins = np.array([10.0, 20.0, 30.0, 40.0, 50.0])  # Wastes/Fill
    dist_matrix = [
        [0, 10, 20, 30, 40, 50],
        [10, 0, 10, 20, 30, 40],
        [20, 10, 0, 10, 20, 30],
        [30, 20, 10, 0, 10, 20],
        [40, 30, 20, 10, 0, 10],
        [50, 40, 30, 20, 10, 0],
    ]
    values = {
        "Q": 1000.0,  # Large capacity to allow visiting all
        "R": 1.0,  # Revenue per unit
        "B": 1.0,
        "C": 1.0,  # Cost per unit distance
        "V": 1.0,
        "Omega": 0.0,
        "delta": 0.0,
        "psi": 0.01,  # Collect almost anything
    }
    binsids = [0, 1, 2, 3, 4, 5]
    mandatory = []  # No forced nodes, let profit drive it

    return {"bins": bins, "dist_matrix": dist_matrix, "values": values, "binsids": binsids, "mandatory": mandatory}
