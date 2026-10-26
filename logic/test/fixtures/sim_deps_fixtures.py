"""
Fixtures for the Simulator pipeline - Dependency fixtures.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_sim_dependencies(mocker, tmp_path, mock_bins_instance):
    """
    Mocks all major external dependencies for single_simulation
    and sequential_simulations.
    """
    # 1. Patch ROOT_DIR in both modules to ensure consistency
    mocker.patch("logic.src.pipeline.simulations.states.base.context.ROOT_DIR", str(tmp_path))
    mocker.patch("logic.src.pipeline.simulations.manager.ROOT_DIR", str(tmp_path))
    mocker.patch("logic.src.pipeline.simulations.progress.ROOT_DIR", str(tmp_path))

    # 2. Mock loader functions
    mock_depot = pd.DataFrame({"ID": [0], "Lat": [40], "Lng": [-8], "Stock": [0], "Accum_Rate": [0]})
    mock_data = pd.DataFrame({"ID": [1, 2], "Stock": [10, 20], "Accum_Rate": [0.1, 0.2]})
    mock_coords = pd.DataFrame({"ID": [1, 2], "Lat": [40.1, 40.2], "Lng": [-8.1, -8.2]})
    mocker.patch("logic.src.data.processor.setup.load_depot", return_value=mock_depot)
    mocker.patch(
        "logic.src.data.processor.setup.load_simulator_data",
        return_value=(mock_data.copy(), mock_coords.copy()),
    )

    # Mock setup_basedata in states to return the tuple directly
    mock_setup_basedata = mocker.patch(
        "logic.src.pipeline.simulations.states.initializing.setup_basedata",
        return_value=(mock_data.copy(), mock_coords.copy(), mock_depot.copy()),
    )

    # 3. Mock processor functions (patched in states where imported)
    mock_proc_data = pd.DataFrame({"ID": [0, 1, 2], "Stock": [0, 10, 20]})
    mock_proc_coords = pd.DataFrame({"ID": [0, 1, 2], "Lat": [40, 40.1, 40.2], "Lng": [-8, -8.1, -8.2]})
    mock_process_data = mocker.patch(
        "logic.src.pipeline.simulations.states.initializing.process_data",
        return_value=(mock_proc_data.copy(), mock_proc_coords.copy()),
    )
    mocker.patch(
        "logic.src.pipeline.simulations.states.initializing.process_model_data",
        return_value=("mock_model_tup_0", "mock_model_tup_1"),
    )

    # 4. Mock network functions (patched in states where imported)
    mock_dist_tup = (
        np.array([[0, 1], [1, 0]]),
        "mock_paths",
        "mock_tensor",
        "mock_distC",
    )
    mock_adj_matrix = np.array([[1, 1], [1, 1]])
    mock_setup_dist = mocker.patch(
        "logic.src.pipeline.simulations.states.initializing.setup_dist_path_tup",
        return_value=(mock_dist_tup, mock_adj_matrix),
    )
    mocker.patch(
        "logic.src.data.processor.setup.compute_distance_matrix",
        return_value=np.array([[0, 1], [1, 0]]),
    )
    mocker.patch(
        "logic.src.data.processor.setup.apply_edges",
        return_value=("mock_dist_edges", "mock_paths", "mock_adj"),
    )
    mocker.patch(
        "logic.src.data.processor.setup.get_paths_between_states",
        return_value="mock_all_paths",
    )

    # 5. Mock Bins class
    mocker.patch("logic.src.pipeline.simulations.bins.Bins", return_value=mock_bins_instance)

    # 6. Mock setup functions
    mock_setup_model = mocker.patch(
        "logic.src.pipeline.simulations.states.initializing.setup_model",
        return_value=(MagicMock(), MagicMock()),
    )
    mock_setup_env = mocker.patch(
        "logic.src.pipeline.simulations.states.initializing.setup_env", return_value="mock_or_env"
    )

    # 7. Mock day function
    mock_dlog = {
        "day": 1,
        "overflows": 0,
        "kg_lost": 0,
        "kg": 0,
        "ncol": 0,
        "km": 0,
        "kg/km": 0,
        "tour": [0],
    }

    mock_return_ctx = mocker.MagicMock()
    mock_return_ctx.new_data = mock_proc_data
    mock_return_ctx.coords = mock_proc_coords
    mock_return_ctx.bins = mock_bins_instance
    mock_return_ctx.overflows = 0
    mock_return_ctx.daily_log = mock_dlog
    mock_return_ctx.cached = None

    mock_run_day = mocker.patch(  # CAPTURE the mock object here
        "logic.src.pipeline.simulations.states.running.run_day", return_value=mock_return_ctx
    )

    # 8. Mock checkpointing
    mock_cp_instance = mocker.MagicMock()
    mock_cp_instance.load_state.return_value = (None, 0)  # Default: no resume
    mocker.patch(
        "logic.src.pipeline.simulations.states.initializing.SimulationCheckpoint",
        return_value=mock_cp_instance,
    )

    # Mock the context manager and its hook
    mock_hook = mocker.MagicMock()
    mock_cm = mocker.MagicMock()
    mock_cm.__enter__.return_value = mock_hook  # Yield the hook
    mocker.patch("logic.src.pipeline.simulations.states.running.checkpoint_manager", return_value=mock_cm)

    # 9. Mock utilities
    mock_update_policy_log_section = mocker.MagicMock()
    mocker.patch(
        "logic.src.pipeline.simulations.states.finishing.update_policy_log_section", mock_update_policy_log_section
    )
    mocker.patch(
        "logic.src.pipeline.simulations.progress.update_policy_log_section", mock_update_policy_log_section
    )
    mock_save_excel = mocker.patch("logic.src.pipeline.simulations.states.finishing.save_matrix_to_excel")
    mocker.patch("time.perf_counter", return_value=1.0)
    mocker.patch("pandas.DataFrame.to_excel")
    mocker.patch("statistics.mean", return_value=1.0)
    mocker.patch("statistics.stdev", return_value=0.1)

    # Return key mocks for modification in tests
    return {
        "checkpoint": mock_cp_instance,
        "hook": mock_hook,
        "run_day": mock_run_day,
        "log_to_json": mock_update_policy_log_section,
        "save_excel": mock_save_excel,
        "bins": mock_bins_instance,
        "setup_model": mock_setup_model,
        "setup_env": mock_setup_env,
        "process_data": mock_process_data,
        "_setup_basedata": mock_setup_basedata,
        "_setup_dist_path_tup": mock_setup_dist,
    }


@pytest.fixture
def mock_load_dependencies(mocker):
    """
    Mocks all external dependencies:
    - os.path.join
    - pandas.read_csv
    - pandas.read_excel
    - the 'udef' module
    """
    # Mock os.path.join to simply concatenate with '/'
    mocker.patch("os.path.join", side_effect=lambda *args: "/".join(args))

    # Mock the file readers
    mock_read_csv = mocker.patch("pandas.read_csv")
    mock_read_excel = mocker.patch("pandas.read_excel")

    # Mock the 'udef' module
    mock_udef = MagicMock()
    mock_udef.WASTE_TYPES = {
        "paper": "Embalagens de papel e cartão",
        "plastic": "Mistura de embalagens",
    }
    mock_udef.LOCK_TIMEOUT = 30
    mocker.patch("logic.src.constants", mock_udef)

    return mock_read_csv, mock_read_excel, mock_udef
