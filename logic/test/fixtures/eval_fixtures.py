"""
Fixtures for evaluation pipeline unit tests.
"""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def eval_opts():
    """Standard options for evaluation tests."""
    return {
        "val_size": 2,
        "offset": 0,
        "data_distribution": "test",
        "vertex_method": "dummy",
        "graph_size": 10,
        "focus_graph": False,
        "focus_size": 0,
        "area": "riomaior",
        "edge_threshold": 100,
        "dm_filepath": "path",
        "waste_type": "all",
        "edge_method": "method",
        "distance_method": "method",
        "model": "model_path.pt",
        "multiprocessing": False,
        "strategy": "sample",
        "eval_batch_size": 2,
        "max_calc_batch_size": 10,
        "results_dir": "res",
        "overwrite": True,
        "output_filename": "out.pkl",
        "compress_mask": False,
        "norm_reward": False,
    }


@pytest.fixture
def mock_eval_model(eval_opts):
    """Mocks the model used in evaluation."""
    mock_model = MagicMock()
    mock_model.problem.NAME = "mvptp"
    mock_model.problem.make_dataset.return_value = MagicMock()

    # Common mock behaviors could go here
    return mock_model
