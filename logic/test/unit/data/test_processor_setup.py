"""Regression tests for simulation distance-matrix persistence."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from logic.src.data.processor.setup import _save_distance_matrix_atomic


def test_save_distance_matrix_atomic_rejects_wrong_shape(tmp_path):
    output = tmp_path / "distance.csv"

    with pytest.raises(ValueError, match=r"expected \(3, 3\)"):
        _save_distance_matrix_atomic(output, [10, 20, 30], np.zeros((2, 3)))

    assert not output.exists()


def test_save_distance_matrix_atomic_never_combines_concurrent_writers(tmp_path):
    output = tmp_path / "distance.csv"
    node_ids = np.arange(40)
    matrices = [np.full((40, 40), value, dtype=float) for value in range(8)]

    with ThreadPoolExecutor(max_workers=len(matrices)) as executor:
        list(executor.map(lambda matrix: _save_distance_matrix_atomic(output, node_ids, matrix), matrices))

    saved = np.loadtxt(output, delimiter=",")
    assert saved.shape == (41, 40)
    assert np.array_equal(saved[0], node_ids)
    assert any(np.array_equal(saved[1:], matrix) for matrix in matrices)
    assert not list(tmp_path.glob(".*.tmp"))
