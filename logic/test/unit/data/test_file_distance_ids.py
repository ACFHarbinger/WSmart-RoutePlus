"""ID-labelled distance files are sliced by bin id, not by an off-by-one column drop.

The archived ``osm_distmat.csv`` and the simulator's own saver write a header row of
bin ids and then an N×N numeric body with no id column. ``np.loadtxt`` therefore
sees shape (N+1, N). Dropping the first row *and* the first column builds an
N×(N-1) matrix and later indexes one past the end (``index 350 out of bounds for
size 350`` on the Figueira-350 file).
"""

import numpy as np
import pandas as pd
import pytest
from logic.src.data.network import compute_distance_matrix

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _write_header_matrix(path, ids, body):
    """Write the archived layout: one id header, then a square body with no id column."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(",".join(str(i) for i in ids) + "\n")
        for row in body:
            handle.write(",".join(str(v) for v in row) + "\n")


def test_header_labelled_file_selects_the_requested_bin_ids(tmp_path):
    """A subset is the rows and columns of those bin ids, in the requested order."""
    ids = [0, 10, 20, 30]
    body = np.array(
        [
            [0, 1, 2, 3],
            [4, 0, 5, 6],
            [7, 8, 0, 9],
            [1, 2, 3, 0],
        ],
        dtype=float,
    )
    path = tmp_path / "osm_distmat.csv"
    _write_header_matrix(path, ids, body)
    coords = pd.DataFrame({"ID": [0, 30, 10], "Lat": [0.0, 1.0, 2.0], "Lng": [0.0, 1.0, 2.0]})

    loaded = compute_distance_matrix(coords, "file", dm_filepath=str(path))

    assert loaded.shape == (3, 3)
    assert np.array_equal(
        loaded,
        np.array(
            [
                [0, 3, 1],
                [1, 0, 2],
                [4, 6, 0],
            ],
            dtype=float,
        ),
    )


def test_header_labelled_file_keeps_the_full_square_body(tmp_path):
    """Requesting every labelled id returns N×N, not the N×(N-1) off-by-one slice."""
    ids = [0, 9884, 9958]
    body = np.array([[0.0, 35.7, 46.9], [32.8, 0.0, 18.1], [45.4, 16.2, 0.0]])
    path = tmp_path / "osm_distmat.csv"
    _write_header_matrix(path, ids, body)
    coords = pd.DataFrame({"ID": ids, "Lat": [0.0, 1.0, 2.0], "Lng": [0.0, 1.0, 2.0]})

    loaded = compute_distance_matrix(coords, "file", dm_filepath=str(path))

    assert loaded.shape == (3, 3)
    assert np.allclose(loaded, body)


def test_header_labelled_subset_does_not_index_past_the_dropped_column(tmp_path):
    """The Figueira failure: 351 labelled columns, and the last id must still resolve."""
    ids = list(range(351))
    body = np.zeros((351, 351), dtype=float)
    body[0, 350] = 12.5
    body[350, 0] = 7.25
    path = tmp_path / "osm_distmat.csv"
    _write_header_matrix(path, ids, body)
    coords = pd.DataFrame({"ID": [0, 350], "Lat": [0.0, 1.0], "Lng": [0.0, 1.0]})

    loaded = compute_distance_matrix(coords, "file", dm_filepath=str(path))

    assert loaded.shape == (2, 2)
    assert np.allclose(loaded, np.array([[0.0, 12.5], [7.25, 0.0]]))
