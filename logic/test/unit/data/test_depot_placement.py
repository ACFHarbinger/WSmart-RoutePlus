"""Central depot placement: centroid-nearest and medoid bins and depot relocation."""

import numpy as np
import pandas as pd
import pytest
from logic.src.data.processor.depot import (
    bins_to_load,
    extend_subset,
    promote_central_depot,
    relocate_depot,
    select_depot_bin,
    synthetic_depot_distances,
)


def test_centroid_picks_the_bin_nearest_to_the_mean():
    coords = np.array([[38.00, -9.00], [38.00, -9.04], [38.00, -9.02], [38.00, -9.40]])
    # the mean longitude is -9.115, so the bin at -9.04 is nearest
    assert select_depot_bin("centroid", coords=coords) == 1


def test_medoid_picks_the_bin_with_the_least_total_distance():
    matrix = np.array([[0.0, 1.0, 9.0], [1.0, 0.0, 1.0], [9.0, 1.0, 0.0]])
    assert select_depot_bin("medoid", distance_matrix=matrix) == 1


@pytest.mark.parametrize("mode", ["remote", "synthetic"])
def test_other_modes_do_not_select_a_bin(mode):
    with pytest.raises(ValueError, match="does not select a bin"):
        select_depot_bin(mode, coords=np.zeros((2, 2)))


def test_missing_inputs_are_rejected():
    with pytest.raises(ValueError, match="coordinates"):
        select_depot_bin("centroid")
    with pytest.raises(ValueError, match="distance matrix"):
        select_depot_bin("medoid")


def test_relocate_depot_reorders_and_drops_the_old_depot():
    matrix = np.arange(16, dtype=float).reshape(4, 4)  # node 0 is the depot, nodes 1..3 are bins
    moved, nodes = relocate_depot(matrix, 2)
    assert nodes.tolist() == [2, 1, 3]
    assert moved.shape == (3, 3)
    assert moved[0, 1] == matrix[2, 1]
    assert moved[1, 0] == matrix[1, 2]
    with pytest.raises(ValueError, match="bin_node"):
        relocate_depot(matrix, 0)


def _frames():
    coords = pd.DataFrame(
        {
            "#bin": [0, 1, 2, 3, 4],
            "ID": [0, 10, 11, 12, 13],
            "Lat": [39.0, 38.0, 38.0, 38.0, 38.0],
            "Lng": [-8.0, -9.00, -9.04, -9.02, -9.40],
        }
    )
    data = pd.DataFrame(
        {
            "#bin": [0, 1, 2, 3, 4],
            "ID": [0, 10, 11, 12, 13],
            "Stock": [0.0, 5, 6, 7, 8],
            "Accum_Rate": [0.0, 1, 2, 3, 4],
        }
    )
    return data, coords


def test_bins_to_load_adds_the_promoted_bin():
    assert bins_to_load(100, "remote") == 100
    assert bins_to_load(100, "centroid") == 101
    assert bins_to_load(100, "synthetic") == 100


def test_remote_mode_leaves_the_frames_unchanged():
    data, coords = _frames()
    out_data, out_coords = promote_central_depot(data, coords, "remote")
    assert out_data is data and out_coords is coords


def test_centroid_mode_moves_the_nearest_bin_to_row_zero():
    data, coords = _frames()
    out_data, out_coords = promote_central_depot(data, coords, "centroid")
    assert out_coords["ID"].tolist() == [11, 10, 12, 13]
    assert out_data["ID"].tolist() == [11, 10, 12, 13]
    assert out_coords["#bin"].tolist() == [0, 1, 2, 3]
    assert out_data.loc[0, ["Stock", "Accum_Rate"]].tolist() == [0.0, 0.0]
    assert out_data["Stock"].tolist() == [0.0, 5, 7, 8]


def test_medoid_is_not_available_at_initialization():
    data, coords = _frames()
    with pytest.raises(NotImplementedError):
        promote_central_depot(data, coords, "medoid")
    with pytest.raises(ValueError, match="depot mode"):
        promote_central_depot(data, coords, "middle")


def test_extend_subset_adds_the_lowest_unused_position():
    assert extend_subset([0, 1, 3, 4], 6) == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="spare"):
        extend_subset([0, 1], 2)


def test_candidate_ids_restrict_the_centroid_and_the_promoted_bin():
    data, coords = _frames()
    # without the far bin 13 the centroid moves to bin 10's side; the promoted bin must be a candidate
    out_data, out_coords = promote_central_depot(data, coords, "centroid", candidate_ids=[10, 11, 12])
    assert out_coords["ID"].iloc[0] in (10, 11, 12)
    assert sorted(out_coords["ID"].tolist()) == [10, 11, 12, 13]


def test_synthetic_mode_keeps_every_bin_and_moves_the_depot_to_the_centroid():
    data, coords = _frames()
    out_data, out_coords = promote_central_depot(data, coords, "synthetic")
    assert out_data is data
    assert out_coords["ID"].tolist() == coords["ID"].tolist()
    lat, lng = out_coords.loc[out_coords.index[0], ["Lat", "Lng"]]
    expected = np.mean(coords.iloc[1:][["Lat", "Lng"]].to_numpy(), axis=0)
    assert np.allclose([lat, lng], expected, atol=1e-3)
    assert coords.loc[0, "Lat"] != lat  # the input frame is not modified


def test_synthetic_depot_distances_reach_the_network_through_the_nearest_bin():
    bins = np.array([[38.00, -9.00], [38.00, -9.04], [38.00, -9.20]])  # centroid near the bin at -9.04
    rng = np.random.default_rng(0)
    matrix = rng.uniform(1, 10, size=(4, 4))
    np.fill_diagonal(matrix, 0.0)
    out = synthetic_depot_distances(matrix, bins)
    access = out[0, 2]
    assert out[2, 0] == access and 0 < access < 5
    assert np.allclose(out[1:, 1:], matrix[1:, 1:])  # bin-to-bin distances are unchanged
    assert np.allclose(out[0, [1, 3]], access + matrix[2, [1, 3]])
    assert np.allclose(out[[1, 3], 0], matrix[[1, 3], 2] + access)
    assert out[0, 0] == 0.0
    assert not np.shares_memory(out, matrix)
