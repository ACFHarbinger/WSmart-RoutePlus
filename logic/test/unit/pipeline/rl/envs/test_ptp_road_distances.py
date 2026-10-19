"""Real-network instances are costed in road km; on-the-fly capacity follows the simulator."""

import pytest
import torch
from logic.src.pipeline.rl.envs import PTPGenerator
from logic.src.pipeline.rl.envs.cptp import CPTPEnv
from logic.src.pipeline.simulations.repository import load_area_and_waste_type_params
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _matrix_file(tmp_path) -> str:
    # Depot (id 0) and bins 11, 22, 33, stored in an order that differs from the dataset's.
    rows = [
        "-1,0,33,11,22",
        "0,0,30,10,20",
        "33,31,0,5,6",
        "11,11,5,0,7",
        "22,21,6,8,0",
    ]
    path = tmp_path / "dm.csv"
    path.write_text("\n".join(rows))
    return str(path)


def _instances(node_ids: list[list[float]]) -> TensorDict:
    batch = len(node_ids)
    return TensorDict(
        {
            "locs": torch.rand(batch, 3, 2),
            "depot": torch.zeros(batch, 2),
            "waste": torch.full((batch, 3), 0.2),
            "capacity": torch.full((batch,), 5.0),
            "node_ids": torch.tensor(node_ids),
        },
        batch_size=[batch],
    )


def _env(dm_filepath=None) -> CPTPEnv:
    params = {"num_loc": 3}
    if dm_filepath is not None:
        params["dm_filepath"] = dm_filepath
    return CPTPEnv(generator_params=params, check_env_specs=False)


def test_tour_length_uses_road_km_in_dataset_bin_order(tmp_path) -> None:
    env = _env(_matrix_file(tmp_path))
    td = env.reset(_instances([[11.0, 22.0, 33.0], [33.0, 22.0, 11.0]]))

    for action in ([1, 1], [2, 2], [0, 0]):
        td["action"] = torch.tensor(action)
        td = env.step(td)["next"]

    # Instance 0: depot -> 11 -> 22 -> depot = 10 + 7 + 21. Instance 1: depot -> 33 -> 22 -> depot.
    assert td["tour_length"].tolist() == pytest.approx([38.0, 57.0])


def test_instances_without_bin_ids_keep_euclidean_lengths(tmp_path) -> None:
    data = _instances([[11.0, 22.0, 33.0]])
    del data["node_ids"]

    assert "dm" not in _env(_matrix_file(tmp_path)).reset(data).keys()
    assert "dm" not in _env().reset(_instances([[11.0, 22.0, 33.0]])).keys()


def test_unknown_bin_is_reported(tmp_path) -> None:
    with pytest.raises(ValueError, match="lacks the depot or bins"):
        _env(_matrix_file(tmp_path)).reset(_instances([[11.0, 22.0, 99.0]]))


def test_generator_capacity_follows_the_simulator() -> None:
    expected = load_area_and_waste_type_params("riomaior", "plastic")[0] / 100.0

    assert PTPGenerator(num_loc=5, area="riomaior", waste_type="plastic").capacity == pytest.approx(expected)
    assert PTPGenerator(num_loc=5, area="riomaior", waste_type="plastic", capacity=2.0).capacity == 2.0
    assert PTPGenerator(num_loc=5).capacity == 1.0
