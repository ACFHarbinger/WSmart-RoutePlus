"""Training TensorDicts carry the simulator's vehicle capacity in the unit of the waste they hold."""

import pytest
import torch
from logic.src.constants import MAX_WASTE
from logic.src.data.generators.builders import VRPInstanceBuilder
from logic.src.pipeline.simulations.repository import load_area_and_waste_type_params


def _build(problem, area, waste_type):
    builder = VRPInstanceBuilder(device="cpu")
    builder.set_dataset_size(2).set_problem_size(20).set_waste_type(waste_type).set_distribution("gamma3")
    builder.set_area(area).set_focus_graph(None, 0).set_problem_name(problem).set_num_days(2).set_seed(42)
    return builder, builder.build_td()


@pytest.mark.parametrize("problem", ["cptp", "cvrp"])
@pytest.mark.parametrize(("area", "waste_type"), [("riomaior", "plastic"), ("figueiradafoz", "paper")])
def test_capacity_is_the_simulator_value_in_bin_fractions(problem, area, waste_type):
    _, td = _build(problem, area, waste_type)
    simulator_percent = load_area_and_waste_type_params(area, waste_type)[0]

    assert torch.all(td["max_waste"] == MAX_WASTE)
    # The simulator plans in percent of a full bin (0-100); the model sees bin fractions (0-1).
    assert torch.allclose(td["capacity"], torch.full_like(td["capacity"], simulator_percent / 100.0 * MAX_WASTE))


def test_unknown_area_falls_back_to_the_builder_default():
    builder, td = _build("cptp", "nowhere", "plastic")

    assert torch.allclose(td["capacity"], torch.full_like(td["capacity"], builder.vehicle_cap / 100.0 * MAX_WASTE))
