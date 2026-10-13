"""Training TensorDicts keep vehicle capacity in the same unit as the waste they carry."""

import pytest
import torch

from logic.src.constants import MAX_WASTE
from logic.src.data.generators.builders import VRPInstanceBuilder


@pytest.mark.parametrize("problem", ["cptp", "cvrp"])
def test_capacity_and_waste_share_units(problem):
    builder = VRPInstanceBuilder(device="cpu")
    builder.set_dataset_size(2).set_problem_size(20).set_waste_type("plastic").set_distribution("gamma3")
    builder.set_area("riomaior").set_focus_graph(None, 0).set_problem_name(problem).set_num_days(2).set_seed(42)
    td = builder.build_td()

    assert torch.all(td["max_waste"] == MAX_WASTE)
    assert torch.allclose(td["capacity"], torch.full_like(td["capacity"], builder.vehicle_cap / 100.0 * MAX_WASTE))
    # The 100-percent default is one full bin; a fraction-scaled capacity must be able to bind.
    assert float(td["capacity"][0]) < float(td["waste"].sum(-1).mean())
