"""Campaign plan: independent scenarios split between the depot placements."""

import pytest
from logic.src.pipeline.simulations.campaign import benchmark_campaign

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_default_campaign_has_five_remote_and_five_central_scenarios():
    remote, central = benchmark_campaign()
    assert (remote.depot_mode, remote.n_samples) == ("remote", 5)
    assert (central.depot_mode, central.n_samples) == ("centroid", 5)


def test_scenarios_use_disjoint_seeds():
    remote, central = benchmark_campaign(base_seed=42)
    assert remote.sample_seeds() == [42, 43, 44, 45, 46]
    assert central.sample_seeds() == [47, 48, 49, 50, 51]
    assert not set(remote.sample_seeds()) & set(central.sample_seeds())


def test_overrides_select_the_scenario():
    _, central = benchmark_campaign()
    assert central.overrides() == ["sim.graph.depot_mode=centroid", "sim.seed=47", "sim.graph.n_samples=5"]


def test_empty_campaign_is_rejected():
    with pytest.raises(ValueError, match="n_per_depot"):
        benchmark_campaign(n_per_depot=0)
