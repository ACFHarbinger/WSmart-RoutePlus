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
    assert central.overrides() == [
        "sim.graph.depot_mode=centroid",
        "sim.seed=47",
        "sim.graph.n_samples=5",
        "sim.graph.load_dataset=null",
    ]


def test_empty_campaign_is_rejected():
    with pytest.raises(ValueError, match="n_per_depot"):
        benchmark_campaign(n_per_depot=0)


def test_network_overrides_quote_the_matrix_file():
    from logic.src.pipeline.simulations.campaign import NETWORKS

    overrides = NETWORKS["rm100"].overrides()
    assert "sim.graph.num_loc=100" in overrides
    assert 'sim.graph.dm_filepath="gmaps_distmat_plastic[riomaior].csv"' in overrides
    assert "sim.graph.load_dataset=null" in benchmark_campaign()[0].overrides()
