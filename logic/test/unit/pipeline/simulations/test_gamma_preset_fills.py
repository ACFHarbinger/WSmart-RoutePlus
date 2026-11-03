"""Regression test: the daily fills must follow the Gamma preset named by the distribution (gamma3 != gamma1)."""

import numpy as np
import pytest
from logic.src.pipeline.simulations.bins import Bins

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _mean_fill(tmp_path, dist: str) -> float:
    bins = Bins(n=100, data_dir=str(tmp_path), sample_dist=dist, area="riomaior", waste_type="paper",
                n_days=200, n_samples=1, seed=1)
    bins.set_sample_waste(0, horizon=200)
    return float(np.mean(bins.waste_fills))


def test_gamma3_fills_use_preset_three_not_preset_one(tmp_path, mock_bins_params_loader):
    """The simulator once cut the trailing digit, so every gammaN run drew the gamma1 fills (mean 25.4 %)."""
    gamma1 = _mean_fill(tmp_path, "gamma1")
    gamma3 = _mean_fill(tmp_path, "gamma3")
    assert gamma1 == pytest.approx(25.4, abs=1.5)
    assert gamma3 == pytest.approx(13.8, abs=1.5)


def test_initializing_state_passes_the_full_gamma_name_to_bins(monkeypatch):
    """_initialize_bins must hand "gamma3" (not "gamma") to Bins, whose generator reads the preset from it."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from logic.src.pipeline.simulations.states import initializing

    captured = {}

    def fake_bins(n, data_dir, sample_dist, **kwargs):
        captured["dist"] = sample_dist
        return MagicMock()

    monkeypatch.setattr(initializing, "Bins", fake_bins)
    graph = SimpleNamespace(num_loc=10, area="riomaior", waste_type="paper", n_days=30, n_samples=1)
    monkeypatch.setattr(initializing, "get_graph_config", lambda sim: graph, raising=False)
    monkeypatch.setattr("logic.src.utils.infrastructure.setup_sims.get_graph_config", lambda sim: graph)
    sim = SimpleNamespace(data_distribution="gamma3", noise_mean=0.0, noise_variance=0.0, seed=1, stats_filepath=None)
    ctx = SimpleNamespace(cfg=SimpleNamespace(sim=sim), data_dir="x", sample_id=0, bins=None)
    initializing.InitializingState()._initialize_bins(ctx)
    assert captured["dist"] == "gamma3"
