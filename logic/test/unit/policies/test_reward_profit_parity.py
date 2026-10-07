"""M-cursor-02: training reward vs simulator profit, via production call sites.

``PTP.get_costs``, ``Bins.collect``, and ``BaseRoutingPolicy._load_area_params``
use three unit systems. This test calls those implementations rather than
re-coding their formulas, so a regression in either production path fails here.
Merging them would rescale RL rewards and crosses other lanes' files.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch
from logic.src.constants.tasks import COST_KM, REVENUE_KG
from logic.src.envs.tasks.ptp import PTP
from logic.src.pipeline.simulations.bins import Bins
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy

pytestmark = [pytest.mark.unit, pytest.mark.fast]


class _AreaParamsProbe(BaseRoutingPolicy):
    """Minimal policy so the test can call the production area-param helper."""

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        raise NotImplementedError


def _plastic_bins(tmp_path: Any, n: int) -> Bins:
    return Bins(
        n=n,
        data_dir=str(tmp_path),
        sample_dist="gamma",
        area="riomaior",
        waste_type="plastic",
        n_days=1,
        n_samples=1,
        seed=0,
    )


def _policy_scaled(area: str = "riomaior", waste_type: str = "plastic") -> Tuple[float, float, Dict[str, Any]]:
    _capacity, revenue_scaled, cost_unit, values = _AreaParamsProbe()._load_area_params(area, waste_type, {})
    return revenue_scaled, cost_unit, values


def test_bins_collect_uses_rio_maior_plastic_coefficients(tmp_path: Any) -> None:
    bins = _plastic_bins(tmp_path, n=1)
    assert bins.revenue == pytest.approx(0.65 * 898 / 1000)
    assert bins.density == pytest.approx(19.0)
    assert bins.expenses == pytest.approx(1.0)
    assert bins.volume == pytest.approx(2.5)


def test_simulator_collect_and_policy_scaled_profit_agree(tmp_path: Any) -> None:
    """Four full 47.5 kg bins = 190 kg; with 96.167 km the archived profit is 14.736."""
    n_bins = 4
    km = 96.167
    bins = _plastic_bins(tmp_path, n=n_bins)
    bins.real_c[:] = 100.0
    bins.c[:] = 100.0
    _mass_vec, kg, n_collected, sim_profit = bins.collect(list(range(n_bins + 1)), cost=km)
    assert n_collected == n_bins
    assert kg == pytest.approx(190.0)
    assert sim_profit == pytest.approx(0.5837 * 190.0 - 96.167, abs=1e-3)
    assert sim_profit == pytest.approx(14.736, abs=1e-3)

    revenue_scaled, cost_unit, values = _policy_scaled()
    fill_percent_sum = 100.0 * n_bins
    scaled_profit = fill_percent_sum * revenue_scaled - km * cost_unit
    assert values["R"] == pytest.approx(revenue_scaled)
    assert values["C"] == pytest.approx(cost_unit)
    assert scaled_profit == pytest.approx(sim_profit)


def test_training_get_costs_is_the_legacy_unit_proxy() -> None:
    """PTP.get_costs uses COST_KM=REVENUE_KG=1.0 on fraction waste, not €/kg."""
    assert COST_KM == 1.0
    assert REVENUE_KG == 1.0
    dataset = {
        "depot": torch.tensor([[0.0, 0.0]]),
        "loc": torch.tensor([[[1.0, 0.0]]]),
        "waste": torch.tensor([[1.0]]),
        "cost_km": COST_KM,
        "revenue_kg": REVENUE_KG,
    }
    pi = torch.tensor([[0, 1, 0]])
    neg_profit, cost_dict, _ = PTP.get_costs(dataset, pi, cw_dict=None)
    assert cost_dict["waste"].item() == pytest.approx(1.0)
    assert cost_dict["length"].item() == pytest.approx(2.0)
    assert neg_profit.item() == pytest.approx(1.0)


def test_training_proxy_does_not_equal_bins_collect_on_the_same_numbers(tmp_path: Any) -> None:
    """One full bin, 2 km: collect uses plastic R and 47.5 kg; training does not."""
    km = 2.0
    dataset = {
        "depot": torch.tensor([[0.0, 0.0]]),
        "loc": torch.tensor([[[1.0, 0.0]]]),
        "waste": torch.tensor([[1.0]]),
        "cost_km": COST_KM,
        "revenue_kg": REVENUE_KG,
    }
    training_neg = PTP.get_costs(dataset, torch.tensor([[0, 1, 0]]), cw_dict=None)[0].item()

    bins = _plastic_bins(tmp_path, n=1)
    bins.real_c[:] = 100.0
    bins.c[:] = 100.0
    _mass_vec, kg, _n, sim_profit = bins.collect([0, 1], cost=km)
    assert training_neg == pytest.approx(1.0)
    assert kg == pytest.approx(bins.volume * bins.density)
    assert sim_profit == pytest.approx(kg * bins.revenue - km * bins.expenses)
    assert abs(sim_profit - (-training_neg)) > 1.0


def test_conversion_identity_through_collect_and_policy_helper(tmp_path: Any) -> None:
    """kg from Bins.collect equals fill_frac × volume × density; that recovers policy-scaled profit."""
    fill_percent = 80.0
    km = 10.0
    bins = _plastic_bins(tmp_path, n=1)
    bins.real_c[:] = fill_percent
    bins.c[:] = fill_percent
    _mass_vec, kg, _n, sim_profit = bins.collect([0, 1], cost=km)
    assert kg == pytest.approx((fill_percent / 100.0) * bins.volume * bins.density)

    revenue_scaled, cost_unit, _values = _policy_scaled()
    assert sim_profit == pytest.approx(fill_percent * revenue_scaled - km * cost_unit)
