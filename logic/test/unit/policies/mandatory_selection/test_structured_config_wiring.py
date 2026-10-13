"""Structured mandatory-selection configs reach the strategy objects they configure."""

import logging

import numpy as np
import pytest
import yaml
from logic.src.configs.policies.other.mandatory_selection import (
    BernoulliRandomSelectionConfig,
    CombinedSelectionConfig,
    FilterAndFanSelectionConfig,
    FPTASKnapsackSelectionConfig,
    FractionalKnapsackSelectionConfig,
    MandatorySelectionConfig,
    MIPKnapsackSelectionConfig,
    PortfolioDispatcherSelectionConfig,
    RevenueSelectionConfig,
)
from logic.src.policies.mandatory_selection.base.selection_factory import MandatorySelectionFactory
from logic.src.policies.mandatory_selection.selection_dispatcher_portfolio import PortfolioDispatcher
from logic.src.policies.mandatory_selection.selection_fptas_knapsack import FPTASKnapsackSelection


def _from(strategy: str, **sub_configs):
    return MandatorySelectionFactory.create_from_config(MandatorySelectionConfig(strategy=strategy, **sub_configs))


def test_bernoulli_probability_reaches_the_strategy():
    strategy = _from("bernoulli_random", bernoulli_random=BernoulliRandomSelectionConfig(p=0.3, min_fill=10.0))
    assert strategy.p == pytest.approx(0.3)
    assert strategy.min_fill == pytest.approx(10.0)


def test_bernoulli_unset_probability_uses_the_context():
    assert _from("bernoulli_random").p is None


def test_filter_and_fan_settings_reach_the_strategy():
    strategy = _from("filter_and_fan", filter_and_fan=FilterAndFanSelectionConfig(filter_width=7, fan_depth=2))
    assert (strategy.filter_width, strategy.fan_depth) == (7, 2)


def test_revenue_settings_reach_the_strategy():
    strategy = _from("revenue", revenue=RevenueSelectionConfig(revenue_kg=2.0, revenue_threshold=5.0))
    assert (strategy.revenue_kg, strategy.revenue_threshold) == (2.0, 5.0)


def test_fractional_knapsack_settings_reach_the_strategy():
    strategy = _from(
        "fractional_knapsack", fractional_knapsack=FractionalKnapsackSelectionConfig(n_vehicles=3, cost_per_km=0.5)
    )
    assert (strategy.n_vehicles, strategy.cost_per_km) == (3, 0.5)


def test_mip_knapsack_penalty_reaches_the_strategy():
    strategy = _from("mip_knapsack", mip_knapsack=MIPKnapsackSelectionConfig(overflow_penalty_frac=2.5))
    assert strategy.overflow_penalty_frac == pytest.approx(2.5)


def test_fptas_settings_reach_the_strategy():
    strategy = _from(
        "fptas_knapsack",
        fptas_knapsack=FPTASKnapsackSelectionConfig(epsilon=0.2, alpha=0.3, overflow_penalty_frac=4.0),
    )
    assert isinstance(strategy, FPTASKnapsackSelection)
    assert (strategy.epsilon, strategy.alpha, strategy.overflow_penalty_frac) == (0.2, 0.3, 4.0)


def test_portfolio_settings_reach_the_strategy_under_the_public_name():
    config = PortfolioDispatcherSelectionConfig(
        dispatcher_mode="intersection", dispatcher_candidate_strategies=["last_minute", "deadline"]
    )
    strategy = _from("dispatcher_portfolio", portfolio_dispatcher=config)
    assert isinstance(strategy, PortfolioDispatcher)
    assert strategy.dispatcher_mode == "intersection"
    assert strategy.dispatcher_candidate_strategies == ["last_minute", "deadline"]


def test_combined_strategies_are_built_from_the_config():
    config = CombinedSelectionConfig(
        strategies=[{"strategy": "last_minute", "threshold": 70.0}, {"strategy": "deadline"}]
    )
    assert _from("combined", combined=config) is not None


def test_portfolio_does_not_silently_skip_a_failing_candidate():
    strategy = PortfolioDispatcher(dispatcher_candidate_strategies=["no_such_strategy"])
    with pytest.raises(ValueError):
        strategy.select_bins(_context())


def test_unset_settings_are_not_reported(caplog):
    with caplog.at_level(logging.WARNING):
        MandatorySelectionFactory.create_strategy("fractional_knapsack")
    assert "does not accept" not in caplog.text


def test_unaccepted_set_setting_is_reported(caplog):
    with caplog.at_level(logging.WARNING):
        MandatorySelectionFactory.create_strategy("fractional_knapsack", not_a_setting=1)
    assert "not_a_setting" in caplog.text


@pytest.mark.parametrize(
    "name",
    [
        "bernoulli_random",
        "filter_and_fan",
        "fptas_knapsack",
        "fractional_knapsack",
        "mip_knapsack",
        "revenue",
        "combined",
        "dispatcher_portfolio",
        "wasserstein",
    ],
)
def test_each_wired_selector_has_a_loadable_yaml(name):
    from pathlib import Path

    root = Path(__file__).resolve().parents[4] / "configs" / "policies" / "other"
    files = list(root.glob(f"ms_{name}.yaml")) + list(
        root.glob(f"ms_{name.replace('dispatcher_', 'dispatcher_')}.yaml")
    )
    if name == "wasserstein":
        files = [root / "ms_wasserstein.yaml"]
    assert files, f"no YAML for {name}"
    assert yaml.safe_load(files[0].read_text()) is not None


def _context():
    from logic.src.interfaces.context import SelectionContext

    n = 4
    return SelectionContext(
        bin_ids=np.arange(n, dtype="int32"),
        current_fill=np.full(n, 50.0),
        threshold=70.0,
        max_fill=100.0,
        bin_volume=2.5,
        bin_density=400.0,
        revenue_kg=1.0,
        vehicle_capacity=100.0,
        distance_matrix=np.zeros((n + 1, n + 1)),
    )
