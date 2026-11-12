"""Sensor-only increment estimates: what a capped fill sensor reveals (a - lost), beside the full-deposit estimates."""

import dataclasses

import numpy as np
import pytest
import yaml
from logic.src.configs.policies.packing.threshold import LookaheadSelectionConfig, ServiceLevelSelectionConfig
from logic.src.pipeline.simulations.bins import Bins


def test_sensor_estimates_miss_the_waste_lost_to_overflow(mock_bins_params_loader, tmp_path) -> None:
    bins = Bins(n=2, data_dir=str(tmp_path))
    for _ in range(3):
        bins._process_filling(np.array([30.0, 60.0]))
    # Bin 1 never overflows: both estimates see 30, 30, 30. Bin 2 reaches 100 on day 2 (observed 60, 40, 0).
    assert bins.means == pytest.approx([30.0, 60.0])
    assert bins.sensor_means == pytest.approx([30.0, np.mean([60.0, 40.0, 0.0])])
    assert bins.sensor_std == pytest.approx([0.0, np.std([60.0, 40.0, 0.0], ddof=1)])
    assert bins.std == pytest.approx([0.0, 0.0])


def test_after_a_collection_the_sensor_sees_the_full_deposit_again(mock_bins_params_loader, tmp_path) -> None:
    bins = Bins(n=1, data_dir=str(tmp_path))
    bins._process_filling(np.array([70.0]))
    bins._process_filling(np.array([70.0]))  # capped: observed 30
    bins.collect([0, 1, 0])  # a tour: bin ids are 1-based, 0 is the depot
    bins._process_filling(np.array([70.0]))  # from empty: observed 70
    assert bins.sensor_means == pytest.approx([np.mean([70.0, 30.0, 70.0])])
    assert bins.means == pytest.approx([70.0])


def test_the_reported_rules_keep_the_full_estimates() -> None:
    assert ServiceLevelSelectionConfig().estimates == "full" and LookaheadSelectionConfig().estimates == "full"
    root = "logic/configs/policies/other/packing"
    with open(f"{root}/pk_service_level.yaml") as fh:
        sl = yaml.safe_load(fh)
    with open(f"{root}/pk_lookahead.yaml") as fh:
        la = yaml.safe_load(fh)
    with open(f"{root}/pk_sensitivity.yaml") as fh:
        sens = yaml.safe_load(fh)
    # The rule files only hold the reported variants: a policy selecting a whole file must not pick up a sensitivity one.
    assert set(la) == {"lookahead"} and set(sl) == {"service_level1", "service_level2"}
    assert "estimates" not in sl["service_level1"] and "estimates" not in la["lookahead"]
    sensor = ServiceLevelSelectionConfig(**{k: v for k, v in sens["service_level1_sensor"].items() if k != "strategy"})
    assert dataclasses.replace(sensor, estimates="full") == ServiceLevelSelectionConfig(threshold=0.84, horizon_days=1)
    assert sens["lookahead_sensor"]["estimates"] == "sensor"
