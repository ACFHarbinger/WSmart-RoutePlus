"""launch_campaign --configs: explicit (policy, improver, packing variants) triples, the fixed schedule included."""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, List

import pytest
import yaml

_SPEC = importlib.util.spec_from_file_location(
    "launch_campaign", Path(__file__).resolve().parents[4] / "logic" / "gen" / "launch_campaign.py"
)
assert _SPEC is not None and _SPEC.loader is not None
launch = importlib.util.module_from_spec(_SPEC)
sys.modules["launch_campaign"] = launch  # its dataclass looks the module up while it is defined
_SPEC.loader.exec_module(launch)


def _args(tmp_path: Path, configs: str) -> argparse.Namespace:
    return argparse.Namespace(
        network="rm100",
        dist="emp",
        depots="both",
        per_depot=5,
        days=90,
        cores=8,
        out=str(tmp_path / "out"),
        delayed_start=False,
        initial_fill_days=0,
        policies="",
        model_path=None,
        profit_aware="configs",
        packing="",
        improvers="",
        configs=configs,
    )


def _packings(node: Any, out: List[Any]) -> List[Any]:
    if isinstance(node, dict):
        for key, value in node.items():
            out.append(value) if key == "packing" else _packings(value, out)
    elif isinstance(node, list):
        for item in node:
            _packings(item, out)
    return out


def test_each_triple_runs_its_own_variants_on_every_scenario(tmp_path: Path) -> None:
    configs = "alns/cls/last_minute_cf90+service_level1,cvrp/none/regular_lvl4"
    cells = launch.build_cells(_args(tmp_path, configs), tmp_path / "overlays")
    assert len(cells) == 2 * 2  # two scenario groups (remote, centroid) x two triples
    assert {c.name.split("_", 2)[2] for c in cells} == {"alns_cls", "cvrp_none"}
    assert all("sim.days=90" in c.argv for c in cells)
    alns = yaml.safe_load(next((tmp_path / "overlays" / "alns_cls").glob("*.yaml")).read_text())
    assert _packings(alns, []) == [
        [
            {"other/packing/pk_last_minute.yaml": ["last_minute_cf90"]},
            {"other/packing/pk_service_level.yaml": ["service_level1"]},
        ]
    ]
    static = yaml.safe_load(next((tmp_path / "overlays" / "cvrp_none").glob("*.yaml")).read_text())
    assert _packings(static, []) == [[{"other/packing/pk_regular.yaml": ["regular_lvl4"]}]]


@pytest.mark.parametrize(
    "configs",
    [
        "alns/cls",
        "alns/fast/last_minute_cf90",
        "unknown/cls/last_minute_cf90",
        "alns/cls/regular_lvl9",
        "cvrp/cls/regular_lvl4",
    ],
)
def test_malformed_triples_are_rejected(tmp_path: Path, configs: str) -> None:
    with pytest.raises(ValueError):
        launch.build_cells(_args(tmp_path, configs), tmp_path / "overlays")
