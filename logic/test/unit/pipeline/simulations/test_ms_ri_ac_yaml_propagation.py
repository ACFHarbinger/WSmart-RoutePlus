"""Fail-before tests: MS / RI / AC yaml parameters must reach the live action path.

The paper policies store these as ``{file.yaml: [variant]}`` (often an OmegaConf
ListConfig). Constructors used to honour only a string variant and silently ran
on dataclass defaults; acceptance yaml never became ``acceptance_criterion``.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from logic.src.pipeline.simulations.actions.node_selection import MandatorySelectionAction
from logic.src.pipeline.simulations.actions.route_improvement import RouteImprovementAction
from logic.src.policies.route_improvement.local_search import ClassicalLocalSearchRouteImprover
from omegaconf import OmegaConf

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_last_minute_list_variant_loads_yaml_threshold() -> None:
    """``{ms_last_minute.yaml: [last_minute_cf70]}`` must not be treated as a strategy name."""
    action = MandatorySelectionAction()
    parsed = action._parse_strategy_item({"other/ms_last_minute.yaml": ["last_minute_cf70"]})
    assert parsed, "expected at least one strategy"
    assert parsed[0]["name"] == "last_minute"
    assert parsed[0]["params"].get("threshold") == 70


def test_last_minute_omegaconf_listconfig_variant_loads_yaml_threshold() -> None:
    item = OmegaConf.create({"other/ms_last_minute.yaml": ["last_minute_cf90"]})
    parsed = MandatorySelectionAction()._parse_strategy_item(item)
    assert parsed[0]["name"] == "last_minute"
    assert parsed[0]["params"].get("threshold") == 90


def test_lookahead_default_list_variant_reads_yaml_day() -> None:
    parsed = MandatorySelectionAction()._parse_strategy_item({"other/ms_lookahead.yaml": ["default"]})
    assert parsed[0]["name"] == "lookahead"
    assert parsed[0]["params"].get("current_collection_day") == 0


def test_bmc_yaml_is_injected_as_typed_acceptance_criterion() -> None:
    """Policy-level ``acceptance_criteria: {ac_bmc.yaml: [bmc]}`` must become BMC 100 / 0.995.

    PSOMA's dataclass default is T0=3 / alpha=0.9; without injection the yaml is dropped.
    """
    bins = MagicMock()
    bins.get.side_effect = lambda key: {
        "c": np.zeros(3),
        "means": np.zeros(3),
        "std": np.zeros(3),
        "coords": None,
    }.get(key)
    bins.n = 3
    context: Dict[str, Any] = {
        "bins": bins,
        "config": {
            "psoma": {
                "custom": [
                    {"acceptance_criteria": {"other/ac_bmc.yaml": ["bmc"]}},
                    {"mandatory_selection": []},
                ]
            }
        },
        "day": 0,
        "max_capacity": 100.0,
    }
    MandatorySelectionAction().execute(context)
    live = context.get("_live_ac") or _find_acceptance(context["config"])
    assert live is not None, "acceptance yaml was not resolved onto the live config"
    method, params = _ac_method_and_params(live)
    assert method == "bmc"
    assert params["initial_temp"] == pytest.approx(100.0)
    assert params["alpha"] == pytest.approx(0.995)
    assert params["seed"] == 42


def test_oi_yaml_is_injected_as_only_improving() -> None:
    bins = MagicMock()
    bins.get.side_effect = lambda key: {
        "c": np.zeros(2),
        "means": np.zeros(2),
        "std": np.zeros(2),
        "coords": None,
    }.get(key)
    bins.n = 2
    context: Dict[str, Any] = {
        "bins": bins,
        "config": {"hgs": {"acceptance_criteria": {"other/ac_oi.yaml": ["oi"]}}},
        "day": 0,
        "max_capacity": 100.0,
    }
    MandatorySelectionAction().execute(context)
    live = context.get("_live_ac") or _find_acceptance(context["config"])
    assert live is not None
    method, _params = _ac_method_and_params(live)
    assert method in ("oi", "only_improving")


def test_psoma_from_config_consumes_injected_bmc_not_dataclass_t0() -> None:
    """PSOMA ``from_config`` must instantiate BMC from ``acceptance_criterion``, not T0=3."""
    from types import SimpleNamespace

    from logic.src.configs.policies.other.acceptance_criteria import (
        AcceptanceConfig,
        BoltzmannAcceptanceConfig,
    )
    from logic.src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.params import (  # noqa: E501
        PSOMAParams,
    )

    cfg = SimpleNamespace(
        pop_size=20,
        omega=1.0,
        c1=2.0,
        c2=2.0,
        max_iterations=5,
        x_min=0.0,
        x_max=4.0,
        v_min=-4.0,
        v_max=4.0,
        L=30,
        T0=3.0,
        lambda_cooling=0.9,
        time_limit=2.0,
        vrpp=True,
        seed=42,
        acceptance_criterion=AcceptanceConfig(
            method="bmc",
            params=BoltzmannAcceptanceConfig(initial_temp=100.0, alpha=0.995, seed=42),
        ),
    )
    params = PSOMAParams.from_config(cfg)
    criterion = params.acceptance_criterion
    assert criterion is not None
    assert float(criterion.T) == pytest.approx(100.0)
    assert float(criterion.alpha) == pytest.approx(0.995)


def test_alns_from_config_consumes_injected_bmc_on_attrdict() -> None:
    """Attribute mappings and plain mappings both preserve injected BMC."""
    from logic.src.configs.policies.other.acceptance_criteria import (
        AcceptanceConfig,
        BoltzmannAcceptanceConfig,
    )
    from logic.src.pipeline.simulations.actions.base import _attrify
    from logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.params import (
        ALNSParams,
    )

    injected = AcceptanceConfig(
        method="bmc",
        params=BoltzmannAcceptanceConfig(initial_temp=100.0, alpha=0.995, seed=42),
    )
    raw = {
        "time_limit": 2.0,
        "max_iterations": 20,
        "start_temp": 0.0,
        "cooling_rate": 0.995,
        "reaction_factor": 0.1,
        "acceptance_criterion": injected,
    }
    wrapped = ALNSParams.from_config(_attrify(raw))
    criterion = wrapped.acceptance_criterion
    assert criterion is not None
    assert float(criterion.T) == pytest.approx(100.0)
    assert float(criterion.alpha) == pytest.approx(0.995)

    patched = ALNSParams.from_config(raw)
    assert float(patched.acceptance_criterion.T) == pytest.approx(100.0)


def test_hgs_from_config_consumes_injected_oi() -> None:
    from types import SimpleNamespace

    from logic.src.configs.policies.other.acceptance_criteria import AcceptanceConfig, OnlyImprovingConfig
    from logic.src.policies.acceptance_criteria.only_improving import OnlyImproving
    from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.params import HGSParams

    cfg = SimpleNamespace(
        acceptance_criterion=AcceptanceConfig(method="oi", params=OnlyImprovingConfig()),
    )
    params = HGSParams.from_config(cfg)
    assert isinstance(params.acceptance_criterion, OnlyImproving)


def test_cls_2opt_runs_only_two_opt_intra() -> None:
    """``ls_operator: 2opt`` in ri_cls.yaml must not silently run the full operator suite."""
    dist = np.array(
        [
            [0, 10, 20, 10],
            [10, 0, 10, 20],
            [20, 10, 0, 10],
            [10, 20, 10, 0],
        ],
        dtype=float,
    )
    called: List[str] = []

    def _track(name: str):
        def _fn(*_a, **_k):
            called.append(name)
            return False

        return _fn

    with patch("logic.src.policies.helpers.local_search.local_search_manager.LocalSearchManager") as mock_cls:
        manager = MagicMock()
        mock_cls.return_value = manager
        manager.relocate.side_effect = _track("relocate")
        manager.swap.side_effect = _track("swap")
        manager.two_opt_intra.side_effect = _track("two_opt_intra")
        manager.or_opt.side_effect = _track("or_opt")
        manager.two_opt_star.side_effect = _track("two_opt_star")
        manager.swap_star.side_effect = _track("swap_star")
        manager.three_opt_intra.side_effect = _track("three_opt_intra")
        manager.four_opt_intra.side_effect = _track("four_opt_intra")
        manager.get_routes.return_value = [[1, 2, 3]]

        ClassicalLocalSearchRouteImprover().process(
            [0, 2, 1, 3, 0],
            distance_matrix=dist,
            ls_operator="2opt",
            iterations=1,
        )

    assert "two_opt_intra" in called
    assert "relocate" not in called
    assert "swap" not in called


def test_ri_yaml_iterations_are_passed_without_clobbering_policy_time_limit() -> None:
    """CLS yaml ``iterations: 1000`` must reach process(); policy ``time_limit`` must stay 60."""
    captured: Dict[str, Any] = {}

    proc = MagicMock()

    def _process(tour, **kwargs):
        captured.update(kwargs)
        return tour, {"algorithm": "spy"}

    proc.process.side_effect = _process

    context: Dict[str, Any] = {
        "tour": [0, 1, 2, 0],
        "distance_matrix": np.zeros((3, 3)),
        "time_limit": 60.0,
        "config": {"route_improvement": {"other/ri_cls.yaml": ["default"]}},
        "sample_id": 0,
        "day": 0,
        "policy_name": "alns",
    }
    with patch(
        "logic.src.policies.route_improvement.base.factory.RouteImproverFactory.create",
        return_value=proc,
    ):
        RouteImprovementAction().execute(context)

    assert captured.get("iterations") == 1000
    assert captured.get("ls_operator") == "2opt"
    assert captured.get("time_limit") == pytest.approx(30.0)
    assert context["time_limit"] == pytest.approx(60.0)


def _find_acceptance(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "acceptance_criterion" in obj:
            return obj["acceptance_criterion"]
        for value in obj.values():
            found = _find_acceptance(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_acceptance(item)
            if found is not None:
                return found
    return None


def _ac_method_and_params(live: Any) -> tuple:
    if isinstance(live, dict):
        method = str(live.get("method", "")).lower()
        params = live.get("params") or {}
        if hasattr(params, "__dict__"):
            params = {k: v for k, v in vars(params).items() if not k.startswith("_")}
        return method, params
    method = str(getattr(live, "method", "")).lower()
    params_obj = getattr(live, "params", None)
    if params_obj is None:
        return method, {}
    if hasattr(params_obj, "__dict__"):
        params = {k: v for k, v in vars(params_obj).items() if not k.startswith("_")}
    elif isinstance(params_obj, dict):
        params = params_obj
    else:
        params = {}
    return method, params


PAPER_POLICIES = [
    "aco_hh",
    "alns",
    "bpc",
    "hgs",
    "ms_bpc_sp",
    "na",
    "psoma",
    "pg_clns",
    "swc_tcf",
]


def test_expanded_paper_policies_match_ms_ri_ac_yaml() -> None:
    """After ``expand_policy_configs``, live parse of every paper-policy variant equals yaml."""
    from hydra import compose, initialize_config_dir
    from logic.src.constants.paths import CONFIGS_DIR
    from logic.src.pipeline.features.test.config import expand_policy_configs
    from logic.src.pipeline.simulations.actions.base import (
        _flatten_config,
        _inject_acceptance_into_config,
        _live_ac_payload,
        _load_policy_yaml_section,
    )
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base=None, config_dir=CONFIGS_DIR):
        cfg = compose(
            config_name="config",
            overrides=[
                "tasks=test_sim",
                f"sim.policies=[{','.join(PAPER_POLICIES)}]",
                "sim.data_distribution=emp",
            ],
        )
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.sim, False)
    cfg.sim.config_path = {}
    cfg.sim.full_policies = []
    expand_policy_configs(cfg)

    action = MandatorySelectionAction()
    assert cfg.sim.config_path, "expander stored no policies"
    cls_yaml = _load_policy_yaml_section("other/ri_cls.yaml", "default")
    bmc_yaml = _load_policy_yaml_section("other/ac_bmc.yaml", "bmc")["params"]
    saw_last_minute = False
    saw_lookahead = False
    saw_bmc = False
    saw_oi = False
    saw_cls = False
    for stored in cfg.sim.config_path.values():
        cfg_copy = OmegaConf.to_container(stored, resolve=True) if OmegaConf.is_config(stored) else stored
        parsed = action._gather_strategies({"config": cfg_copy})
        assert parsed, "mandatory selection did not resolve"
        for item in parsed:
            if item["name"] == "last_minute":
                saw_last_minute = True
                assert item["params"]["threshold"] in (70, 90)
            elif item["name"] == "lookahead":
                saw_lookahead = True
                assert item["params"].get("current_collection_day") == 0
            else:
                raise AssertionError(f"unexpected strategy {item['name']}")
        ac = _inject_acceptance_into_config(cfg_copy)
        live_ac = _live_ac_payload(ac) if ac else None
        if live_ac and live_ac.get("method") == "bmc":
            saw_bmc = True
            assert live_ac["params"]["initial_temp"] == pytest.approx(bmc_yaml["initial_temp"])
            assert live_ac["params"]["alpha"] == pytest.approx(bmc_yaml["alpha"])
            assert live_ac["params"]["seed"] == bmc_yaml["seed"]
        if live_ac and live_ac.get("method") == "oi":
            saw_oi = True
        ri_val = _flatten_config(cfg_copy).get("route_improvement")
        if ri_val:
            saw_cls = True
            assert cls_yaml["ls_operator"] == "2opt"
            assert cls_yaml["iterations"] == 1000
            assert cls_yaml["time_limit"] == pytest.approx(30.0)
    assert saw_last_minute and saw_lookahead and saw_bmc and saw_oi and saw_cls


@pytest.mark.integration
@pytest.mark.slow
def test_live_test_sim_captures_ms_ri_ac(tmp_path, monkeypatch) -> None:
    """Capture MS / RI / AC inside ``run_simulation`` (the ``main.py test_sim`` path)."""
    import json

    from hydra import compose, initialize_config_dir
    from logic.controllers.jobs.pipeline_runner import run_simulation
    from logic.src.constants.paths import CONFIGS_DIR
    from omegaconf import OmegaConf

    capture = tmp_path / "live.jsonl"
    monkeypatch.setenv("WSR_CAPTURE_STRATEGY_JSON", str(capture))
    custom = {
        "engine": "custom",
        "time_limit": 2.0,
        "max_iterations": 20,
        "mandatory_selection": {"other/ms_last_minute.yaml": "last_minute_cf70"},
        "route_improvement": {"other/ri_cls.yaml": "default"},
        "acceptance_criteria": {"other/ac_bmc.yaml": "bmc"},
    }
    with initialize_config_dir(version_base=None, config_dir=CONFIGS_DIR):
        cfg = compose(
            config_name="config",
            overrides=[
                "tasks=test_sim",
                "sim.graph.area=riomaior",
                "sim.graph.num_loc=20",
                "sim.graph.n_days=10",
                "sim.data_distribution=emp",
                "sim.cpu_cores=2",
                "sim.no_cuda=true",
                "sim.graph.n_samples=1",
                "sim.graph.load_dataset=null",
            ],
        )
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.sim, False)
    OmegaConf.set_struct(cfg.sim.graph, False)
    cfg.sim.policies = [{"alns": custom}]
    cfg.sim.config_path = {}
    cfg.sim.full_policies = []
    cfg.task = "test_sim"
    run_simulation(cfg)

    records = [json.loads(line) for line in capture.read_text().splitlines() if line.strip()]
    kinds = {row["kind"] for row in records}
    assert "mandatory_selection" in kinds
    assert "route_improvement" in kinds
    assert "acceptance_criteria" in kinds
    ms = next(row["payload"] for row in records if row["kind"] == "mandatory_selection")
    assert ms["name"] == "last_minute"
    assert ms["params"]["threshold"] == 70
    ri = next(row["payload"] for row in records if row["kind"] == "route_improvement")
    assert "classical_local_search" in ri["methods"]
    assert ri["ls_operator"] == "2opt"
    assert ri["iterations"] == 1000
    assert ri["time_limit"] == pytest.approx(30.0)
    ac = next(row["payload"] for row in records if row["kind"] == "acceptance_criteria")
    assert ac["method"] == "bmc"
    assert ac["params"]["initial_temp"] == pytest.approx(100.0)
    assert ac["params"]["alpha"] == pytest.approx(0.995)
    assert "consumer_acceptance" in kinds
    assert "consumer_last_minute" in kinds
    assert "consumer_cls" in kinds
    bmc = next(
        row["payload"]
        for row in records
        if row["kind"] == "consumer_acceptance" and row["payload"].get("name") == "bmc"
    )
    assert bmc["params"]["initial_temp"] == pytest.approx(100.0)
    assert bmc["params"]["alpha"] == pytest.approx(0.995)
    assert bmc.get("instance_T") == pytest.approx(100.0)
    assert bmc.get("instance_alpha") == pytest.approx(0.995)
    lm = next(row["payload"] for row in records if row["kind"] == "consumer_last_minute")
    assert lm["threshold"] == pytest.approx(70.0)
    cls_live = next(row["payload"] for row in records if row["kind"] == "consumer_cls")
    assert cls_live["ls_operator"] == "2opt"
    assert cls_live["iterations"] == 1000
    assert cls_live["time_limit"] == pytest.approx(30.0)
