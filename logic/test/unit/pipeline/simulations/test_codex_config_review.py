"""Regression checks for delivered config propagation."""

from typing import Any, Iterator
from unittest.mock import MagicMock

import numpy as np
from logic.src.pipeline.simulations.actions.node_selection import MandatorySelectionAction
from logic.src.pipeline.simulations.actions.route_improvement import RouteImprovementAction
from omegaconf import OmegaConf


def test_hydra_mapping_keeps_selected_service_level():
    cfg = OmegaConf.create({"mandatory_selection": {"other/ms_service_level.yaml": "service_level2"}})
    parsed = MandatorySelectionAction()._gather_strategies({"config": cfg})
    assert len(parsed) == 1
    assert parsed[0]["name"] == "service_level"
    assert parsed[0]["params"]["horizon_days"] == 2


def test_route_improver_yaml_params_do_not_leak_to_next_entry():
    ctx = {"_ri_yaml_params": {"iterations": 1000, "time_limit": 30.0}}
    RouteImprovementAction()._create_processors("fast_tsp", ctx)
    assert not ctx.get("_ri_yaml_params")


def test_inject_acceptance_into_struct_locked_nested_hydra_node() -> None:
    """Live ``test_sim`` keeps nested DictConfig children under a plain dict."""
    bins = MagicMock()
    bins.get.side_effect = lambda key: {
        "c": np.zeros(2),
        "means": np.zeros(2),
        "std": np.zeros(2),
        "coords": None,
    }.get(key)
    bins.n = 2
    nested = OmegaConf.create({"acceptance_criteria": {"other/ac_bmc.yaml": "bmc"}})
    OmegaConf.set_struct(nested, True)
    context = {
        "bins": bins,
        "config": {"alns": nested},
        "day": 0,
        "max_capacity": 100.0,
        "mandatory_selection": [],
    }
    MandatorySelectionAction().execute(context)
    live = context.get("_live_ac")
    assert live is not None
    assert live["method"] == "bmc"
    assert live["params"]["initial_temp"] == 100.0


class _DayContextLike:
    """Mapping without ``setdefault`` / ``pop``, as ``SimulationDayContext`` is."""

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dict__)

    def __len__(self) -> int:
        return len(self.__dict__)


def test_mandatory_selection_does_not_call_setdefault_on_day_context() -> None:
    bins = MagicMock()
    bins.get.side_effect = lambda key: {
        "c": np.zeros(2),
        "means": np.zeros(2),
        "std": np.zeros(2),
        "coords": None,
    }.get(key)
    bins.n = 2
    ctx = _DayContextLike(
        bins=bins,
        config={"mandatory_selection": {"other/ms_last_minute.yaml": "last_minute_cf70"}},
        day=0,
        max_capacity=100.0,
        threshold=None,
    )
    MandatorySelectionAction().execute(ctx)
    assert ctx.get("_live_ms")
    assert ctx["_live_ms"][0]["params"]["threshold"] == 70


def test_constructor_id_from_expanded_policy_name() -> None:
    from logic.src.pipeline.simulations.actions.base import _constructor_from_policy_id

    assert _constructor_from_policy_id("last_minute_cf70_alns_custom_bmc_cls_emp") == "alns"
    assert _constructor_from_policy_id("lookahead_ms_bpc_sp_custom_emp") == "ms_bpc_sp"
    assert _constructor_from_policy_id("last_minute_cf90_swc_tcf_gurobi_cls_emp") == "swc_tcf"
    assert _constructor_from_policy_id("lookahead_na_amgat_emp") == "na"
