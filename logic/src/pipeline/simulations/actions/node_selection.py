"""
Action for mandatory bin selection.

This module provides the MandatorySelectionAction class, which identifies
which bins MUST be visited during a simulation day based on configured strategies.

Attributes:
    MandatorySelectionAction: Command for mandatory bin identification.

Example:
    >>> # action = MandatorySelectionAction()
    >>> # action.execute(context)
"""

import os
from dataclasses import asdict
from typing import Any, Dict, List, cast

import numpy as np

from logic.src.configs import MandatorySelectionConfig
from logic.src.constants import MAX_CAPACITY_PERCENT, ROOT_DIR  # noqa: F401  (ROOT_DIR: test patch point)
from logic.src.constants.paths import CONFIGS_DIR
from logic.src.interfaces import IBinContainer, ITraversable
from logic.src.interfaces.context.search_context import SearchContext
from logic.src.policies.mandatory_selection import MandatorySelectionFactory, SelectionContext
from logic.src.utils.configs.config_loader import load_config

from .base import (
    SimulationAction,
    _append_context_list,
    _as_plain_dict,
    _ensure_mutable_config,
    _file_yaml_pairs,
    _flatten_config,
    _inject_acceptance_into_config,
    _is_mapping,
    _is_non_string_sequence,
    _live_ac_payload,
    _load_policy_yaml_section,
    _record_live_params,
    _set_live_capture_meta,
)


class MandatorySelectionAction(SimulationAction):
    """
    Identifies which bins are targets for collection based on various strategies.

    Attributes:
        None
    """

    def execute(self, context: Dict[str, Any]) -> None:  # noqa: C901
        """
        Execute mandatory selection strategies and update context['mandatory'].

        Args:
            context: Shared dictionary containing simulation state.

        Returns:
            None
        """
        _set_live_capture_meta(context)
        raw_cfg = _ensure_mutable_config(context)
        live_ac = _inject_acceptance_into_config(raw_cfg)
        if live_ac is not None:
            ac_payload = _live_ac_payload(live_ac)
            context["_live_ac"] = ac_payload
            _record_live_params("acceptance_criteria", ac_payload)

        # Early exit if neural network explicitly predicts selection
        model_name = ""
        pol_cfg = context.get("config", {})
        if isinstance(pol_cfg, dict) and "model" in pol_cfg:
            model_name = pol_cfg["model"].get("name", "").lower()
        if "ptr" in model_name or "trans" in model_name:
            return

        # 1. Gather all strategies to run
        strategies = self._gather_strategies(context)
        if not strategies:
            # Check if there is a 'cached' tour and no strategies
            if context.get("cached"):
                context["mandatory"] = []
            # If no strategies and not cached, then no mandatory bins
            context["mandatory"] = []
            return

        # 2. Execute all strategies and union results
        bins = context["bins"]
        # Use IBinContainer protocol for unified bin access
        if isinstance(bins, IBinContainer):
            current_fill = bins.get("c")
            accumulation_rates = bins.get("means")
            std_deviations = bins.get("std")
        else:
            # Fallback for dict-like objects
            current_fill = bins.get("c") if hasattr(bins, "get") else getattr(bins, "c", None)
            accumulation_rates = bins.get("means") if hasattr(bins, "get") else getattr(bins, "means", None)
            std_deviations = bins.get("std") if hasattr(bins, "get") else getattr(bins, "std", None)

        n_bins = getattr(bins, "n", None)
        if n_bins is None:
            n_bins = len(current_fill) if current_fill is not None else 0

        final_mandatory: set = set()

        for strat_info in strategies:
            # 1. Normalize name and parameters
            if "config" in strat_info:
                m_config = strat_info["config"]
                s_name = m_config.strategy
                if s_name is None:
                    continue

                # Extract sub-config parameters for SelectionContext
                s_params = m_config.params.copy()

                # Find the sub-config attribute name
                from logic.src.policies.mandatory_selection.base.selection_factory import CONFIG_MAPPING

                attr_name = CONFIG_MAPPING.get(s_name.lower())
                if attr_name and hasattr(m_config, attr_name):
                    sub_config = getattr(m_config, attr_name)
                    if hasattr(sub_config, "__dataclass_fields__"):
                        s_params.update(asdict(sub_config))
                    elif isinstance(sub_config, dict):
                        s_params.update(sub_config)
            else:
                s_name = strat_info.get("name")
                s_params = strat_info.get("params", {})

            if not s_name:
                continue

            live_ms = {"name": s_name, "params": dict(s_params) if hasattr(s_params, "items") else {}}
            _append_context_list(context, "_live_ms", live_ms)
            _record_live_params("mandatory_selection", live_ms)

            # 2. Extract threshold and frequency parameters
            thresh = None
            if hasattr(s_params, "get"):
                thresh = (
                    s_params.get("threshold")
                    if s_params.get("threshold") is not None
                    else s_params.get("period")
                    if s_params.get("period") is not None
                    else s_params.get("n_sectors")
                    if s_params.get("n_sectors") is not None
                    else s_params.get("n_points")
                    if s_params.get("n_points") is not None
                    else s_params.get("frequency")
                    if s_params.get("frequency") is not None
                    else s_params.get("confidence_factor")
                    if s_params.get("confidence_factor") is not None
                    # Fallback for learned strategy
                    else s_params.get("learned_threshold")
                    if s_params.get("learned_threshold") is not None
                    # Fallback to context/global thresh if not in strategy params
                    else context.get("threshold")
                )

            if thresh is None:
                thresh = context.get("threshold")

            # Basic sanitization
            final_thresh = float(thresh) if thresh is not None else 0.0

            # Extract optional advanced parameters from strategy params
            _p = s_params if hasattr(s_params, "get") else {}

            def _pf(key: str, default: float, _params: Any = _p) -> float:
                v = _params.get(key) if hasattr(_params, "get") else None
                return float(v) if v is not None else default

            def _pi(key: str, default: int, _params: Any = _p) -> int:
                v = _params.get(key) if hasattr(_params, "get") else None
                return int(v) if v is not None else default

            sel_ctx = SelectionContext(
                bin_ids=np.arange(0, n_bins, dtype="int32"),
                current_fill=np.array(current_fill) if current_fill is not None else np.array([]),
                accumulation_rates=accumulation_rates,
                std_deviations=std_deviations,
                current_day=context.get("day", 0),
                threshold=final_thresh,
                next_collection_day=context.get("next_collection_day"),
                distance_matrix=context.get("distance_matrix"),
                paths_between_states=context.get("paths_between_states"),
                vehicle_capacity=context.get("max_capacity", 100.0),
                revenue_kg=context.get("revenue_kg", 1.0),
                bin_density=context.get("bin_density", 1.0),
                bin_volume=context.get("bin_volume", 2.5),
                max_fill=MAX_CAPACITY_PERCENT,
                # New fields for advanced strategies
                horizon_days=_pi("horizon_days", 3),
                critical_threshold=_pf("critical_threshold", 0.90),
                synergy_threshold=_pf("synergy_threshold", 0.60),
                radius=_pf("radius", 10.0),
                n_vehicles=_pi("n_vehicles", 1),
                cost_per_km=_pf("cost_per_km", 0.0),
                use_eoq_threshold=bool(_p.get("use_eoq_threshold", False)) if hasattr(_p, "get") else False,
                holding_cost_per_kg_day=_pf("holding_cost_per_kg_day", 0.0),
                ordering_cost_per_visit=_pf("ordering_cost_per_visit", 0.0),
                rollout_horizon=_pi("rollout_horizon", 5),
                rollout_base_policy=_p.get("rollout_base_policy", "last_minute")
                if hasattr(_p, "get")
                else "last_minute",
                rollout_n_scenarios=_pi("rollout_n_scenarios", 1),
                whittle_discount=_pf("whittle_discount", 0.95),
                whittle_grid_size=_pi("whittle_grid_size", 21),
                cvar_alpha=_pf("cvar_alpha", 0.95),
                savings_min_fill_ratio=_pf("savings_min_fill_ratio", 0.5),
                service_radius=_pf("service_radius", 5.0),
                modular_alpha=_pf("modular_alpha", 1.0),
                modular_budget=_pi("modular_budget", 0),
                learned_model_path=_p.get("learned_model_path") if hasattr(_p, "get") else None,
                learned_threshold=_pf("learned_threshold", 0.5),
                dispatcher_state_path=_p.get("dispatcher_state_path") if hasattr(_p, "get") else None,
                dispatcher_candidate_strategies=_p.get("dispatcher_candidate_strategies")
                if hasattr(_p, "get")
                else None,
                dispatcher_exploration=_pf("dispatcher_exploration", 1.0),
                dispatcher_mode=_p.get("dispatcher_mode", "union") if hasattr(_p, "get") else "union",
                wasserstein_radius=_pf("wasserstein_radius", 0.1),
                wasserstein_p=_pi("wasserstein_p", 1),
                overflow_penalty_frac=_pf("overflow_penalty_frac", 1.0),
                coordinates=bins.get("coords") if isinstance(bins, IBinContainer) else context.get("coordinates"),
                current_collection_day=_pi("current_collection_day", 0),
                seed=s_params.get("seed") if hasattr(s_params, "get") else None,
            )

            if "config" in strat_info:
                # Use structured config
                m_config = strat_info["config"]
                strategy = MandatorySelectionFactory.create_from_config(m_config)
                res, sub_ctx = strategy.select_bins(sel_ctx)
                # s_name is already normalized above
            elif s_name == "select_all":
                res = (sel_ctx.bin_ids + 1).tolist()
                sub_ctx = SearchContext.initialize(selection_metrics={"strategy": "select_all"})
            else:
                strategy = MandatorySelectionFactory.create_strategy(str(s_name), **cast(Dict[str, Any], s_params))
                res, sub_ctx = strategy.select_bins(sel_ctx)

            # Ensure list
            if hasattr(res, "tolist"):
                res = res.tolist()
            elif not isinstance(res, list):
                res = list(res)

            # Update SearchContext in simulation context
            if "search_context" not in context:
                context["search_context"] = sub_ctx
            else:
                context["search_context"] = context["search_context"].merge(sub_ctx)

            final_mandatory.update(res)

        context["mandatory"] = sorted(list(final_mandatory))

    def _gather_strategies(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gather all strategies from configuration.

        Args:
            context: Shared dictionary containing simulation state.

        Returns:
            A list of dictionaries representing resolved strategies.
        """
        strategies: List[Dict[str, Any]] = []
        raw_cfg = context.get("config", {})
        flat_cfg = _flatten_config(raw_cfg)
        config_mandatory = flat_cfg.get("mandatory_selection")

        if isinstance(config_mandatory, MandatorySelectionConfig):
            strategies.append({"config": config_mandatory})
        elif config_mandatory:
            items_to_parse = config_mandatory if _is_non_string_sequence(config_mandatory) else [config_mandatory]

            for item in items_to_parse:
                strategies.extend(self._parse_strategy_item(item))
        return strategies

    def _parse_strategy_item(self, item: Any) -> List[Dict[str, Any]]:  # noqa: C901
        """Parse a single mandatory configuration item.

        Args:
            item: The configuration item to parse (Config, string path, (file, variant) tuple, or dict).

        Returns:
            A list of strategy definitions parsed from the item.
        """
        strategies: List[Dict[str, Any]] = []
        if isinstance(item, MandatorySelectionConfig):
            strategies.append({"config": item})
            return strategies

        file_pairs = _file_yaml_pairs(item)
        if file_pairs:
            for file_key, variant in file_pairs:
                section = _load_policy_yaml_section(file_key, variant)
                if "strategy" in section:
                    cfg_dict = dict(section)
                    strat_name = cfg_dict.pop("strategy")
                    strategies.append({"name": strat_name, "params": cfg_dict})
                else:
                    for k, v in section.items():
                        strategies.append({"name": k, "params": v if _is_mapping(v) else {}})
            return strategies

        if (isinstance(item, (dict, ITraversable)) or hasattr(item, "items")) and len(item) == 1:
            item_dict2 = _as_plain_dict(item)
            if "strategy" in item_dict2:
                strat_name = item_dict2.pop("strategy")
                strategies.append({"name": strat_name, "params": item_dict2})
            else:
                for k, v in item_dict2.items():
                    strategies.append({"name": k, "params": v if _is_mapping(v) else {}})
            return strategies
        elif isinstance(item, str) and (item.endswith(".xml") or item.endswith(".yaml")):
            fpath = os.path.join(CONFIGS_DIR, "policies", item)
            cfg = load_config(fpath)
            if "config" in cfg and len(cfg) == 1:
                cfg = cfg["config"]

            if "strategy" in cfg:
                # Convert to dict to allow pop() and type-safe kwargs
                cfg_dict = dict(cfg.items()) if hasattr(cfg, "items") else dict(cfg)  # type: ignore
                strat_name = cfg_dict.pop("strategy")
                strategies.append({"name": strat_name, "params": cfg_dict})
            else:
                for k, v in cfg.items():
                    strategies.append({"name": k, "params": v if isinstance(v, ITraversable) else {}})
        elif isinstance(item, ITraversable):
            # Convert to dict to allow pop() and type-safe kwargs
            item_dict = dict(item.items()) if hasattr(item, "items") else dict(item)  # type: ignore
            if "strategy" in item_dict:
                strat_name = item_dict.pop("strategy")
                strategies.append({"name": strat_name, "params": item_dict})
            else:
                for k, v in item_dict.items():
                    strategies.append({"name": k, "params": v if isinstance(v, (dict, ITraversable)) else {}})
        else:
            strategies.append({"name": item, "params": {}})
        return strategies
