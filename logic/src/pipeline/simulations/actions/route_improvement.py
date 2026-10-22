"""
Action for tour route improvement and refinement.

This module provides the RouteImprovementAction class, which applies
post-processing algorithms (Local Search, ALNS-improvement, etc.) to refine tours.

Attributes:
    RouteImprovementAction: Command for tour refinement.

Example:
    >>> # action = RouteImprovementAction()
    >>> # action.execute(context)
"""

import os
from typing import Any, Dict

from loguru import logger

from logic.src.configs import RouteImprovingConfig
from logic.src.constants.paths import CONFIGS_DIR
from logic.src.interfaces import ITraversable
from logic.src.pipeline.simulations.actions.policy_config import (
    _as_plain_dict,
    _as_policy_mapping,
    _file_yaml_pairs,
    _is_mapping,
    _load_policy_yaml_section,
)
from logic.src.policies.route_improvement import RouteImproverFactory
from logic.src.utils.configs.config_loader import load_config
from logic.src.utils.routing.tours import get_route_cost

from .base import (
    SimulationAction,
    _pop_context_key,
    _record_live_params,
    _set_live_capture_meta,
    bind_context,
)
from logic.src.utils.configs.policy_files import policy_file


def _stops(tour: Any) -> list:
    """Customers and interior depot returns of a tour, as a sorted list."""
    nodes = [int(n) for n in tour]
    while nodes and nodes[0] == 0:
        nodes.pop(0)
    while nodes and nodes[-1] == 0:
        nodes.pop()
    return sorted(nodes)


class RouteImprovementAction(SimulationAction):
    """
    Refines generated collection tours using modular refinement strategies.

    Attributes:
        None
    """

    timed = True
    reads = ("tour", "config")
    writes = ("tour", "cost")
    finalizes = ()
    requires_final = ()

    def execute(self, context: Dict[str, Any]) -> None:
        """
        Refine the generated collection tour.

        Args:
            context: Shared dictionary containing simulation state.
        """

        context = bind_context(context)
        _set_live_capture_meta(context)
        pp_configs = self._get_route_improvement_configs(context)
        if not pp_configs:
            _record_live_params("consumer_ri", {"entries": []})
        tour = getattr(context, "tour", None)
        if not tour or len(tour) <= 2:
            return

        if pp_configs:
            for entry in pp_configs:
                processors = self._create_processors(entry, context)
                self._apply_processors(processors, context)

    def _get_route_improvement_configs(self, context: Dict[str, Any]) -> list:
        """Retrieve and normalize route improvement configurations.

        Args:
            context: Shared dictionary containing simulation state.

        Returns:
            A list of RouteImprovingConfig objects.
        """
        raw_cfg = getattr(context, "config", {})
        flat_cfg = _as_policy_mapping(raw_cfg)
        pp_list = flat_cfg.get("route_improvement") or getattr(context, "route_improvement", None)

        if isinstance(pp_list, RouteImprovingConfig):
            return [pp_list]

        if not pp_list:
            return []

        if not isinstance(pp_list, list):
            if isinstance(pp_list, str) and pp_list.lower() != "none":
                pp_list = [pp_list]
            elif hasattr(pp_list, "__iter__") and not isinstance(pp_list, str) and not hasattr(pp_list, "items"):
                # OmegaConf ListConfig or other non-dict sequence
                pp_list = list(pp_list)
            elif hasattr(pp_list, "items"):
                # New dict format: {"other/route_improvement/ri_ftsp.yaml": ["ftsp"]} → wrap as single item
                pp_list = [pp_list]
            else:
                pp_list = []

        pp_configs = []
        for item in pp_list:
            pp_configs.append(item)
        return pp_configs

    def _create_processors(self, entry: Any, context: Dict[str, Any]) -> list:
        """Create processors from a configuration entry.

        Args:
            entry: The configuration entry for the processors.
            context: Shared dictionary containing simulation state.

        Returns:
            A list of instantiated route improver processors.
        """
        # Each entry owns its own parameters; do not inherit the previous yaml.
        _pop_context_key(context, "_ri_yaml_params")
        if isinstance(entry, RouteImprovingConfig):
            return RouteImproverFactory.create_from_config(entry)

        # Legacy handling for strings/files
        return self._create_legacy_processors(entry, context, RouteImproverFactory)

    def _create_legacy_processors(self, item: Any, context: Dict[str, Any], factory) -> list:  # noqa: C901
        """Handle legacy string/file configuration for processors.

        Args:
            item: The legacy configuration item.
            context: Shared dictionary containing simulation state.
            factory: The factory used to create processors.

        Returns:
            A list of instantiated route improver processors.
        """
        pp_name = ""
        # Resolve dict format: {"other/route_improvement/ri_ftsp.yaml": "default"} or {"other/route_improvement/ri_ftsp.yaml": ["default"]}
        file_pairs = _file_yaml_pairs(item)
        if file_pairs:
            file_key, variant_val = file_pairs[0]
            try:
                cfg = _load_policy_yaml_section(file_key, variant_val)
                if "methods" in cfg:
                    processors = []
                    ri_params = {k: v for k, v in cfg.items() if k != "methods"}
                    context._ri_yaml_params = ri_params
                    live_ri = {"methods": list(cfg["methods"]), **ri_params}
                    context._live_ri = live_ri
                    _record_live_params("route_improvement", live_ri)
                    for method in cfg["methods"]:
                        try:
                            processors.append(factory.create(method, **ri_params))
                        except TypeError:
                            proc = factory.create(method)
                            if hasattr(proc, "config") and isinstance(proc.config, dict):
                                proc.config.update(ri_params)
                            processors.append(proc)
                        except Exception as e:
                            logger.warning(f"Failed to create route improver {method}: {e}")
                    return processors
                for k, v in cfg.items():
                    pp_name = k
                    if _is_mapping(v):
                        context._ri_yaml_params = _as_plain_dict(v)
            except (OSError, ValueError) as e:
                logger.warning(f"Error loading route_improvement config {file_key}: {e}")
                return []
        elif isinstance(item, (dict, ITraversable)) or (hasattr(item, "items") and not isinstance(item, str)):
            item_dict = dict(item.items()) if hasattr(item, "items") else dict(item)
            if len(item_dict) == 1:
                file_key, variant_val = next(iter(item_dict.items()))
                file_key = str(file_key)
                if file_key.endswith(".yaml") or file_key.endswith(".xml"):
                    # Unwrap list variants (including OmegaConf ListConfig) to a single string
                    if hasattr(variant_val, "__iter__") and not isinstance(variant_val, str):
                        try:
                            v_list = list(variant_val)
                            variant_val = v_list[0] if v_list else None
                        except Exception:
                            pass
                    variant_val = str(variant_val) if variant_val is not None else ""
                    fpath = str(policy_file(file_key))
                    try:
                        cfg = load_config(fpath)
                        if "config" in cfg and len(cfg) == 1:
                            cfg = cfg["config"]
                        # Navigate into named variant sub-key if present
                        if variant_val and variant_val in cfg:
                            cfg = cfg[variant_val]
                        elif len(cfg) == 1 and "methods" not in cfg:
                            # Single-key YAML (e.g. ftsp: {methods: [...]}) — navigate in
                            _inner = next(iter(cfg.values()))
                            if isinstance(_inner, dict) and "methods" in _inner:
                                cfg = _inner
                        if "methods" in cfg:
                            processors = []
                            ri_params = {k: v for k, v in cfg.items() if k != "methods"}
                            context._ri_yaml_params = ri_params
                            live_ri = {"methods": list(cfg["methods"]), **ri_params}
                            context._live_ri = live_ri
                            _record_live_params("route_improvement", live_ri)
                            for method in cfg["methods"]:
                                try:
                                    processors.append(factory.create(method, **ri_params))
                                except TypeError:
                                    proc = factory.create(method)
                                    if hasattr(proc, "config") and isinstance(proc.config, dict):
                                        proc.config.update(ri_params)
                                    processors.append(proc)
                                except Exception as e:
                                    logger.warning(f"Failed to create route improver {method}: {e}")
                            return processors
                        for k, v in cfg.items():
                            pp_name = k
                            v_obj: object = v
                            if isinstance(v_obj, ITraversable):
                                for _k, _v in v_obj.items():
                                    setattr(context, _k, _v)
                    except (OSError, ValueError) as e:
                        logger.warning(f"Error loading route_improvement config {file_key}: {e}")
                        return []
        elif isinstance(item, str) and (item.endswith(".xml") or item.endswith(".yaml")):
            fpath = str(policy_file(item))
            try:
                cfg = load_config(fpath)
                if "config" in cfg and len(cfg) == 1:
                    cfg = cfg["config"]

                # Single-key YAML (e.g. ftsp: {methods: [...]}) — navigate in automatically
                if len(cfg) == 1 and "methods" not in cfg:
                    _inner = next(iter(cfg.values()))
                    if isinstance(_inner, dict) and "methods" in _inner:
                        cfg = _inner

                if "methods" in cfg:
                    processors = []
                    ri_params = {k: v for k, v in cfg.items() if k != "methods"}
                    context._ri_yaml_params = ri_params
                    live_ri = {"methods": list(cfg["methods"]), **ri_params}
                    context._live_ri = live_ri
                    _record_live_params("route_improvement", live_ri)
                    for method in cfg["methods"]:
                        try:
                            processors.append(factory.create(method, **ri_params))
                        except TypeError:
                            proc = factory.create(method)
                            if hasattr(proc, "config") and isinstance(proc.config, dict):
                                proc.config.update(ri_params)
                            processors.append(proc)
                        except Exception as e:
                            logger.warning(f"Failed to create route improver {method}: {e}")
                    return processors

                for k, v in cfg.items():
                    pp_name = k
                    v_obj: object = v
                    if isinstance(v_obj, ITraversable):
                        for _k, _v in v_obj.items():
                            setattr(context, _k, _v)
            except (OSError, ValueError) as e:
                logger.warning(f"Error loading route_improvement config {item}: {e}")
                return []
        else:
            pp_name = item

        if not pp_name or (isinstance(pp_name, str) and pp_name.lower() == "none"):
            return []

        try:
            return [factory.create(pp_name)]
        except Exception as e:
            logger.warning(f"Failed to create route improver {pp_name}: {e}")
            return []

    def _apply_processors(self, processors: list, context: Dict[str, Any]) -> None:
        """Apply a list of processors to the tour in the context.

        Args:
            processors: A list of processors to apply.
            context: Shared dictionary containing simulation state.
        """
        tour = getattr(context, "tour", None)
        for processor in processors:
            try:
                pf_params = {k: v for k, v in context.as_kwargs().items() if k not in ("tour", "_ri_yaml_params")}
                ri_yaml = getattr(context, "_ri_yaml_params", None) or {}
                pf_params.update(ri_yaml)
                from logic.src.tracking.logging.modules.policy_viz_emit import PolicyVizStreamSession

                viz_policy = str(
                    getattr(context, "display_name", None) or getattr(context, "policy_name", None) or "unknown"
                )
                with PolicyVizStreamSession(
                    processor,
                    viz_policy,
                    int(getattr(context, "sample_id", 0)),
                    int(getattr(context, "day", 0)),
                    getattr(context, "realtime_log_path", None),
                    getattr(context, "lock", None),
                ):
                    refined_tour, metrics = processor.process(tour, **pf_params)
                if refined_tour != tour:
                    dist_matrix = getattr(context, "distance_matrix", None)
                    new_cost = get_route_cost(dist_matrix, refined_tour) if dist_matrix is not None else None
                    old_cost = getattr(context, "cost", None)
                    if old_cost is None and dist_matrix is not None:
                        old_cost = get_route_cost(dist_matrix, tour)
                    # A longer reordering of the same stops is not an improvement.
                    # Insertion improvers add profitable bins and split repairs add
                    # depot returns; both lengthen the tour on purpose, so only a
                    # same-stops tour is judged by distance.
                    same_stops = _stops(refined_tour) == _stops(tour)
                    if (
                        same_stops
                        and new_cost is not None
                        and old_cost is not None
                        and float(new_cost) > float(old_cost) + 1e-9
                    ):
                        continue

                    tour = refined_tour
                    context.tour = refined_tour
                    context.cost = new_cost if new_cost is not None else old_cost

                    # Merge into the ledger if active in the simulation context
                    incoming_ctx = getattr(context, "search_context", None)
                    if incoming_ctx is not None:
                        from logic.src.interfaces.context.search_context import SearchPhase, merge_context

                        context.search_context = merge_context(
                            incoming_ctx,
                            phase=SearchPhase.IMPROVEMENT,
                            improvement_metrics=metrics,
                        )

            except Exception as e:
                logger.warning(f"Route improvement skipped due to error: {e}")
