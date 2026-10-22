"""
Policy Summary Callback.

Display a summary of the policies that will be run in the simulation.

Attributes:
    PolicySummaryCallback: Callback to display a summary of the policies.

Example:
    >>> from logic.src.configs import Config
    >>> from logic.src.pipeline.callbacks.simulation.policy_summary import PolicySummaryCallback
    >>> callback = PolicySummaryCallback()
    >>> callback.display(cfg)
"""

import os
import sys
from typing import Any, Dict, Iterable, List, cast

from rich import box
from rich.console import Console
from rich.table import Table

from logic.src.configs import Config, PackingConfig
from logic.src.constants import CONFIGS_DIR
from logic.src.interfaces import ITraversable
from logic.src.pipeline.simulations.actions.policy_config import _as_policy_mapping
from logic.src.tracking.logging.logger_writer import LoggerWriter
from logic.src.utils.configs.config_loader import load_config
from logic.src.utils.configs.policy_files import policy_file


class PolicySummaryCallback:
    """
    Callback to print a detailed summary of the policies.

    Attributes:
        None
    """

    def display(self, cfg: Config) -> None:
        """
        Display the policy summary table.

        Args:
            cfg: Root Config with ``cfg.sim.full_policies`` and
                ``cfg.sim.config_path``.
        """
        main_console_out = sys.stdout
        if isinstance(main_console_out, LoggerWriter):
            main_console_out = main_console_out.terminal

        console = Console(file=main_console_out)
        table = Table(
            show_header=True,
            header_style="bold magenta",
            title="Simulation Policies Summary",
            box=box.HEAVY_EDGE,
            padding=(0, 1),
            expand=False,
        )

        table.add_column("Idx", justify="right", style="dim")
        table.add_column("Policy Name", style="bold cyan")
        table.add_column("Engine", style="green")
        table.add_column("Selection Strategy", style="yellow")
        table.add_column("Route Improvement", style="blue")
        table.add_column("Acceptance", style="red")

        from logic.src.pipeline.simulations.naming import resolve_policy_display_name

        policies: List[str] = cfg.sim.full_policies or []
        config_paths: Dict[str, Any] = dict(cfg.sim.config_path) if cfg.sim.config_path else {}

        for idx, policy_name in enumerate(policies):
            # Resolve config
            config = config_paths.get(policy_name, {})
            if isinstance(config, str):
                try:
                    config = load_config(config)
                except Exception:
                    config = {}

            # Resolve the full descriptive display name (includes ms prefix + ri suffix)
            try:
                _, display_name = resolve_policy_display_name(policy_name, cfg.sim)
            except Exception:
                display_name = policy_name

            # Extract details
            engine = self._extract_engine(policy_name, config)
            selection = self._extract_selection(config)
            route_imp = self._extract_route_improvement(config)
            acceptance = self._extract_acceptance_criterion(config)

            table.add_row(str(idx + 1), display_name, engine, selection, route_imp, acceptance)

        console.print(table)
        console.print("\n")

    def _extract_engine(self, policy_name: str, config: Dict[str, Any]) -> str:
        """Extract the engine name dynamically.

        Args:
            policy_name: Name of the policy.
            config: Configuration of the policy.

        Returns:
            The engine name.
        """
        raw_cfg = config
        solver_key = None

        # 1. Try to find engine in 'policy' sub-config
        flat_cfg = _as_policy_mapping(raw_cfg)
        policy_cfg = flat_cfg.get("policy", {})
        if isinstance(policy_cfg, ITraversable):
            solver_key = policy_cfg.get("type") or policy_cfg.get("solver") or policy_cfg.get("engine")
        elif isinstance(policy_cfg, str):
            solver_key = policy_cfg

        # 2. Check top-level keys for explicit type/solver/engine
        if not solver_key:
            solver_key = flat_cfg.get("engine") or flat_cfg.get("solver") or flat_cfg.get("type")

        # 3. Default to CUSTOM
        if not solver_key:
            solver_key = "CUSTOM"

        return str(solver_key).replace("_", "-").upper()

    def _extract_selection(self, config: Dict[str, Any]) -> str:
        """Extract selection strategy details.

        Args:
            config: Configuration of the policy.

        Returns:
            The selection strategy.
        """
        raw_cfg = config
        flat_cfg = _as_policy_mapping(raw_cfg)
        config_mandatory = flat_cfg.get("packing")

        strategies = []

        items = []
        if isinstance(config_mandatory, PackingConfig):
            items = [config_mandatory]
        elif isinstance(config_mandatory, (list, tuple)):
            items = list(config_mandatory)
        elif not isinstance(config_mandatory, (str, dict, type(None))) and hasattr(config_mandatory, "__iter__"):
            # config_mandatory is Iterable but not str/dict/None
            items = list(cast(Iterable[Any], config_mandatory))
        elif config_mandatory is not None:
            items = [config_mandatory]

        for item in items:
            name, params = self._parse_selection_item(item)
            strategies.append("{}{}".format(name.lower(), ":" + params if params else ""))

        return ", ".join(strategies) if strategies else "None"

    def _parse_selection_item(self, item: Any) -> tuple[str, str]:  # noqa: C901
        """Parse a single selection item into name and params.

        Args:
            item: Description of item.

        Returns:
            Description of return value.
        """
        name: str = "Unknown"
        params: str = ""

        item_obj: object = item
        if isinstance(item_obj, PackingConfig):
            name = str(item_obj.strategy)
            params = self._parse_mandatory_config_params(item_obj)

        elif isinstance(item_obj, ITraversable):
            # Plain Python dicts also satisfy ITraversable (runtime_checkable Protocol).
            # Check for file-path dict format FIRST: {"other/pk_*.yaml": variant}
            if len(item_obj) == 1:
                first_key = str(next(iter(item_obj.keys())))
                if first_key.endswith(".yaml") or first_key.endswith(".xml"):
                    raw_val = item_obj[first_key]
                    # Unwrap OmegaConf ListConfig or plain list
                    if hasattr(raw_val, "__iter__") and not isinstance(raw_val, str):
                        try:
                            v_list = list(raw_val)
                            raw_val = v_list[0] if v_list else ""
                        except Exception:
                            raw_val = ""
                    variant_val = str(raw_val) if raw_val is not None else ""
                    fpath = str(policy_file(first_key))
                    try:
                        cfg = load_config(fpath)
                        if variant_val and variant_val in cfg:
                            variant_cfg = cfg[variant_val]
                            if hasattr(variant_cfg, "get") and variant_cfg.get("strategy"):
                                name = str(variant_cfg.get("strategy"))
                                params = self._parse_traversable_params(variant_cfg)
                            else:
                                name = variant_val
                        else:
                            base = os.path.basename(first_key)
                            for p in ["ms_", "ri_", "ac_", ".yaml", ".xml"]:
                                base = base.replace(p, "")
                            name = variant_val if (variant_val and variant_val != "default") else base
                    except Exception:
                        base = os.path.basename(first_key)
                        for p in ["ms_", "ri_", "ac_", ".yaml", ".xml"]:
                            base = base.replace(p, "")
                        name = variant_val if variant_val else base
                    return name, params

            if "strategy" in item_obj:
                name = str(item_obj.get("strategy") or "Unknown")
                params = self._parse_traversable_params(item_obj)
            else:
                item_keys = list(item_obj.keys())
                if item_keys:
                    name = str(item_keys[0])
                    val = item_obj[name]
                    if isinstance(val, ITraversable):
                        params = self._parse_traversable_params(cast(Any, val))

        elif isinstance(item_obj, str):
            name = item_obj
            # Strip 'other/pk_', 'other/ri_' and '.yaml'
            if name.startswith("other/pk_"):
                name = name[len("other/pk_") :]
            if name.startswith("other/ri_"):
                name = name[len("other/ri_") :]
            if name.endswith(".yaml"):
                name = name[: -len(".yaml")]

        return name, params

    def _parse_mandatory_config_params(self, config: PackingConfig) -> str:
        """Extract formatted parameters from a PackingConfig object.

        Args:
            config: Configuration of the policy.

        Returns:
            Formatted parameters for the selection strategy.
        """
        strategy = config.strategy
        if strategy == "regular":
            return f"(freq={config.regular.frequency})"
        if strategy == "revenue":
            return f"(thresh={config.revenue.revenue_threshold})"
        if strategy == "service_level":
            return f"(cf={config.service_level.confidence_factor})"
        if strategy == "deadline":
            return f"(horizon={config.deadline.horizon_days}, t={config.deadline.threshold})"
        if strategy == "multi_day_prob":
            return f"(horizon={config.multi_day_prob.horizon_days}, t={config.multi_day_prob.threshold})"
        if strategy == "last_minute":
            return f"(t={config.last_minute.threshold})"
        if strategy == "lookahead":
            return f"(t={config.lookahead.threshold})"
        if strategy == "pareto_front":
            return f"(t={config.pareto_front.threshold})"
        if strategy == "profit_per_km":
            return f"(t={config.profit_per_km.threshold})"
        if strategy == "stochastic_regret":
            return f"(t={config.stochastic_regret.threshold})"
        if strategy == "spatial_synergy":
            s = config.spatial_synergy
            return f"(crit={s.critical_threshold}, syn={s.synergy_threshold}, r={s.radius})"
        if strategy == "mip_knapsack":
            return f"(v={config.mip_knapsack.n_vehicles}, p={config.mip_knapsack.overflow_penalty_frac})"
        return ""

    def _parse_traversable_params(self, item: ITraversable) -> str:
        """Extract formatted parameters from an ITraversable (Config/Dict) object.

        Args:
            item: Configuration of the policy.

        Returns:
            Formatted parameters for the selection strategy.
        """
        if "threshold" in item:
            return f"(t={item['threshold']})"
        if "horizon_days" in item:
            return f"(h={item['horizon_days']})"
        if "frequency" in item:
            return f"(f={item['frequency']})"
        if "critical_threshold" in item:
            return f"(c={item['critical_threshold']}, s={item.get('synergy_threshold')})"
        return ""

    def _extract_route_improvement(self, config: Dict[str, Any]) -> str:  # noqa: C901
        """Extract route improvement steps.

        Args:
            config: Configuration of the policy.

        Returns:
            Route improvement steps.
        """
        raw_cfg = config
        flat_cfg = _as_policy_mapping(raw_cfg)
        pp = flat_cfg.get("route_improvement")

        if not pp:
            return "None"

        steps = []
        _is_sequence = isinstance(pp, (list, tuple)) or (
            hasattr(pp, "__iter__") and not isinstance(pp, str) and not hasattr(pp, "items")
        )
        items = list(pp) if _is_sequence else [pp]
        for item in items:
            item_obj: object = item
            # Handle new dict format: {"other/route_improvement/ri_ftsp.yaml": "default"} or {"other/route_improvement/ri_ftsp.yaml": ["default"]}
            if (isinstance(item_obj, dict) or hasattr(item_obj, "items")) and not isinstance(item_obj, str):
                item_dict = dict(item_obj.items()) if hasattr(item_obj, "items") else dict(item_obj)
                if len(item_dict) == 1:
                    file_key, variant_val = next(iter(item_dict.items()))
                    file_key = str(file_key)
                    if file_key.endswith(".yaml") or file_key.endswith(".xml"):
                        if hasattr(variant_val, "__iter__") and not isinstance(variant_val, str):
                            try:
                                v_list = list(variant_val)
                                variant_val = v_list[0] if v_list else None
                            except Exception:
                                pass
                        variant_val = str(variant_val) if variant_val else ""
                        fpath = str(policy_file(file_key))
                        if os.path.exists(fpath):
                            try:
                                ri_cfg = load_config(fpath)
                                if variant_val and variant_val in ri_cfg:
                                    ri_cfg = ri_cfg[variant_val]
                                if "methods" in ri_cfg:
                                    steps.extend(ri_cfg["methods"])
                                    continue
                                else:
                                    if "config" in ri_cfg and len(ri_cfg) == 1:
                                        ri_cfg = ri_cfg["config"]
                                    steps.extend(list(ri_cfg.keys()))
                                    continue
                            except Exception:
                                pass
                        base = os.path.basename(file_key)
                        for p in ["ri_", "ms_", "ac_", ".yaml", ".xml"]:
                            base = base.replace(p, "")
                        if variant_val and variant_val != "default" and variant_val != base:
                            base = f"{base}_{variant_val}"
                        steps.append(base)
                        continue
                steps.append(str(next(iter(item_dict.keys()))))
            elif isinstance(item_obj, str):
                if item_obj.endswith(".yaml") or item_obj.endswith(".xml"):
                    fpath = str(policy_file(item_obj))
                    if os.path.exists(fpath):
                        try:
                            ri_cfg = load_config(fpath)
                            if "methods" in ri_cfg:
                                steps.extend(ri_cfg["methods"])
                                continue
                            else:
                                if "config" in ri_cfg and len(ri_cfg) == 1:
                                    ri_cfg = ri_cfg["config"]
                                steps.extend(list(ri_cfg.keys()))
                                continue
                        except Exception:
                            pass

                # Fallback: strip path and extension if file not found or parse failed
                name = item_obj
                if name.startswith("other/ri_"):
                    name = name[len("other/ri_") :]
                elif name.startswith("other/"):
                    name = name[len("other/") :]
                if name.endswith(".yaml"):
                    name = name[: -len(".yaml")]
                elif name.endswith(".xml"):
                    name = name[: -len(".xml")]
                steps.append(name)

        return ", ".join(steps) if steps else "None"

    def _extract_acceptance_criterion(self, config: Dict[str, Any]) -> str:  # noqa: C901
        """Extract the acceptance criterion method.

        Args:
            config: Configuration of the policy.

        Returns:
            The acceptance criterion method.
        """
        raw_cfg = config
        flat_cfg = _as_policy_mapping(raw_cfg)

        # Try both keys, prefer acceptance_criteria
        ac = flat_cfg.get("acceptance_criteria") or flat_cfg.get("acceptance_criterion")

        if not ac:
            return "None"

        steps = []
        _is_ac_sequence = isinstance(ac, (list, tuple)) or (
            hasattr(ac, "__iter__") and not isinstance(ac, str) and not hasattr(ac, "items")
        )
        items = list(ac) if _is_ac_sequence else [ac]
        for item in items:
            item_obj: object = item
            # Handle new dict format: {"other/acceptance_criteria/ac_oi.yaml": "oi"} or {"other/acceptance_criteria/ac_oi.yaml": ["oi"]}
            if (isinstance(item_obj, dict) or hasattr(item_obj, "items")) and not isinstance(item_obj, str):
                item_dict = dict(item_obj.items()) if hasattr(item_obj, "items") else dict(item_obj)
                if len(item_dict) == 1:
                    file_key, variant_val = next(iter(item_dict.items()))
                    file_key = str(file_key)
                    if file_key.endswith(".yaml") or file_key.endswith(".xml"):
                        if hasattr(variant_val, "__iter__") and not isinstance(variant_val, str):
                            try:
                                v_list = list(variant_val)
                                variant_val = v_list[0] if v_list else None
                            except Exception:
                                pass
                        variant_val = str(variant_val) if variant_val else ""
                        fpath = str(policy_file(file_key))
                        if os.path.exists(fpath):
                            try:
                                ac_cfg = load_config(fpath)
                                if variant_val and variant_val in ac_cfg:
                                    ac_cfg = ac_cfg[variant_val]
                                method = ac_cfg.get("method") if hasattr(ac_cfg, "get") else None
                                if method:
                                    steps.append(str(method).upper())
                                    continue
                            except Exception:
                                pass
                        base = os.path.basename(file_key)
                        for p in ["ac_", "ms_", "ri_", ".yaml", ".xml"]:
                            base = base.replace(p, "")
                        label = variant_val if (variant_val and variant_val != "default") else base
                        steps.append(label.upper())
                        continue
                steps.append(str(next(iter(item_dict.values()), "Unknown")).upper())
            elif isinstance(item_obj, str):
                name = item_obj
                if name.startswith("other/ac_"):
                    name = name[len("other/ac_") :]
                elif name.startswith("other/"):
                    name = name[len("other/") :]
                if name.endswith(".yaml"):
                    name = name[: -len(".yaml")]
                elif name.endswith(".xml"):
                    name = name[: -len(".xml")]
                steps.append(name.upper())
            elif isinstance(item_obj, ITraversable):
                steps.append(str(item_obj.get("method") or "Unknown").upper())
            else:
                steps.append(str(item_obj).upper())

        return ", ".join(steps) if steps else "None"
