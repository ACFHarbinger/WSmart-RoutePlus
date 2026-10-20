"""

Variant extraction lives in ``config_variants``; this module keeps ``expand_policy_configs`` and re-exports those names.
Configuration loader for Test Engine.

Attributes:
    expand_policy_configs: Expands policy names into full configuration paths and variants.
    _resolve_policy_cfg_path: Resolve the YAML configuration path for a policy.
    _extract_variants: Extract policy variants from its configuration.
    _find_inner_config: Find the list of configurations/variants within a policy config.
    _parse_inner_components: Parse the inner components (MS and PP) from a policy configuration.
    _apply_ms_override: Apply the multi-start (MS) override to a policy configuration.
    _clean_id: Clean an identifier string by removing a prefix and replacing underscores with hyphens.

Example:
    >>> from logic.src.pipeline.features.test.config import expand_policy_configs
    >>> config = Config()
    >>> expand_policy_configs(config)
"""

import copy
from typing import Any, Dict, List, Optional, Set, Tuple

from omegaconf import OmegaConf

from logic.src.configs import Config
from logic.src.utils.configs.config_loader import load_config

from .config_variants import (
    _apply_overrides as _apply_overrides,
)
from .config_variants import (
    _clean_id as _clean_id,
)
from .config_variants import (
    _expand_dict_ms_list as _expand_dict_ms_list,
)
from .config_variants import (
    _extract_variants as _extract_variants,
)
from .config_variants import (
    _find_inner_config as _find_inner_config,
)
from .config_variants import (
    _parse_inner_components as _parse_inner_components,
)
from .config_variants import (
    _resolve_policy_cfg_path as _resolve_policy_cfg_path,
)


def expand_policy_configs(cfg: Config) -> None:  # noqa: C901
    """
    Expands policy names into full configuration paths and variants.

    Populates ``cfg.sim.full_policies`` with expanded policy names and
    ``cfg.sim.config_path`` with their loaded configuration dicts.

    Args:
        cfg: Root configuration object. Reads from ``cfg.sim.policies``
            and ``cfg.sim.data_distribution``.
    """
    sim = cfg.sim
    policies: List[str] = []

    config_path: Dict[str, Any] = dict(sim.config_path) if sim.config_path else {}
    dist_suffix = f"_{sim.data_distribution}"
    resolved_policies = OmegaConf.to_container(sim.policies, resolve=True)
    if not isinstance(resolved_policies, list):
        resolved_policies = [resolved_policies]

    for item in resolved_policies:
        if isinstance(item, dict) and len(item) == 1:
            pol_name = list(item.keys())[0]
            custom_overrides = item[pol_name]
        else:
            pol_name = str(item)
            custom_overrides = None

        cfg_path = _resolve_policy_cfg_path(pol_name)
        variants, variant_name = _extract_variants(pol_name, cfg_path)
        base_overrides = custom_overrides
        caller_selection = _selection_keys_set_by(base_overrides)
        # A caller-supplied packing replaces the yaml fan-out. Keeping
        # every yaml variant would store the same selection under several names.
        if "packing" in caller_selection:
            variants = _collapse_selection_variants(variants)

        for prefix, suffix, custom_cfg in variants:
            custom_overrides = copy.deepcopy(base_overrides)
            middle_name = pol_name.replace("policy_", "")
            # Only append variant name if it's not already in the middle_name
            if variant_name and variant_name.lower() != "default" and variant_name.lower() not in middle_name.lower():
                middle_name = f"{middle_name}_{variant_name}"

            # Prevent doubling of prefixes if the policy name is already expanded
            full_name = pol_name if prefix and prefix in pol_name else f"{prefix}{middle_name}{suffix}"

            if not full_name.endswith(dist_suffix):
                full_name = f"{full_name}{dist_suffix}"

            # If custom_overrides is already a fully-resolved variant config (contains
            # top-level policy-level keys such as engine/packing/route_improvement),
            # it IS the specific variant — store it directly rather than loading the full YAML
            # and merging.  The full-YAML merge path leaves sibling variants (e.g. og_a/og_b)
            # inside the stored config, which causes find_policy_keys to discover the wrong
            # packing when it recurses over all sub-trees.
            _override_keys: set = set(custom_overrides.keys()) if isinstance(custom_overrides, dict) else set()
            _is_resolved_variant = bool(
                _override_keys & {"engine", "packing", "route_improvement", "time_limit", "params"}
            )
            if custom_overrides and _is_resolved_variant:
                # Use the resolved variant config directly, wrapped under the policy name key
                final_cfg = {pol_name: copy.deepcopy(dict(custom_overrides))}
            else:
                final_cfg = copy.deepcopy(custom_cfg or cfg_path)
                if isinstance(final_cfg, str):
                    # Always load the config if it's just a path
                    final_cfg = load_config(final_cfg)

                if custom_overrides and final_cfg and isinstance(final_cfg, dict):
                    # Special handling for single-key policy configs
                    pol_key = list(final_cfg.keys())[0] if len(final_cfg) == 1 else None
                    if pol_key and pol_key == pol_name:
                        # Ensure both are flattened if they are lists (OmegaConf list-of-dicts style)
                        target = final_cfg[pol_key]
                        if isinstance(target, (list, tuple)) or (
                            hasattr(target, "__iter__") and not hasattr(target, "update")
                        ):
                            merged_target = {}
                            for item in target:
                                if hasattr(item, "items"):
                                    merged_target.update(item)
                            final_cfg[pol_key] = merged_target

                        if isinstance(custom_overrides, (list, tuple)) or (
                            hasattr(custom_overrides, "__iter__") and not hasattr(custom_overrides, "update")
                        ):
                            merged_overrides = {}
                            for item in custom_overrides:
                                if hasattr(item, "items"):
                                    merged_overrides.update(item)
                            custom_overrides = merged_overrides

                        final_cfg[pol_key].update(custom_overrides)
                    else:
                        if hasattr(final_cfg, "update"):
                            # Handle case where custom_overrides might be a list here too
                            if isinstance(custom_overrides, (list, tuple)) or (
                                hasattr(custom_overrides, "__iter__") and not hasattr(custom_overrides, "update")
                            ):
                                merged_overrides = {}
                                for item in custom_overrides:
                                    if hasattr(item, "items"):
                                        merged_overrides.update(item)
                                custom_overrides = merged_overrides
                            final_cfg.update(custom_overrides)

            # The flattener keeps a top-level packing and ignores the same
            # key inside a nested custom list. Copy the caller's value onto every
            # occurrence so the running policy sees that selection, not the yaml variant.
            _apply_caller_selection(final_cfg, base_overrides)
            # Pin this variant's packing / acceptance criteria everywhere in the
            # stored config (including nested 'custom' lists) so naming and the daily
            # actions see exactly one variant instead of the unexpanded {file: [all]} form.
            # Keys the caller already set are left alone.
            _pin_variant_selection(final_cfg, custom_cfg, protected=caller_selection)

            policies.append(full_name)
            config_path[full_name] = final_cfg

    sim.full_policies = policies
    sim.config_path = config_path


_SELECTION_KEYS = ("packing", "acceptance_criteria")


def _is_mapping(node: Any) -> bool:
    """Return whether ``node`` is a dict-like config node."""
    return isinstance(node, dict) or (hasattr(node, "items") and not isinstance(node, (str, bytes)))


def _is_sequence(node: Any) -> bool:
    """Return whether ``node`` is a non-string sequence of config nodes."""
    return isinstance(node, (list, tuple)) or (
        hasattr(node, "__iter__") and not isinstance(node, (str, bytes)) and not hasattr(node, "items")
    )


def _selection_keys_set_by(node: Any) -> Set[str]:
    """Names among the pinned selection keys that ``node`` already sets.

    Args:
        node: Custom override tree from a ``sim.policies`` entry.

    Returns:
        Subset of ``packing`` and ``acceptance_criteria`` present in ``node``.
    """
    found: Set[str] = set()

    def _walk(obj: Any) -> None:
        if _is_mapping(obj):
            for key, value in dict(obj).items():
                if key in _SELECTION_KEYS:
                    found.add(str(key))
                _walk(value)
        elif _is_sequence(obj):
            for item in obj:
                _walk(item)

    if node is not None:
        _walk(node)
    return found


def _first_keyed_value(node: Any, key: str) -> Any:
    """Return the first value stored under ``key`` in a nested config.

    Args:
        node: Config tree to search.
        key: Key to look up.

    Returns:
        A deep copy of the first matching value, or ``None`` when absent.
    """
    found: List[Any] = []

    def _walk(obj: Any) -> None:
        if found:
            return
        if _is_mapping(obj):
            data = dict(obj)
            if key in data:
                found.append(copy.deepcopy(data[key]))
                return
            for value in data.values():
                _walk(value)
        elif _is_sequence(obj):
            for item in obj:
                _walk(item)

    _walk(node)
    return found[0] if found else None


def _overwrite_key(node: Any, key: str, value: Any) -> None:
    """Replace every occurrence of ``key`` in ``node`` with ``value``.

    Args:
        node: Config tree updated in place.
        key: Key to replace.
        value: Value copied onto each occurrence.
    """

    def _walk(obj: Any) -> None:
        if _is_mapping(obj):
            if key in obj:
                obj[key] = copy.deepcopy(value)
            for child in list(obj.values()):
                _walk(child)
        elif _is_sequence(obj):
            for item in obj:
                _walk(item)

    _walk(node)


def _apply_caller_selection(final_cfg: Any, overrides: Any) -> None:
    """Copy selection keys the caller set onto every matching key in ``final_cfg``.

    Args:
        final_cfg: Stored policy configuration, updated in place.
        overrides: Custom override tree from the ``sim.policies`` entry.
    """
    for key in _selection_keys_set_by(overrides):
        value = _first_keyed_value(overrides, key)
        # Presence was established above; null explicitly disables the yaml selection.
        _overwrite_key(final_cfg, key, value)


def _collapse_selection_variants(
    variants: List[Tuple[str, str, Any]],
) -> List[Tuple[str, str, Any]]:
    """Keep one body per route/acceptance suffix when the caller set the selection.

    The yaml's packing prefix is dropped so the stored name does not
    claim a variant the caller replaced.

    Args:
        variants: ``(prefix, suffix, config)`` tuples from :func:`_extract_variants`.

    Returns:
        Variants with a single entry for each suffix and an empty selection prefix.
    """
    collapsed: List[Tuple[str, str, Any]] = []
    seen: Set[str] = set()
    for _prefix, suffix, custom_cfg in variants:
        if suffix in seen:
            continue
        seen.add(suffix)
        collapsed.append(("", suffix, custom_cfg))
    return collapsed


def _pin_variant_selection(obj: Any, var_cfg: Any, protected: Optional[Set[str]] = None) -> None:
    """Recursively overwrite selection/acceptance entries with the variant-specific ones.

    ``var_cfg`` (from :func:`_extract_variants`) carries ``packing`` /
    ``acceptance_criteria`` already reduced to a single ``[{file: variant}]`` entry.
    Keys in ``protected`` were set by the custom policy entry and are not replaced.

    Args:
        obj: Policy configuration (nested dicts / lists) to update in place.
        var_cfg: Variant configuration providing the pinned values.
        protected: Selection keys the caller already set.
    """
    if not isinstance(var_cfg, dict):
        return
    pinned = {k: var_cfg[k] for k in _SELECTION_KEYS if var_cfg.get(k)}
    if protected:
        for key in protected:
            pinned.pop(key, None)
    if not pinned:
        return

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in pinned.items():
                if key in node:
                    node[key] = copy.deepcopy(value)
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(obj)







