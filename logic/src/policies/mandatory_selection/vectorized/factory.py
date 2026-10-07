"""Selection strategy factory.

This module provides factory functions to create vectorized bin selection
strategies from configuration objects or by explicit name.

Attributes:
    get_vectorized_selector: Function to instantiate selectors by name.
    create_selector_from_config: Function to instantiate selectors from configuration.

Example:
    >>> selector = get_vectorized_selector("last_minute", threshold=80)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, cast

from logic.src.interfaces import ITraversable

from .base import VectorizedSelector
from .combined import CombinedSelector
from .last_minute import LastMinuteSelector
from .lookahead import LookaheadSelector
from .manager import ManagerSelector
from .regular import RegularSelector
from .revenue import RevenueSelector
from .service_level import ServiceLevelSelector


def create_selector_from_config(cfg: Any) -> Optional[VectorizedSelector]:
    """Create a vectorized selector from a MandatorySelectionConfig or dict.

    Args:
        cfg: Configuration object containing 'strategy' and 'params'.

    Returns:
        Optional[VectorizedSelector]: The instantiated selector or None.
    """
    if cfg is None:
        return None

    strategy = _get_strategy(cfg)
    if not strategy or strategy.lower() == "none":
        return None

    strategy = strategy.lower()
    params = _get_params(cfg)

    if strategy == "combined":
        return _create_combined_selector(params)
    if strategy == "manager":
        return _create_manager_selector(params)

    strategy_params = _get_strategy_params(strategy, params)
    return get_vectorized_selector(strategy, **strategy_params)


def _get_strategy(cfg: object) -> Optional[str]:
    """Extract strategy name from config.

    Args:
        cfg: The configuration object.

    Returns:
        Optional[str]: Strategy name if found, else None.
    """
    if hasattr(cfg, "strategy"):
        return str(cfg.strategy)
    if isinstance(cfg, ITraversable):
        return cast(Optional[str], cfg.get("strategy"))
    return None


def _get_params(cfg: object) -> Dict[str, Any]:
    """Extract parameters from config, excluding strategy.

    Args:
        cfg: The configuration object.

    Returns:
        Dict[str, Any]: Dictionary of strategy parameters.
    """
    # OmegaConf DictConfig: use OmegaConf.to_container for a clean plain dict.
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(cfg, DictConfig):
            raw = OmegaConf.to_container(cfg, resolve=True)
            if isinstance(raw, dict):
                return cast(Dict[str, Any], {k: v for k, v in raw.items() if k != "strategy" and v is not None})
    except ImportError:
        pass

    # Dataclasses / plain objects with .items() (ITraversable)
    if isinstance(cfg, ITraversable):
        return {k: v for k, v in cfg.items() if k != "strategy" and v is not None}

    # Last resort: __dict__ (may work for simple dataclass-like objects)
    if hasattr(cfg, "__dict__"):
        return {k: v for k, v in vars(cfg).items() if k != "strategy" and v is not None}

    return {}


def _create_combined_selector(params: Dict[str, Any]) -> Optional[VectorizedSelector]:
    """Helper to create a CombinedSelector.

    Args:
        params: Combined selector parameters.

    Returns:
        Optional[VectorizedSelector]: Instantiated CombinedSelector or None.
    """
    combined_configs = params.get("combined_strategies", [])
    if not combined_configs:
        return None
    selectors = []
    for sub_cfg in combined_configs:
        sub_selector = create_selector_from_config(sub_cfg)
        if sub_selector is not None:
            selectors.append(sub_selector)
    if not selectors:
        return None
    return CombinedSelector(selectors, logic=params.get("logic", "or"))


def _create_manager_selector(params: Dict[str, Any]) -> VectorizedSelector:
    """Helper to create a ManagerSelector.

    Args:
        params: Neural manager parameters.

    Returns:
        VectorizedSelector: Instantiated ManagerSelector.
    """
    manager_config = {
        "hidden_dim": params.get("hidden_dim", 128),
        "lstm_hidden": params.get("lstm_hidden", 64),
        "history_length": params.get("history_length", 10),
        "critical_threshold": params.get("critical_threshold", 0.9),
    }
    selector = ManagerSelector(
        manager_config=manager_config,
        threshold=params.get("threshold", 0.5),
        device=params.get("device", "cuda"),
    )
    if params.get("manager_weights"):
        selector.load_weights(params["manager_weights"])
    return selector


def _get_strategy_params(strategy: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Map strategy name to its specific parameters.

    Looks first in the nested sub-config (e.g. params["last_minute"]["threshold"])
    then falls back to top-level params, then to hard-coded defaults.

    Args:
        strategy: name of the strategy.
        params: raw parameter dictionary (may contain nested sub-configs).

    Returns:
        Dict[str, Any]: mapped parameters for the selector constructor.

    Raises:
        ValueError: if strategy is unknown.
    """
    # Pull out the nested sub-config for this strategy, if present.
    nested = params.get(strategy, {})
    if nested is None:
        nested = {}
    # Normalise DictConfig / OmegaConf → plain dict so .get() behaves predictably.
    if hasattr(nested, "items") and not isinstance(nested, dict):
        nested = {k: v for k, v in nested.items()}

    def _get(key: str, default: Any) -> Any:
        """Return nested[key] → params[key] → default."""
        if key in nested:
            return nested[key]
        return params.get(key, default)

    mappings: Dict[str, Any] = {
        "last_minute": {"threshold": _get("threshold", 70.0)},  # percent
        "regular": {"frequency": _get("frequency", 3)},
        "lookahead": {"current_collection_day": _get("current_collection_day", 0)},
        "revenue": {
            "revenue_kg": _get("revenue_kg", 1.0),
            "bin_capacity": _get("bin_capacity", 1.0),
            "threshold": _get("threshold", _get("revenue_threshold", 0.0)),
        },
        "service_level": {
            "confidence_factor": _get("confidence_factor", _get("threshold", 1.0)),
            "horizon_days": _get("horizon_days", 1),
        },
    }
    if strategy not in mappings:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(mappings.keys())} + manager, combined")
    return mappings[strategy]


def get_vectorized_selector(name: str, **kwargs: Any) -> VectorizedSelector:
    """Create a vectorized selector by name.

    Args:
        name: Name of the selection strategy.
        kwargs: Strategy-specific parameters.

    Returns:
        VectorizedSelector: The instantiated selector.

    Raises:
        ValueError: If the selector name is unknown.
    """
    selectors = {
        "last_minute": LastMinuteSelector,
        "regular": RegularSelector,
        "lookahead": LookaheadSelector,
        "revenue": RevenueSelector,
        "service_level": ServiceLevelSelector,
        "manager": ManagerSelector,
    }

    if name.lower() not in selectors:
        raise ValueError(f"Unknown selector: {name}. Available: {list(selectors.keys())}")

    selector_cls = cast(Any, selectors[name.lower()])
    return cast(VectorizedSelector, selector_cls(**kwargs))
