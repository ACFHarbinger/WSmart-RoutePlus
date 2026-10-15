"""Factory for packing algorithms. Construction is ``PACKING.create``."""

import logging
from dataclasses import asdict
from typing import Any

from logic.src.interfaces.packing import IPackingAlgorithm
from logic.src.utils.registry import PACKING

from .selection_registry import PackingRegistry

logger = logging.getLogger(__name__)

CONFIG_MAPPING = {
    "last_minute": "last_minute",
    "regular": "regular",
    "service_level": "service_level",
    "revenue": "revenue",
    "revenue_threshold": "revenue",
    "lookahead": "lookahead",
    "deadline": "deadline",
    "multi_day_prob": "multi_day_prob",
    "pareto_front": "pareto_front",
    "profit_per_km": "profit_per_km",
    "spatial_synergy": "spatial_synergy",
    "stochastic_regret": "stochastic_regret",
    "combined": "combined",
    "manager": "manager",
    "mip_knapsack": "mip_knapsack",
    "fractional_knapsack": "fractional_knapsack",
    "rollout": "rollout",
    "whittle": "whittle",
    "cvar": "cvar",
    "savings": "savings",
    "set_cover": "set_cover",
    "submodular_greedy": "modular_greedy",
    "supermodular_greedy": "modular_greedy",
    "greedy_routing_heuristic": "modular_greedy",
    "learned": "learned",
    "wasserstein_robust": "wasserstein",
    "dispatcher_thompson": "thompson_dispatcher",
    "dispatcher_portfolio": "portfolio_dispatcher",
    "lagrangian": "lagrangian",
    "filter_and_fan": "filter_and_fan",
    "bernoulli_random": "bernoulli_random",
    "kmeans_sector": "kmeans_sector",
    "staggered_regular": "staggered_regular",
    "fptas_knapsack": "fptas_knapsack",
}


class PackingFactory:
    """Create a packing algorithm by registry name."""

    @staticmethod
    def create_strategy(name: str, **kwargs: Any) -> IPackingAlgorithm:
        if PackingRegistry.get(name) is None:
            raise ValueError(f"Unknown selection strategy: {name}")
        return PACKING.create(name, **kwargs)

    @classmethod
    def create_from_config(cls, config: Any) -> IPackingAlgorithm:
        if config.strategy is None:
            raise ValueError("No strategy specified in PackingConfig")

        strategy_name = config.strategy.lower()
        params = config.params.copy() if hasattr(config, "params") else {}

        attr_name = CONFIG_MAPPING.get(strategy_name)
        if attr_name and hasattr(config, attr_name):
            sub_config = getattr(config, attr_name)
            if hasattr(sub_config, "__dataclass_fields__"):
                sub_params = asdict(sub_config)
                if strategy_name == "regular" and "frequency" in sub_params:
                    sub_params["threshold"] = sub_params.pop("frequency")
                elif strategy_name == "service_level" and "confidence_factor" in sub_params:
                    sub_params["threshold"] = sub_params.pop("confidence_factor")
                elif strategy_name == "learned" and "learned_threshold" in sub_params:
                    sub_params["threshold"] = sub_params.pop("learned_threshold")
                elif strategy_name == "kmeans_sector" and "n_sectors" in sub_params:
                    sub_params["threshold"] = sub_params.pop("n_sectors")
                elif strategy_name == "staggered_regular" and "period" in sub_params:
                    sub_params["threshold"] = sub_params.pop("period")
                params.update(sub_params)
            elif isinstance(sub_config, dict):
                params.update(sub_config)

        if strategy_name == "combined" and "combined_strategies" not in params and "strategies" in params:
            params["combined_strategies"] = params.pop("strategies")

        return PACKING.create(config.strategy, **params)
